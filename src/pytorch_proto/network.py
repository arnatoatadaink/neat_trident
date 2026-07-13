"""Batched population forward pass: `max_nodes` rounds of Jacobi-style relaxation instead of
TensorNEAT's topological-sort + fori_loop, since genome.py's mutations guarantee a DAG (so depth
<= max_nodes) and this needs only batched matmul/einsum — no vmap, portable to CUDA and ROCm."""

from __future__ import annotations

import numpy as np
import torch

from src.pytorch_proto.genome import ACT_IDENTITY, ACT_RELU, ACT_SIGMOID, ACT_TANH, Genome


def build_batch_tensors(
    genomes: list[Genome], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (W, bias, act_codes, valid_mask), each with a leading pop_size dim; W[g, i, j] = weight of edge j -> i."""
    max_nodes = genomes[0].max_nodes
    if any(g.max_nodes != max_nodes for g in genomes):
        raise ValueError("all genomes must share max_nodes for batching")

    pop = len(genomes)
    W = np.zeros((pop, max_nodes, max_nodes), dtype=np.float32)
    bias = np.zeros((pop, max_nodes), dtype=np.float32)
    act_codes = np.zeros((pop, max_nodes), dtype=np.int64)
    valid_mask = np.zeros((pop, max_nodes), dtype=bool)

    for g_idx, genome in enumerate(genomes):
        valid_nodes = ~np.isnan(genome.nodes[:, 0])
        valid_mask[g_idx] = valid_nodes
        node_idx = np.where(valid_nodes)[0]
        bias[g_idx, node_idx] = genome.nodes[node_idx, 1]
        act_codes[g_idx, node_idx] = genome.nodes[node_idx, 2].astype(np.int64)

        valid_conns = ~np.isnan(genome.conns[:, 0])
        for src, dst, weight in genome.conns[valid_conns]:
            W[g_idx, int(dst), int(src)] = weight

    return (
        torch.from_numpy(W).to(device),
        torch.from_numpy(bias).to(device),
        torch.from_numpy(act_codes).to(device),
        torch.from_numpy(valid_mask).to(device),
    )


def apply_activation(x: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
    """Dispatch per-element activation via gather instead of data-dependent branching."""
    stacked = torch.stack([x, torch.sigmoid(x), torch.tanh(x), torch.relu(x)], dim=-1)
    # index order must match ACT_* constants
    assert (ACT_IDENTITY, ACT_SIGMOID, ACT_TANH, ACT_RELU) == (0, 1, 2, 3)
    return torch.gather(stacked, -1, codes.unsqueeze(-1)).squeeze(-1)


def batched_forward(
    W: torch.Tensor,
    bias: torch.Tensor,
    act_codes: torch.Tensor,
    valid_mask: torch.Tensor,
    x: torch.Tensor,
    num_inputs: int,
    num_outputs: int,
) -> torch.Tensor:
    """x: (pop, samples, num_inputs) -> output: (pop, samples, num_outputs)."""
    pop, samples, _ = x.shape
    max_nodes = W.shape[1]

    values = torch.zeros(pop, samples, max_nodes, device=x.device, dtype=x.dtype)
    values[:, :, :num_inputs] = x

    input_mask = torch.zeros(max_nodes, dtype=torch.bool, device=x.device)
    input_mask[:num_inputs] = True
    input_mask = input_mask.view(1, 1, -1)
    valid_mask_b = valid_mask.unsqueeze(1)  # (pop, 1, max_nodes)
    act_codes_b = act_codes.unsqueeze(1).expand(-1, samples, -1)
    bias_b = bias.unsqueeze(1)  # (pop, 1, max_nodes)

    for _ in range(max_nodes):
        pre_act = torch.einsum("pij,psj->psi", W, values) + bias_b
        activated = apply_activation(pre_act, act_codes_b)
        new_values = torch.where(input_mask, values, activated)
        new_values = torch.where(valid_mask_b, new_values, 0.0)
        values = new_values

    return values[:, :, num_inputs : num_inputs + num_outputs]
