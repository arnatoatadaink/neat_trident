"""Cross-checks the batched torch forward pass against an independent, unbatched numpy reference."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.pytorch_proto.genome import Genome, add_connection, add_node, create_random_genome, mutate_weight
from src.pytorch_proto.network import build_batch_tensors, batched_forward

_ACT_FNS = (
    lambda v: v,
    lambda v: 1.0 / (1.0 + np.exp(-v)),
    np.tanh,
    lambda v: np.maximum(v, 0.0),
)


def naive_forward_single(genome: Genome, x: np.ndarray) -> np.ndarray:
    max_nodes = genome.max_nodes
    values = np.zeros(max_nodes, dtype=np.float32)
    values[: genome.num_inputs] = x

    valid = ~np.isnan(genome.nodes[:, 0])
    bias = np.nan_to_num(genome.nodes[:, 1])
    act_codes = np.nan_to_num(genome.nodes[:, 2]).astype(int)

    W = np.zeros((max_nodes, max_nodes), dtype=np.float32)
    valid_conns = genome.conns[~np.isnan(genome.conns[:, 0])]
    for src, dst, weight in valid_conns:
        W[int(dst), int(src)] = weight

    node_positions = np.arange(max_nodes)
    for _ in range(max_nodes):
        pre_act = W @ values + bias
        activated = np.array([_ACT_FNS[c](p) for c, p in zip(act_codes, pre_act)], dtype=np.float32)
        new_values = np.where(node_positions < genome.num_inputs, values, activated)
        new_values = np.where(valid, new_values, 0.0)
        values = new_values.astype(np.float32)

    return values[genome.num_inputs : genome.num_inputs + genome.num_outputs]


def _make_population(rng: np.random.Generator, pop_size: int) -> list[Genome]:
    num_inputs, num_outputs = 3, 2
    max_nodes, max_conns = 20, 40
    genomes = []
    for _ in range(pop_size):
        genome = create_random_genome(num_inputs, num_outputs, max_nodes, max_conns, rng)
        for _ in range(rng.integers(0, 4)):
            add_connection(genome, rng)
        for _ in range(rng.integers(0, 3)):
            add_node(genome, rng)
        mutate_weight(genome, rng)
        genomes.append(genome)
    return genomes


def test_batched_forward_matches_naive_reference():
    rng = np.random.default_rng(0)
    genomes = _make_population(rng, pop_size=8)

    device = torch.device("cpu")
    W, bias, act_codes, valid_mask = build_batch_tensors(genomes, device)

    x_np = rng.standard_normal((genomes[0].num_inputs,)).astype(np.float32)
    x = torch.from_numpy(np.tile(x_np, (len(genomes), 1, 1))).to(device)  # (pop, 1, num_inputs)

    batched_out = batched_forward(
        W, bias, act_codes, valid_mask, x, genomes[0].num_inputs, genomes[0].num_outputs
    )
    batched_out_np = batched_out.squeeze(1).cpu().numpy()

    for g_idx, genome in enumerate(genomes):
        expected = naive_forward_single(genome, x_np)
        np.testing.assert_allclose(batched_out_np[g_idx], expected, rtol=1e-4, atol=1e-5)


def test_batched_forward_handles_varying_topologies_in_one_batch():
    rng = np.random.default_rng(1)
    genomes = _make_population(rng, pop_size=5)
    node_counts = [int((~np.isnan(g.nodes[:, 0])).sum()) for g in genomes]
    assert len(set(node_counts)) > 1, "test setup should produce genomes with differing node counts"

    device = torch.device("cpu")
    W, bias, act_codes, valid_mask = build_batch_tensors(genomes, device)
    x = torch.from_numpy(rng.standard_normal((len(genomes), 1, genomes[0].num_inputs)).astype(np.float32))

    out = batched_forward(W, bias, act_codes, valid_mask, x, genomes[0].num_inputs, genomes[0].num_outputs)
    assert out.shape == (len(genomes), 1, genomes[0].num_outputs)
    assert torch.isfinite(out).all()


def test_build_batch_tensors_rejects_mismatched_max_nodes():
    rng = np.random.default_rng(2)
    g1 = create_random_genome(2, 1, max_nodes=10, max_conns=20, rng=rng)
    g2 = create_random_genome(2, 1, max_nodes=12, max_conns=20, rng=rng)
    with pytest.raises(ValueError):
        build_batch_tensors([g1, g2], torch.device("cpu"))
