"""Padded NEAT genome representation, mirroring TensorNEAT's NaN-padded array convention."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ACT_IDENTITY = 0
ACT_SIGMOID = 1
ACT_TANH = 2
ACT_RELU = 3
_HIDDEN_ACTS = (ACT_SIGMOID, ACT_TANH, ACT_RELU)


@dataclass
class Genome:
    nodes: np.ndarray  # (max_nodes, 3) float32: [index, bias, activation_code], NaN row = unused slot
    conns: np.ndarray  # (max_conns, 3) float32: [in_index, out_index, weight], NaN row = unused slot
    num_inputs: int
    num_outputs: int
    max_nodes: int
    max_conns: int


def _first_free_row(arr: np.ndarray) -> int | None:
    free = np.where(np.isnan(arr[:, 0]))[0]
    return int(free[0]) if free.size > 0 else None


def create_random_genome(
    num_inputs: int,
    num_outputs: int,
    max_nodes: int,
    max_conns: int,
    rng: np.random.Generator,
) -> Genome:
    """Input/output nodes fully connected, no hidden nodes — TensorNEAT DefaultGenome's initial topology."""
    assert max_nodes >= num_inputs + num_outputs
    assert max_conns >= num_inputs * num_outputs

    nodes = np.full((max_nodes, 3), np.nan, dtype=np.float32)
    conns = np.full((max_conns, 3), np.nan, dtype=np.float32)

    for i in range(num_inputs):
        nodes[i] = [i, 0.0, ACT_IDENTITY]
    for j in range(num_outputs):
        out_idx = num_inputs + j
        nodes[out_idx] = [out_idx, rng.normal(0.0, 0.5), rng.choice(_HIDDEN_ACTS)]

    conn_row = 0
    for i in range(num_inputs):
        for j in range(num_outputs):
            conns[conn_row] = [i, num_inputs + j, rng.normal(0.0, 1.0)]
            conn_row += 1

    return Genome(nodes, conns, num_inputs, num_outputs, max_nodes, max_conns)


def mutate_weight(genome: Genome, rng: np.random.Generator, sigma: float = 0.5, prob: float = 0.8) -> None:
    """Perturb each valid connection's weight in place with probability `prob`."""
    valid_rows = np.where(~np.isnan(genome.conns[:, 0]))[0]
    for row in valid_rows:
        if rng.random() < prob:
            genome.conns[row, 2] += rng.normal(0.0, sigma)


def _creates_cycle(genome: Genome, src: int, dst: int) -> bool:
    """True if dst can already reach src — required since network.py's relaxation loop assumes a DAG."""
    adjacency: dict[int, list[int]] = {}
    for row_src, row_dst, _ in genome.conns[~np.isnan(genome.conns[:, 0])]:
        adjacency.setdefault(int(row_src), []).append(int(row_dst))

    stack, seen = [dst], {dst}
    while stack:
        node = stack.pop()
        if node == src:
            return True
        for nxt in adjacency.get(node, []):
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return False


def add_connection(genome: Genome, rng: np.random.Generator) -> bool:
    """Add a random new connection into a non-input node, rejecting any pair that would create a cycle."""
    free_row = _first_free_row(genome.conns)
    if free_row is None:
        return False

    valid_node_idx = np.where(~np.isnan(genome.nodes[:, 0]))[0]
    dst_candidates = valid_node_idx[valid_node_idx >= genome.num_inputs]
    if len(valid_node_idx) < 2 or len(dst_candidates) == 0:
        return False

    existing = {(int(r[0]), int(r[1])) for r in genome.conns if not np.isnan(r[0])}
    for _ in range(10):  # bounded retries, this is a validation prototype not production NEAT
        src = int(rng.choice(valid_node_idx))
        dst = int(rng.choice(dst_candidates))
        if src != dst and (src, dst) not in existing and not _creates_cycle(genome, src, dst):
            genome.conns[free_row] = [src, dst, rng.normal(0.0, 1.0)]
            return True
    return False


def add_node(genome: Genome, rng: np.random.Generator) -> bool:
    """Split an existing connection with a new hidden node; original edge is kept (redundant but harmless)."""
    node_slot = _first_free_row(genome.nodes)
    conn_slots_needed = 2
    free_conn_rows = np.where(np.isnan(genome.conns[:, 0]))[0]
    if node_slot is None or len(free_conn_rows) < conn_slots_needed:
        return False

    valid_conn_rows = np.where(~np.isnan(genome.conns[:, 0]))[0]
    if len(valid_conn_rows) == 0:
        return False

    split_row = int(rng.choice(valid_conn_rows))
    src, dst, weight = genome.conns[split_row]

    genome.nodes[node_slot] = [node_slot, rng.normal(0.0, 0.5), rng.choice(_HIDDEN_ACTS)]
    genome.conns[free_conn_rows[0]] = [src, node_slot, 1.0]
    genome.conns[free_conn_rows[1]] = [node_slot, dst, weight]
    return True
