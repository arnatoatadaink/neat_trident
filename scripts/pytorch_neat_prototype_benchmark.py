"""
TRIDENT — PyTorch/ROCm 技術検証: バッチ順伝播ベンチマーク

scripts/neat_benchmark.py (JAX/TensorNEAT版) と同一の問題形状 (Spiral 2→1,
VectorNeighbor 32次元) で、src/pytorch_proto の batched_forward のみを計測する。

対象は「GPU律速部分である集団順伝播」のみ。選択・交叉・種分化を含む世代ループは
含まないため、JAX版の gen_time_ms とは厳密には比較できない — 順伝播コストの
上界比較として使う。

実行方法:
  poetry run python scripts/pytorch_neat_prototype_benchmark.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pytorch_proto.genome import Genome, add_connection, add_node, create_random_genome, mutate_weight
from src.pytorch_proto.network import build_batch_tensors, batched_forward

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)


def detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_population(
    rng: np.random.Generator,
    pop_size: int,
    num_inputs: int,
    num_outputs: int,
    max_nodes: int,
    max_conns: int,
) -> list[Genome]:
    genomes = []
    for _ in range(pop_size):
        genome = create_random_genome(num_inputs, num_outputs, max_nodes, max_conns, rng)
        for _ in range(rng.integers(0, 6)):
            add_connection(genome, rng)
        for _ in range(rng.integers(0, 4)):
            add_node(genome, rng)
        mutate_weight(genome, rng)
        genomes.append(genome)
    return genomes


def run_benchmark(
    label: str,
    num_inputs: int,
    num_outputs: int,
    num_samples: int,
    pop_size: int,
    max_nodes: int,
    max_conns: int,
    device: torch.device,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    # margins cover make_population's worst-case mutation budget: up to 5 add_connection (+5 conns)
    # and up to 3 add_node calls (+3 nodes, +6 conns) per genome
    min_nodes = num_inputs + num_outputs + 10
    min_conns = num_inputs * num_outputs + 20
    max_nodes = max(max_nodes, min_nodes)
    max_conns = max(max_conns, min_conns)

    rng = np.random.default_rng(seed)
    genomes = make_population(rng, pop_size, num_inputs, num_outputs, max_nodes, max_conns)
    W, bias, act_codes, valid_mask = build_batch_tensors(genomes, device)

    samples_np = rng.standard_normal((num_samples, num_inputs)).astype(np.float32)
    x = torch.from_numpy(np.tile(samples_np, (pop_size, 1, 1))).to(device)

    # warm-up (kernel launch / any lazy init on the device)
    batched_forward(W, bias, act_codes, valid_mask, x, num_inputs, num_outputs)
    if device.type == "cuda":
        torch.cuda.synchronize()

    times_ms = []
    for _ in range(repeats):
        start = time.perf_counter()
        batched_forward(W, bias, act_codes, valid_mask, x, num_inputs, num_outputs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times_ms.append((time.perf_counter() - start) * 1000)

    mean_nodes = float(np.mean([(~np.isnan(g.nodes[:, 0])).sum() for g in genomes]))
    mean_conns = float(np.mean([(~np.isnan(g.conns[:, 0])).sum() for g in genomes]))

    return {
        "label": label,
        "device": str(device),
        "pop_size": pop_size,
        "num_samples": num_samples,
        "max_nodes": max_nodes,
        "max_conns": max_conns,
        "mean_forward_ms": round(float(np.mean(times_ms)), 3),
        "std_forward_ms": round(float(np.std(times_ms)), 3),
        "mean_nodes": round(mean_nodes, 1),
        "mean_conns": round(mean_conns, 1),
        "repeats": repeats,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pop-size", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = detect_device()
    print(f"device: {device}" + (f" (torch {torch.__version__})" if device.type == "cuda" else ""))
    if device.type != "cuda":
        print("  note: CUDA/ROCm not available on this box — running CPU-only.")

    results = [
        run_benchmark(
            label="Spiral",
            num_inputs=2,
            num_outputs=1,
            num_samples=200,
            pop_size=args.pop_size,
            max_nodes=80,
            max_conns=200,
            device=device,
            repeats=args.repeats,
            seed=args.seed,
        ),
        run_benchmark(
            label="VectorNeighbor",
            num_inputs=32,
            num_outputs=32,
            num_samples=20,
            pop_size=args.pop_size,
            max_nodes=80,
            max_conns=200,
            device=device,
            repeats=args.repeats,
            seed=args.seed,
        ),
    ]

    log_path = LOG_DIR / "pytorch_proto_benchmark.jsonl"
    with open(log_path, "a") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"\n{'label':<16} {'device':<6} {'pop':>5} {'mean_ms':>10} {'std_ms':>8}")
    for r in results:
        print(f"{r['label']:<16} {r['device']:<6} {r['pop_size']:>5} {r['mean_forward_ms']:>10} {r['std_forward_ms']:>8}")
    print(f"\nlogged to {log_path}")
    print(
        "\nnote: this times batched_forward only (no selection/mutation/speciation), "
        "so it is a lower-bound comparison against JAX gen_time_ms in logs/benchmark_*.jsonl, "
        "not an apples-to-apples full-generation figure."
    )


if __name__ == "__main__":
    main()
