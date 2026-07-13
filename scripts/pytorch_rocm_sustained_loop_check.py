"""
TRIDENT — PyTorch/ROCm 技術検証: 反復呼び出し + デバイス→ホスト転送の生存確認

jax-rocm-spike (docs/jax_rocm_spike_20260714.md) で判明した JAX-ROCm のクラッシュは
「2世代目の jax.device_get() (デバイス→ホスト転送)」で HIP_ERROR_InvalidValue が
決定論的に発生するというものだった。これは XLA の HIP グラフキャプチャ機構に
起因する可能性が高く、PyTorch eager モードには同じ機構が存在しないため
同じ失敗が起きるとは限らない — が未検証のため、同じ失敗面(反復呼び出し +
デバイス→ホスト転送)を狙って確認する。

種分化・交叉・世代ループそのものは実装しない(スコープ外) — 集団のミューテーションを
挟んだ順伝播+ホスト転送をN回繰り返すだけの最小テスト。

実行方法:
  ROCPROFILER_REGISTER_ENABLED=0 ROCR_VISIBLE_DEVICES=0 \
    poetry run python scripts/pytorch_rocm_sustained_loop_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pytorch_proto.genome import create_random_genome, mutate_weight
from src.pytorch_proto.network import build_batch_tensors, batched_forward

ITERATIONS = 30
POP_SIZE = 100
NUM_INPUTS, NUM_OUTPUTS = 2, 1
NUM_SAMPLES = 200
MAX_NODES, MAX_CONNS = 80, 200


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    rng = np.random.default_rng(42)
    genomes = [
        create_random_genome(NUM_INPUTS, NUM_OUTPUTS, MAX_NODES, MAX_CONNS, rng)
        for _ in range(POP_SIZE)
    ]
    samples_np = rng.standard_normal((NUM_SAMPLES, NUM_INPUTS)).astype(np.float32)
    x = torch.from_numpy(np.tile(samples_np, (POP_SIZE, 1, 1))).to(device)

    for i in range(1, ITERATIONS + 1):
        # mutate the population between iterations, mirroring a generation loop's
        # changing tensor contents (topology-changing ops kept out of scope)
        for g in genomes:
            mutate_weight(g, rng)
        W, bias, act_codes, valid_mask = build_batch_tensors(genomes, device)

        out = batched_forward(W, bias, act_codes, valid_mask, x, NUM_INPUTS, NUM_OUTPUTS)
        # force a device->host transfer every iteration — this is the exact
        # operation (jax.device_get equivalent) that crashed JAX-ROCm at generation 2
        host_val = out.mean().cpu().item()

        if i == 1 or i % 5 == 0 or i == ITERATIONS:
            print(f"  iter {i:3d}: mean_output={host_val:.6f}")

    print(f"\nOK: {ITERATIONS} iterations of forward + mutate + device->host transfer completed without crash.")


if __name__ == "__main__":
    main()
