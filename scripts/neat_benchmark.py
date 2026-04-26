"""
TRIDENT — NEAT ベンチマーク
2種類の問題を CPU / GPU で 30 分間実行し、世代ごとのログを取得する。

問題:
  Problem A: Double Spiral Classification (32入力→1出力, 二値分類)
             トポロジー進化が本質的に有効な古典問題
  Problem B: VectorNeighbor (32次元, corpus=100)
             TRIDENT に直接関連するベクトル近傍探索問題

出力:
  logs/benchmark_cpu_spiral.jsonl
  logs/benchmark_cpu_vector.jsonl
  logs/benchmark_summary.txt

実行方法:
  poetry run python scripts/neat_benchmark.py [--max-seconds 1800]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap

sys.path.insert(0, str(Path(__file__).parent.parent))

from tensorneat.common import State
from tensorneat.genome import DefaultGenome
from tensorneat.algorithm.neat import NEAT
from tensorneat import Pipeline
from tensorneat.problem.base import BaseProblem

# ──────────────────────────────────────────────
# ユーティリティ
# ──────────────────────────────────────────────

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)


def detect_devices() -> dict[str, Any]:
    devices = jax.devices()
    device_strs = [str(d) for d in devices]
    has_gpu = any("gpu" in s.lower() or "cuda" in s.lower() for s in device_strs)
    return {
        "devices": device_strs,
        "has_gpu": has_gpu,
        "backend": jax.default_backend(),
    }


# ──────────────────────────────────────────────
# Problem A: Double Spiral Classification
# ──────────────────────────────────────────────

def make_spiral_data(n_per_class: int = 100, noise: float = 0.2, seed: int = 0):
    """
    2クラスの二重螺旋データを生成する。

    Returns
    -------
    X : (2*n, 2) float32  入力座標 [-1, 1]
    y : (2*n, 1) float32  ラベル {0, 1}
    """
    rng = np.random.default_rng(seed)
    n = n_per_class
    theta = np.linspace(0, 4 * np.pi, n)
    r = np.linspace(0.1, 1.0, n)

    x1 = r * np.cos(theta) + rng.normal(0, noise, n)
    y1 = r * np.sin(theta) + rng.normal(0, noise, n)
    x2 = r * np.cos(theta + np.pi) + rng.normal(0, noise, n)
    y2 = r * np.sin(theta + np.pi) + rng.normal(0, noise, n)

    X = np.stack([
        np.concatenate([x1, x2]),
        np.concatenate([y1, y2])
    ], axis=1).astype(np.float32)
    y = np.concatenate([
        np.zeros(n, dtype=np.float32),
        np.ones(n, dtype=np.float32),
    ]).reshape(-1, 1)

    return X, y


class SpiralProblem(BaseProblem):
    """
    二重螺旋分類問題。

    入力: (2,) 座標
    出力: (1,) 分類スコア (sigmoid > 0.5 で class 1)
    fitness = -(BCE loss)
    """

    jitable = True

    def __init__(self, X: np.ndarray, y: np.ndarray):
        super().__init__()
        self._X = jnp.array(X, dtype=jnp.float32)
        self._y = jnp.array(y, dtype=jnp.float32)

    @property
    def input_shape(self):
        return (2,)

    @property
    def output_shape(self):
        return (1,)

    def setup(self, state=State()):
        return state

    def evaluate(self, state, randkey, act_func, params) -> float:
        eps = 1e-7

        def bce(x, label):
            logit = act_func(state, params, x)[0]
            p = jax.nn.sigmoid(logit)
            p = jnp.clip(p, eps, 1 - eps)
            return -(label[0] * jnp.log(p) + (1 - label[0]) * jnp.log(1 - p))

        losses = vmap(bce)(self._X, self._y)
        return -jnp.mean(losses)

    def show(self, state, randkey, act_func, params, *args, **kwargs):
        fit = float(self.evaluate(state, randkey, act_func, params))

        def predict(x):
            return (jax.nn.sigmoid(act_func(state, params, x)[0]) > 0.5).astype(jnp.float32)

        preds = vmap(predict)(self._X)
        acc = float(jnp.mean(preds == self._y[:, 0]))
        print(f"  SpiralProblem — neg-BCE: {fit:.4f}, accuracy: {acc:.4f}")


# ──────────────────────────────────────────────
# Problem B: VectorNeighbor (32次元)
# ──────────────────────────────────────────────

class VectorNeighborBenchProblem(BaseProblem):
    """
    32次元ベクトル近傍探索問題 (TRIDENT直結)。

    NEAT ゲノムが query → output_vec を学習し、
    output_vec とコーパスのコサイン類似度で近傍スコアを近似する。
    fitness = -MSE(predicted_scores, ground_truth_cosine_similarity)
    """

    jitable = True

    def __init__(self, queries: np.ndarray, corpus: np.ndarray):
        super().__init__()
        qn = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-8)
        cn = corpus  / (np.linalg.norm(corpus,  axis=1, keepdims=True) + 1e-8)
        targets = (qn @ cn.T).astype(np.float32)

        self._queries  = jnp.array(qn, dtype=jnp.float32)
        self._corpus   = jnp.array(cn, dtype=jnp.float32)
        self._targets  = jnp.array(targets, dtype=jnp.float32)
        self._dim = queries.shape[1]

    @property
    def input_shape(self):
        return (self._dim,)

    @property
    def output_shape(self):
        return (self._dim,)

    def setup(self, state=State()):
        return state

    def evaluate(self, state, randkey, act_func, params) -> float:
        def mse_query(q, t):
            out = act_func(state, params, q)
            out_n = out / (jnp.linalg.norm(out) + 1e-8)
            pred = self._corpus @ out_n
            return jnp.mean((pred - t) ** 2)

        losses = vmap(mse_query)(self._queries, self._targets)
        return -jnp.mean(losses)

    def show(self, state, randkey, act_func, params, *args, **kwargs):
        fit = float(self.evaluate(state, randkey, act_func, params))
        print(f"  VectorNeighborBench — neg-MSE: {fit:.6f}")


# ──────────────────────────────────────────────
# ロギング付きパイプライン実行
# ──────────────────────────────────────────────

def run_with_log(
    problem: BaseProblem,
    label: str,
    log_path: Path,
    pop_size: int = 100,
    species_size: int = 10,
    max_seconds: int = 1800,
    fitness_target: float = -0.01,
    seed: int = 42,
) -> dict:
    """
    NEAT を max_seconds 秒間実行し、世代ごとのログを JSONL に書き出す。

    Returns
    -------
    summary dict
    """
    inp_dim  = problem.input_shape[0]
    out_dim  = problem.output_shape[0]
    min_nodes = inp_dim + out_dim + 10
    min_conns = inp_dim * out_dim + 10

    genome = DefaultGenome(
        num_inputs=inp_dim,
        num_outputs=out_dim,
        max_nodes=max(80, min_nodes),
        max_conns=max(200, min_conns),
    )
    algorithm = NEAT(genome=genome, pop_size=pop_size, species_size=species_size)

    pipeline = Pipeline(
        algorithm=algorithm,
        problem=problem,
        seed=seed,
        fitness_target=fitness_target,
        generation_limit=999999,  # 時間制限で止める
    )

    print(f"\n{'='*60}")
    print(f"[{label}] 開始 (最大 {max_seconds}s / {max_seconds//60}分)")
    print(f"  入力={inp_dim}, 出力={out_dim}, pop={pop_size}, species={species_size}")
    print(f"  ログ → {log_path}")
    print(f"{'='*60}")

    state = pipeline.setup()

    # JIT コンパイル
    print("  コンパイル中...")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        compiled_step = jax.jit(pipeline.step).lower(state).compile()
    print("  コンパイル完了")

    generation = 0
    start_wall = time.time()
    best_fitness_ever = -float("inf")
    gen_times = []

    with open(log_path, "w") as f:
        while True:
            wall_elapsed = time.time() - start_wall
            if wall_elapsed >= max_seconds:
                print(f"\n  ⏱️  時間制限 {max_seconds}s に達しました (gen={generation})")
                break

            gen_start = time.time()
            state, prev_pop, fitnesses = compiled_step(state)
            fitnesses = jax.device_get(fitnesses)
            gen_elapsed = time.time() - gen_start
            gen_times.append(gen_elapsed)

            valid_mask = ~np.isinf(fitnesses) & ~np.isnan(fitnesses)
            valid = fitnesses[valid_mask]
            best_fit  = float(np.max(valid))   if len(valid) > 0 else float("nan")
            mean_fit  = float(np.mean(valid))  if len(valid) > 0 else float("nan")
            std_fit   = float(np.std(valid))   if len(valid) > 0 else float("nan")
            valid_cnt = int(len(valid))

            if np.isfinite(best_fit) and best_fit > best_fitness_ever:
                best_fitness_ever = best_fit

            # 種数の取得
            try:
                member_count = jax.device_get(state.species.member_count)
                species_cnt = int(np.sum(member_count > 0))
            except Exception:
                species_cnt = -1

            # ノード・コネクション数
            try:
                pop_nodes = jax.device_get(state.pop_nodes)
                pop_conns = jax.device_get(state.pop_conns)
                mean_nodes = float(np.mean((~np.isnan(pop_nodes[:, :, 0])).sum(axis=1)))
                mean_conns = float(np.mean((~np.isnan(pop_conns[:, :, 0])).sum(axis=1)))
            except Exception:
                mean_nodes = mean_conns = -1.0

            record = {
                "generation":    generation,
                "wall_time_s":   round(wall_elapsed, 3),
                "gen_time_ms":   round(gen_elapsed * 1000, 2),
                "best_fitness":  round(best_fit, 6) if np.isfinite(best_fit) else None,
                "mean_fitness":  round(mean_fit, 6) if np.isfinite(mean_fit) else None,
                "std_fitness":   round(std_fit, 6)  if np.isfinite(std_fit)  else None,
                "valid_count":   valid_cnt,
                "best_ever":     round(best_fitness_ever, 6),
                "species_count": species_cnt,
                "mean_nodes":    round(mean_nodes, 1),
                "mean_conns":    round(mean_conns, 1),
            }
            f.write(json.dumps(record) + "\n")
            f.flush()

            # コンソール出力 (10世代ごと)
            if generation % 10 == 0:
                print(
                    f"  Gen {generation:4d} | "
                    f"wall={wall_elapsed:6.1f}s | "
                    f"best={best_fit:8.5f} | "
                    f"mean={mean_fit:8.5f} | "
                    f"gen={gen_elapsed*1000:.1f}ms | "
                    f"species={species_cnt} | "
                    f"nodes={mean_nodes:.1f}"
                )

            generation += 1

            # fitness target 到達チェック
            if np.isfinite(best_fit) and best_fit >= fitness_target:
                print(f"\n  ✅ fitness_target={fitness_target} 到達 (gen={generation})")
                break

    total_time = time.time() - start_wall
    avg_gen_ms = float(np.mean(gen_times)) * 1000 if gen_times else 0.0

    summary = {
        "label":            label,
        "total_generations": generation,
        "total_time_s":     round(total_time, 2),
        "avg_gen_time_ms":  round(avg_gen_ms, 2),
        "best_fitness":     round(best_fitness_ever, 6),
        "fitness_target":   fitness_target,
        "converged":        best_fitness_ever >= fitness_target,
        "pop_size":         pop_size,
        "species_size":     species_size,
    }
    print(f"\n  完了: {generation} 世代, {total_time:.1f}s, best={best_fitness_ever:.6f}")
    return summary


# ──────────────────────────────────────────────
# メイン
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-seconds", type=int, default=1800,
                        help="各問題の最大実行秒数 (デフォルト: 1800 = 30分)")
    parser.add_argument("--pop-size", type=int, default=100)
    parser.add_argument("--species-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None,
                        help="ログファイル名プレフィックス (例: gpu, cpu). 未指定時は自動検出")
    args = parser.parse_args()

    print("=" * 60)
    print("TRIDENT NEAT ベンチマーク")
    print("=" * 60)

    # デバイス検出
    dev_info = detect_devices()
    print(f"\nデバイス情報:")
    print(f"  backend  : {dev_info['backend']}")
    print(f"  devices  : {dev_info['devices']}")
    print(f"  GPU利用  : {'✅ 利用可能' if dev_info['has_gpu'] else '❌ CPU のみ (CUDA未設定)'}")

    device_prefix = args.device if args.device else ("gpu" if dev_info["has_gpu"] else "cpu")

    summaries = []

    # ──────────────────────────────────────────────
    # Problem A: Double Spiral Classification
    # ──────────────────────────────────────────────
    print("\n[Problem A] Double Spiral Classification")
    X, y = make_spiral_data(n_per_class=100, noise=0.15, seed=args.seed)
    print(f"  データ: X={X.shape}, y={y.shape}, class balance={int(y.sum())}/{len(y)}")

    prob_a = SpiralProblem(X, y)
    summary_a = run_with_log(
        problem=prob_a,
        label=f"{device_prefix.upper()}_Spiral",
        log_path=LOG_DIR / f"benchmark_{device_prefix}_spiral.jsonl",
        pop_size=args.pop_size,
        species_size=args.species_size,
        max_seconds=args.max_seconds,
        fitness_target=-0.3,   # BCE 0.3 以下 ≈ accuracy ~80%
        seed=args.seed,
    )
    summaries.append(summary_a)

    # ──────────────────────────────────────────────
    # Problem B: VectorNeighbor (32次元)
    # ──────────────────────────────────────────────
    print("\n[Problem B] VectorNeighbor 32次元")
    DIM = 32
    rng = np.random.default_rng(args.seed)
    corpus  = rng.standard_normal((100, DIM)).astype(np.float32)
    queries = rng.standard_normal((20, DIM)).astype(np.float32)
    print(f"  corpus={corpus.shape}, queries={queries.shape}")

    prob_b = VectorNeighborBenchProblem(queries, corpus)
    summary_b = run_with_log(
        problem=prob_b,
        label=f"{device_prefix.upper()}_VectorNeighbor",
        log_path=LOG_DIR / f"benchmark_{device_prefix}_vector.jsonl",
        pop_size=args.pop_size,
        species_size=args.species_size,
        max_seconds=args.max_seconds,
        fitness_target=-0.01,
        seed=args.seed,
    )
    summaries.append(summary_b)

    # ──────────────────────────────────────────────
    # GPU セクション
    # ──────────────────────────────────────────────
    gpu_note = ""
    if not dev_info["has_gpu"]:
        gpu_note = (
            "GPU 実行: ❌ CUDA 非対応 jaxlib がインストールされていないため実行不可。\n"
            "  → WSL2 + CUDA 設定後に jaxlib[cuda] をインストールすれば\n"
            "    同一スクリプトで GPU ベンチマークが実行可能。\n"
            "  → GPU 時の期待スループット: TensorNEAT 公式ベンチでは\n"
            "    CPU比 最大 500x 高速化 (GPU並列 vmap による集団評価)。"
        )
    else:
        gpu_note = "GPU 実行: ✅ (上記 CPU と同一パラメータで実行済み — GPU ログは benchmark_gpu_*.jsonl)"

    # ──────────────────────────────────────────────
    # サマリーレポート生成
    # ──────────────────────────────────────────────
    report_path = LOG_DIR / "benchmark_summary.txt"
    with open(report_path, "w") as f:
        def w(s=""):
            print(s)
            f.write(s + "\n")

        w("=" * 60)
        w("TRIDENT NEAT ベンチマーク サマリーレポート")
        w("=" * 60)
        w()
        w(f"実行日時 : {time.strftime('%Y-%m-%d %H:%M:%S')}")
        w(f"backend  : {dev_info['backend']}")
        w(f"devices  : {dev_info['devices']}")
        w()

        w(f"■ {device_prefix.upper()} 実行結果")
        w("-" * 40)
        for s in summaries:
            w(f"  [{s['label']}]")
            w(f"    総世代数          : {s['total_generations']}")
            w(f"    総実行時間        : {s['total_time_s']}s ({s['total_time_s']/60:.1f}分)")
            w(f"    平均世代時間      : {s['avg_gen_time_ms']:.2f} ms/gen")
            w(f"    最良 fitness      : {s['best_fitness']}")
            w(f"    fitness target    : {s['fitness_target']}")
            w(f"    収束              : {'✅' if s['converged'] else '未達'}")
            w()

        w("■ GPU 実行結果")
        w("-" * 40)
        for line in gpu_note.split("\n"):
            w(f"  {line}")
        w()

        w("■ 比較メモ (CPU vs GPU 理論値)")
        w("-" * 40)
        for s in summaries:
            gen_ms = s['avg_gen_time_ms']
            expected_gpu_ms = gen_ms / 100   # TensorNEAT 公式: 最大100x (保守的見積)
            expected_gpu_ms_max = gen_ms / 500  # 公式最大値
            w(f"  [{s['label']}]")
            w(f"    CPU 平均世代時間       : {gen_ms:.2f} ms/gen")
            w(f"    GPU 期待値 (100x)     : {expected_gpu_ms:.2f} ms/gen")
            w(f"    GPU 期待値 (最大500x) : {expected_gpu_ms_max:.2f} ms/gen")
            w(f"    30分で期待できる世代数:")
            w(f"      CPU 実績            : {s['total_generations']} 世代")
            w(f"      GPU 期待 (100x)     : {int(s['total_generations'] * 100)} 世代")
            w(f"      GPU 期待 (最大500x) : {int(s['total_generations'] * 500)} 世代")
            w()

        w("■ ログファイル")
        w("-" * 40)
        for s in summaries:
            fname = f"benchmark_{s['label'].lower()}.jsonl"
            w(f"  {fname}")
        w()
        w(f"  サマリー: {report_path}")

    print(f"\nレポート出力: {report_path}")

    # JSON サマリーも保存
    with open(LOG_DIR / "benchmark_summary.json", "w") as f:
        json.dump({
            "device_info": dev_info,
            "summaries": summaries,
            "gpu_note": gpu_note,
        }, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
