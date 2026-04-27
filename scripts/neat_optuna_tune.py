"""
Optuna による NeatIndexer ハイパーパラメーター最適化。

探索対象:
  pop_size          int  log [50, 1000]
  species_size      int  [5, 100]
  generation_limit  int  [10, 80]
  max_nodes         categorical  [64, 128, 256]  (JIT再コンパイル回避)
  max_conns         categorical  [128, 512, 1024]

目的: best_fitness (最大化) = -MSE(predicted_scores, true_cosine_sim)

実行例:
  poetry run python scripts/neat_optuna_tune.py
  poetry run python scripts/neat_optuna_tune.py --n-trials 50 --dim 32 --corpus-size 200
  poetry run python scripts/neat_optuna_tune.py --study-name neat_v2 --n-trials 100
"""

from __future__ import annotations

import argparse
import time
import os
import sys

import numpy as np
import optuna
from optuna.samplers import TPESampler

sys.path.insert(0, ".")

from tensorneat.common import State
from tensorneat.genome import DefaultGenome
from tensorneat.algorithm.neat import NEAT
from tensorneat import Pipeline
from src.interfaces.neat_indexer import VectorNeighborProblem


# ──────────────────────────────────────────────
# 1 試行の実行
# ──────────────────────────────────────────────

def run_trial(
    pop_size: int,
    species_size: int,
    generation_limit: int,
    max_nodes: int,
    max_conns: int,
    corpus: np.ndarray,
    queries: np.ndarray,
    seed: int = 0,
) -> float:
    """NeatIndexer 相当の NEAT を実行し best_fitness を返す。"""

    dim = corpus.shape[1]
    min_nodes = dim * 2 + 10
    min_conns = dim * dim + 10
    eff_nodes = max(max_nodes, min_nodes)
    eff_conns = max(max_conns, min_conns)

    problem = VectorNeighborProblem(
        queries=queries.astype(np.float32),
        corpus=corpus.astype(np.float32),
    )

    genome = DefaultGenome(
        num_inputs=dim,
        num_outputs=dim,
        max_nodes=eff_nodes,
        max_conns=eff_conns,
    )

    algorithm = NEAT(
        genome=genome,
        pop_size=pop_size,
        species_size=species_size,
    )

    pipeline = Pipeline(
        algorithm=algorithm,
        problem=problem,
        seed=seed,
        fitness_target=-0.001,   # 実質到達しない; generation_limit で制御
        generation_limit=generation_limit,
    )

    init_state = pipeline.setup()
    pipeline.auto_run(init_state)   # state は不要; best_fitness は pipeline 属性に保持
    return float(pipeline.best_fitness)


# ──────────────────────────────────────────────
# Optuna objective
# ──────────────────────────────────────────────

def make_objective(corpus: np.ndarray, queries: np.ndarray, seed_base: int = 0):
    def objective(trial: optuna.Trial) -> float:
        pop_size         = trial.suggest_int("pop_size", 50, 1000, log=True)
        species_size     = trial.suggest_int("species_size", 5, 100)
        generation_limit = trial.suggest_int("generation_limit", 10, 80)
        max_nodes        = trial.suggest_categorical("max_nodes", [64, 128, 256])
        max_conns        = trial.suggest_categorical("max_conns", [128, 512, 1024])

        t0 = time.perf_counter()
        fitness = run_trial(
            pop_size=pop_size,
            species_size=species_size,
            generation_limit=generation_limit,
            max_nodes=max_nodes,
            max_conns=max_conns,
            corpus=corpus,
            queries=queries,
            seed=seed_base + trial.number,
        )
        elapsed = time.perf_counter() - t0

        trial.set_user_attr("elapsed_s", round(elapsed, 2))
        print(
            f"  Trial {trial.number:3d} | "
            f"pop={pop_size:4d} sp={species_size:3d} gen={generation_limit:3d} "
            f"nodes={max_nodes} conns={max_conns} | "
            f"fitness={fitness:.6f} ({elapsed:.1f}s)"
        )
        return fitness

    return objective


# ──────────────────────────────────────────────
# エントリーポイント
# ──────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna NEAT hyperparameter tuning")
    parser.add_argument("--n-trials",    type=int,   default=30,          help="試行数 (default: 30)")
    parser.add_argument("--dim",         type=int,   default=16,          help="埋め込み次元 (default: 16)")
    parser.add_argument("--corpus-size", type=int,   default=100,         help="コーパスサイズ (default: 100)")
    parser.add_argument("--study-name",  type=str,   default="neat_tune", help="Optuna study 名")
    parser.add_argument("--seed",        type=int,   default=0,           help="乱数シード")
    parser.add_argument("--n-jobs",      type=int,   default=1,           help="並列試行数 (default: 1 = 直列)")
    args = parser.parse_args()

    os.makedirs("logs", exist_ok=True)
    storage = f"sqlite:///logs/optuna_neat.db"

    # コーパス・クエリ生成
    rng = np.random.default_rng(args.seed)
    corpus = rng.standard_normal((args.corpus_size, args.dim)).astype(np.float32)
    n_queries = max(10, args.corpus_size // 5)
    queries = corpus[rng.permutation(args.corpus_size)[:n_queries]]

    print(f"=== Optuna NEAT Tuning ===")
    print(f"study       : {args.study_name}")
    print(f"storage     : {storage}")
    print(f"trials      : {args.n_trials}")
    print(f"dim         : {args.dim}  corpus_size={args.corpus_size}  n_queries={n_queries}")
    print()

    # optuna ログを WARNING 以上に絞る
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=TPESampler(seed=args.seed),
    )

    study.optimize(
        make_objective(corpus, queries, seed_base=args.seed),
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        show_progress_bar=False,
    )

    best = study.best_trial
    print()
    print("=== 最適パラメーター ===")
    print(f"  best_fitness : {best.value:.6f}")
    print(f"  trial#       : {best.number}")
    for k, v in best.params.items():
        print(f"  {k:<20}: {v}")
    print(f"\n結果は {storage} に保存されました。")
    print("可視化: optuna-dashboard logs/optuna_neat.db")


if __name__ == "__main__":
    main()
