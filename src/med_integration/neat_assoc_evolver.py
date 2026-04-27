"""
TRIDENT — NEAT → AssociationFn 進化ループ
フィードバックペアから NEAT で association_fn アーキテクチャを進化させる。

入力  : 4特徴量 [cos(q,c), cos(q,ctx), cos(c,ctx), cos(q-ctx,c)]
出力  : 関連度スコア (スカラー)
適合度: -MSE(predicted_score, label)

ファイル: src/med_integration/neat_assoc_evolver.py
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap

from tensorneat.common import State
from tensorneat.problem.base import BaseProblem
from tensorneat.genome import DefaultGenome
from tensorneat.algorithm.neat import NEAT
from tensorneat import Pipeline


# ──────────────────────────────────────────────
# 特徴量計算ユーティリティ
# ──────────────────────────────────────────────

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def compute_features(
    query:     np.ndarray,
    candidate: np.ndarray,
    context:   np.ndarray | None,
) -> np.ndarray:
    """
    (query, candidate, context) → f32[4] 特徴量ベクトル。
    context=None のとき ctx 関連特徴は 0 になる。

    features:
      [0] cos(query, candidate)
      [1] cos(query, context)      / 0 if no context
      [2] cos(candidate, context)  / 0 if no context
      [3] cos(query - context, candidate) / 0 if no context
    """
    f0 = _cosine(query, candidate)
    if context is None:
        return np.array([f0, 0.0, 0.0, 0.0], dtype=np.float32)
    f1 = _cosine(query, context)
    f2 = _cosine(candidate, context)
    f3 = _cosine(query - context, candidate)
    return np.array([f0, f1, f2, f3], dtype=np.float32)


# ──────────────────────────────────────────────
# BaseProblem: 適合度問題
# ──────────────────────────────────────────────

class AssociationFnProblem(BaseProblem):
    """
    NEAT適合度問題: フィードバックペアから association_fn を進化させる。

    入力  : f32[4] 事前計算済み類似度特徴量
    出力  : f32[1] 関連度スコア
    適合度: -MSE(score, label)  ∈ (-∞, 0]  高いほど良い
    """

    jitable = True

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        """
        Parameters
        ----------
        features : (N, 4) 各ペアの類似度特徴量
        labels   : (N,)   正例=1.0, 負例=0.0
        """
        super().__init__()
        assert features.ndim == 2 and features.shape[1] == 4
        assert labels.ndim == 1 and len(labels) == len(features)

        self._features = jnp.array(features, dtype=jnp.float32)   # (N, 4)
        self._labels   = jnp.array(labels,   dtype=jnp.float32)   # (N,)

    @property
    def input_shape(self):
        return (4,)

    @property
    def output_shape(self):
        return (1,)

    def setup(self, state: State = State()):
        return state

    def evaluate(self, state: State, randkey, act_func, params) -> float:
        """fitness = -mean_MSE(network(features), labels)"""
        def mse_sample(feat, label):
            out = act_func(state, params, feat)   # f32[1]
            return (out[0] - label) ** 2

        losses = vmap(mse_sample)(self._features, self._labels)   # (N,)
        return -jnp.mean(losses)

    def show(self, state, randkey, act_func, params, *args, **kwargs):
        fitness = float(self.evaluate(state, randkey, act_func, params))

        def pred(feat):
            return act_func(state, params, feat)[0]

        preds   = np.array(vmap(pred)(self._features))
        labels  = np.array(self._labels)
        binary  = (preds >= 0.5).astype(float)
        acc     = float(np.mean(binary == labels))
        print(f"AssociationFnProblem — fitness: {fitness:.4f}, "
              f"binary_acc: {acc:.4f}")


# ──────────────────────────────────────────────
# NEATAssociationFn: 進化済みスコア関数
# ──────────────────────────────────────────────

class NEATAssociationFn:
    """
    NEAT 進化で得た association_fn。AssociationFnProtocol 準拠。

    score(q, c, ctx) = NEAT_network([cos(q,c), cos(q,ctx), cos(c,ctx), cos(q-ctx,c)])
    """

    def __init__(self, pipeline: Pipeline, state: Any, best_params: Any, generation: int = 0):
        self._pipeline     = pipeline
        self._state        = state
        self._best_params  = best_params
        self.arch_meta: dict = {
            "arch_type": "neat_cppn",
            "generation": generation,
            "fitness_history": [],
        }

    # ─── スコア計算 ────────────────────────────

    def _transform(self):
        return self._pipeline.algorithm.transform(self._state, self._best_params)

    def score(
        self,
        query:     np.ndarray,
        candidate: np.ndarray,
        context:   np.ndarray | None = None,
    ) -> float:
        feat = compute_features(query, candidate, context)
        feat_j = jnp.array(feat, dtype=jnp.float32)
        transformed = self._transform()
        out = self._pipeline.algorithm.forward(self._state, transformed, feat_j)
        return float(out[0])

    def score_batch(
        self,
        query:     np.ndarray,
        candidates: np.ndarray,
        context:   np.ndarray | None = None,
    ) -> np.ndarray:
        """(n, dim) → (n,) scores"""
        transformed = self._transform()

        def single(c):
            feat = compute_features(query, c, context)
            feat_j = jnp.array(feat, dtype=jnp.float32)
            return self._pipeline.algorithm.forward(self._state, transformed, feat_j)[0]

        return np.array([single(c) for c in candidates], dtype=np.float64)

    # ─── 永続化 ───────────────────────────────

    def to_dict(self) -> dict:
        return {
            "arch_type":      "neat_cppn",
            "generation":     self.arch_meta["generation"],
            "fitness_history": self.arch_meta["fitness_history"],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "NEATAssociationFn":
        raise NotImplementedError(
            "NEATAssociationFn のロードは load() (pickle) を使ってください"
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_bytes(pickle.dumps(self))

    @classmethod
    def load(cls, path: str | Path) -> "NEATAssociationFn":
        return pickle.loads(Path(path).read_bytes())


# ──────────────────────────────────────────────
# AssociationFnEvolver: 進化ループ管理
# ──────────────────────────────────────────────

class AssociationFnEvolver:
    """
    NEAT で AssociationFn アーキテクチャを進化させるメインクラス。

    feedback_pairs から特徴量を計算し、NEAT 進化を実行する。
    結果を NEATAssociationFn として返す。
    """

    def __init__(
        self,
        pop_size:   int = 50,
        species_size: int = 5,
        max_nodes:  int = 20,
        max_conns:  int = 30,
        seed:       int = 42,
    ):
        self.pop_size     = pop_size
        self.species_size = species_size
        self.max_nodes    = max_nodes
        self.max_conns    = max_conns
        self.seed         = seed

    def evolve(
        self,
        feedback_pairs: list[dict],
        generation_limit: int = 100,
        fitness_target: float = -0.1,
        verbose: bool = True,
    ) -> NEATAssociationFn:
        """
        フィードバックペアから NEATAssociationFn を進化させる。

        Parameters
        ----------
        feedback_pairs : list of dicts with keys:
            query, candidate, context (or None), label (1.0/0.0)
        generation_limit : 最大世代数
        fitness_target   : 到達目標 fitness (-MSE)
        verbose          : 進捗表示

        Returns
        -------
        NEATAssociationFn
        """
        if not feedback_pairs:
            raise ValueError("feedback_pairs が空です")

        features, labels = self._build_dataset(feedback_pairs)
        problem  = AssociationFnProblem(features, labels)
        genome   = DefaultGenome(
            num_inputs=4,
            num_outputs=1,
            max_nodes=self.max_nodes,
            max_conns=self.max_conns,
        )
        algorithm = NEAT(
            genome=genome,
            pop_size=self.pop_size,
            species_size=self.species_size,
        )
        pipeline = Pipeline(
            algorithm=algorithm,
            problem=problem,
            seed=self.seed,
            fitness_target=fitness_target,
            generation_limit=generation_limit,
        )

        if verbose:
            print(f"[AssociationFnEvolver] 進化開始: "
                  f"pop={self.pop_size}, gen_limit={generation_limit}, "
                  f"n_pairs={len(feedback_pairs)}")

        init_state = pipeline.setup()
        state, best_params = pipeline.auto_run(init_state)

        if verbose:
            print("[AssociationFnEvolver] 進化完了")

        return NEATAssociationFn(
            pipeline=pipeline,
            state=state,
            best_params=best_params,
            generation=generation_limit,
        )

    @staticmethod
    def _build_dataset(
        feedback_pairs: list[dict],
    ) -> tuple[np.ndarray, np.ndarray]:
        """feedback_pairs → (features (N,4), labels (N,))"""
        feats  = []
        labels = []
        for pair in feedback_pairs:
            q   = np.asarray(pair["query"],     dtype=np.float64)
            c   = np.asarray(pair["candidate"], dtype=np.float64)
            ctx = pair.get("context")
            ctx = np.asarray(ctx, dtype=np.float64) if ctx is not None else None
            feats.append(compute_features(q, c, ctx))
            labels.append(float(pair["label"]))

        return np.array(feats, dtype=np.float32), np.array(labels, dtype=np.float32)
