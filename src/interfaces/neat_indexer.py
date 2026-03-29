"""
TRIDENT — A型 I/F: NeatIndexer
NEATトポロジによるクエリ→近傍スコアベクトル変換。FAISSの進化的代替。

入力 : クエリ埋め込み f32[input_dim]
出力 : 近傍候補スコア f32[input_dim]  (コーパスとのコサイン類似度近似)
BCS  : (平均活性化強度, 検索再現率Recall@k)  格子 64×64
"""

from __future__ import annotations

from typing import Literal, Optional
import numpy as np
import jax
import jax.numpy as jnp
from jax import vmap

from tensorneat.common import State
from tensorneat.problem.base import BaseProblem
from tensorneat.genome import DefaultGenome
from tensorneat.algorithm.neat import NEAT
from tensorneat import Pipeline


# ──────────────────────────────────────────────
# カスタム適合度問題: ベクトル近傍探索フィッティング
# ──────────────────────────────────────────────

class VectorNeighborProblem(BaseProblem):
    """
    コーパスベクトル群に対して、クエリ→近傍スコアを学習する適合度問題。

    fitness = -MSE(predicted_scores, target_scores)

    target_scores は各コーパスベクトルとのコサイン類似度 (ground truth)。
    predicted_scores は NEAT ゲノムの出力をコサイン類似度空間への写像とみなす。
    """

    jitable = True

    def __init__(self, queries: np.ndarray, corpus: np.ndarray, k: int = 10):
        """
        Parameters
        ----------
        queries : (Q, D) クエリベクトル群
        corpus  : (C, D) コーパスベクトル群
        k       : Recall@k の k
        """
        super().__init__()
        assert queries.ndim == 2 and corpus.ndim == 2
        assert queries.shape[1] == corpus.shape[1]

        # 正規化してコサイン類似度をドット積で計算
        queries_n = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-8)
        corpus_n  = corpus  / (np.linalg.norm(corpus,  axis=1, keepdims=True) + 1e-8)

        # target: (Q, C) コサイン類似度行列
        target_scores = (queries_n @ corpus_n.T).astype(np.float32)

        self._queries = jnp.array(queries_n, dtype=jnp.float32)  # (Q, D)
        self._targets = jnp.array(target_scores, dtype=jnp.float32)  # (Q, C)
        self._corpus  = jnp.array(corpus_n, dtype=jnp.float32)  # (C, D)
        self._k = k
        self._input_dim  = queries.shape[1]
        self._corpus_size = corpus.shape[0]

    # BaseProblem required properties
    @property
    def input_shape(self):
        return (self._input_dim,)

    @property
    def output_shape(self):
        return (self._input_dim,)

    def setup(self, state: State = State()):
        return state

    def evaluate(self, state: State, randkey, act_func, params) -> float:
        """
        適合度: クエリ→出力ベクトルとコーパスとのコサイン類似度ランキングの精度。

        NEAT ゲノムは query f32[D] → output f32[D] を出力する。
        output をコーパス C (C, D) に対してドット積し、スコアを計算。
        target スコアとの MSE を最小化する。
        """
        def score_query(query, target_row):
            out = act_func(state, params, query)          # f32[D]
            out_n = out / (jnp.linalg.norm(out) + 1e-8)   # 正規化
            pred_scores = self._corpus @ out_n              # (C,)
            loss = jnp.mean((pred_scores - target_row) ** 2)
            return loss

        losses = vmap(score_query)(self._queries, self._targets)  # (Q,)
        return -jnp.mean(losses)

    def show(self, state, randkey, act_func, params, *args, **kwargs):
        fitness = float(self.evaluate(state, randkey, act_func, params))
        print(f"VectorNeighborProblem fitness (neg-MSE): {fitness:.6f}")


# ──────────────────────────────────────────────
# BCS 記述子計算
# ──────────────────────────────────────────────

def compute_bcs_descriptor(
    output_vectors: np.ndarray, recall_at_k: float
) -> np.ndarray:
    """
    MAP-Elites 行動特性空間 (BCS) の記述子を計算する。

    Parameters
    ----------
    output_vectors : (Q, D) 全クエリに対するゲノム出力
    recall_at_k    : Recall@k スコア [0, 1]

    Returns
    -------
    descriptor : f32[2]
        [mean_activation_magnitude, recall_at_k]
        どちらも [0, 1] に正規化済み
    """
    # 軸1: 出力の平均活性化強度 (絶対値の平均を sigmoid で [0,1] に圧縮)
    mean_mag = float(np.mean(np.abs(output_vectors)))
    act_intensity = float(1.0 / (1.0 + np.exp(-mean_mag)))  # sigmoid

    # 軸2: recall_at_k はそのまま [0, 1]
    return np.array([act_intensity, float(recall_at_k)], dtype=np.float32)


def recall_at_k(
    act_func,
    state,
    params,
    queries: np.ndarray,
    corpus: np.ndarray,
    true_neighbors: np.ndarray,
    k: int = 10,
) -> float:
    """
    NEAT ゲノムの Recall@k を計算する。

    Parameters
    ----------
    true_neighbors : (Q, k) 各クエリの正解近傍インデックス
    """
    outputs = np.array(
        vmap(lambda q: act_func(state, params, jnp.array(q, dtype=jnp.float32)))(
            jnp.array(queries, dtype=jnp.float32)
        )
    )  # (Q, D)
    corpus_n = corpus / (np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-8)
    outputs_n = outputs / (np.linalg.norm(outputs, axis=1, keepdims=True) + 1e-8)

    scores = outputs_n @ corpus_n.T  # (Q, C)
    pred_top_k = np.argsort(-scores, axis=1)[:, :k]  # (Q, k)

    hits = 0
    total = 0
    for q_idx in range(queries.shape[0]):
        true = set(true_neighbors[q_idx].tolist())
        pred = set(pred_top_k[q_idx].tolist())
        hits += len(true & pred)
        total += len(true)

    return hits / total if total > 0 else 0.0


# ──────────────────────────────────────────────
# NeatIndexer (A型 I/F メインクラス)
# ──────────────────────────────────────────────

class NeatIndexer:
    """
    A型 I/F: float vector ベースの NEAT 近傍探索器。

    NEAT でクエリ埋め込み → 近傍スコアベクトルを学習し、
    MAP-Elites（QDax）でスキルアーカイブを管理する。

    Parameters
    ----------
    input_dim         : 入力・出力次元 (FAISS と合わせる場合は 384)
    pop_size          : NEAT 集団サイズ
    species_size      : 種数
    max_nodes         : ゲノムの最大ノード数
    max_conns         : ゲノムの最大コネクション数
    generation_limit  : 進化の最大世代数
    k                 : Recall@k の k
    seed              : 乱数シード
    """

    def __init__(
        self,
        input_dim: int = 16,  # プロトタイプ用デフォルト (本番: 384)
        pop_size: int = 50,
        species_size: int = 5,
        max_nodes: int = 50,
        max_conns: int = 100,
        generation_limit: int = 100,
        k: int = 5,
        seed: int = 42,
    ):
        self.input_dim = input_dim
        self.pop_size = pop_size
        self.species_size = species_size
        self.max_nodes = max_nodes
        self.max_conns = max_conns
        self.generation_limit = generation_limit
        self.k = k
        self.seed = seed

        # 学習済み状態
        self._state: Optional[State] = None
        self._best_params = None
        self._pipeline: Optional[Pipeline] = None
        self._problem: Optional[VectorNeighborProblem] = None
        self._corpus: Optional[np.ndarray] = None

    def fit(
        self,
        corpus: np.ndarray,
        queries: Optional[np.ndarray] = None,
        fitness_target: float = -0.01,
    ) -> "NeatIndexer":
        """
        コーパスに対して NEAT を進化させる。

        Parameters
        ----------
        corpus          : (C, D) コーパスベクトル群
        queries         : (Q, D) 学習用クエリ (None の場合はコーパスをシャッフルして使用)
        fitness_target  : 到達目標 fitness (-MSE)

        Returns
        -------
        self
        """
        assert corpus.ndim == 2 and corpus.shape[1] == self.input_dim, (
            f"corpus は (C, {self.input_dim}) である必要があります。"
            f"受け取った shape: {corpus.shape}"
        )

        if queries is None:
            rng = np.random.default_rng(self.seed)
            idx = rng.permutation(len(corpus))[:max(10, len(corpus) // 5)]
            queries = corpus[idx]

        self._corpus = corpus.astype(np.float32)

        self._problem = VectorNeighborProblem(
            queries=queries.astype(np.float32),
            corpus=corpus.astype(np.float32),
            k=self.k,
        )

        # 初期ノード数 = input + output。max_nodes / max_conns はそれ以上が必須
        min_nodes = self.input_dim * 2 + 10         # 入力+出力+マージン
        min_conns = self.input_dim * self.input_dim + 10  # 初期全結合+マージン
        effective_max_nodes = max(self.max_nodes, min_nodes)
        effective_max_conns = max(self.max_conns, min_conns)

        genome = DefaultGenome(
            num_inputs=self.input_dim,
            num_outputs=self.input_dim,
            max_nodes=effective_max_nodes,
            max_conns=effective_max_conns,
        )

        algorithm = NEAT(
            genome=genome,
            pop_size=self.pop_size,
            species_size=self.species_size,
        )

        self._pipeline = Pipeline(
            algorithm=algorithm,
            problem=self._problem,
            seed=self.seed,
            fitness_target=fitness_target,
            generation_limit=self.generation_limit,
        )

        # 進化実行
        print(f"[NeatIndexer] 進化開始: pop={self.pop_size}, dim={self.input_dim}, "
              f"max_gen={self.generation_limit}")
        init_state = self._pipeline.setup()
        self._state, self._best_params = self._pipeline.auto_run(init_state)
        print("[NeatIndexer] 進化完了")

        return self

    def transform(self, query: np.ndarray) -> np.ndarray:
        """
        クエリベクトルを近傍スコアベクトルに変換する。

        Parameters
        ----------
        query : (D,) または (Q, D) クエリベクトル

        Returns
        -------
        output : (D,) または (Q, D)
        """
        assert self._state is not None, "fit() を先に実行してください"

        forward = self._pipeline.algorithm.forward  # (state, params, x) -> output

        single = query.ndim == 1
        q = jnp.array(query[None] if single else query, dtype=jnp.float32)

        # best_params を transform して推論可能な形式に変換
        best_transformed = self._pipeline.algorithm.transform(
            self._state, self._best_params
        )
        outputs = vmap(lambda x: forward(self._state, best_transformed, x))(q)

        result = np.array(outputs)
        return result[0] if single else result

    def search(self, query: np.ndarray, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
        """
        NEAT で変換したスコアを使ってコーパスから k 近傍を返す。

        Parameters
        ----------
        query : (D,) クエリベクトル

        Returns
        -------
        indices : (k,) 近傍インデックス
        scores  : (k,) コサイン類似度スコア
        """
        assert self._corpus is not None, "fit() を先に実行してください"
        assert query.ndim == 1

        out = self.transform(query)  # (D,)
        out_n = out / (np.linalg.norm(out) + 1e-8)
        corpus_n = self._corpus / (np.linalg.norm(self._corpus, axis=1, keepdims=True) + 1e-8)

        scores = corpus_n @ out_n  # (C,)
        top_k_idx = np.argsort(-scores)[:k]
        return top_k_idx, scores[top_k_idx]

    def bcs_descriptor(self, queries: np.ndarray, true_neighbors: Optional[np.ndarray] = None) -> np.ndarray:
        """
        この個体の MAP-Elites BCS 記述子を返す。

        Returns
        -------
        descriptor : f32[2] — [活性化強度, Recall@k]
        """
        assert self._state is not None, "fit() を先に実行してください"

        outputs = []
        for q in queries:
            outputs.append(self.transform(q))
        outputs = np.stack(outputs)  # (Q, D)

        if true_neighbors is not None:
            best_transformed = self._pipeline.algorithm.transform(
                self._state, self._best_params
            )
            rak = recall_at_k(
                act_func=self._pipeline.algorithm.forward,
                state=self._state,
                params=best_transformed,
                queries=queries,
                corpus=self._corpus,
                true_neighbors=true_neighbors,
                k=self.k,
            )
        else:
            rak = 0.0

        return compute_bcs_descriptor(outputs, rak)


# ──────────────────────────────────────────────
# HybridIndexer: FAISS と NeatIndexer の並列稼働
# ──────────────────────────────────────────────

class HybridIndexer:
    """
    FAISS (既存) と NeatIndexer (TRIDENT) を並列稼働させる橋渡し層。

    mode:
        "faiss" — FAISS のみ使用 (既存動作維持)
        "neat"  — NeatIndexer のみ使用
        "hybrid"— 両方実行してスコアを平均
    """

    def __init__(
        self,
        neat_indexer: NeatIndexer,
        faiss_index=None,
        mode: Literal["faiss", "neat", "hybrid"] = "neat",
    ):
        self.neat_indexer = neat_indexer
        self.faiss_index = faiss_index
        self.mode = mode

        if mode in ("faiss", "hybrid") and faiss_index is None:
            raise ValueError("mode='faiss'/'hybrid' には faiss_index が必要です")

    def search(self, query: np.ndarray, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
        """k 近傍検索 (mode に応じて使い分け)。"""
        q = query.astype(np.float32)

        if self.mode == "faiss":
            scores, indices = self.faiss_index.search(q[None], k)
            return indices[0], scores[0]

        if self.mode == "neat":
            return self.neat_indexer.search(q, k)

        # hybrid: 両スコアを正規化して平均
        neat_idx, neat_scores = self.neat_indexer.search(q, k)

        faiss_scores, faiss_idx = self.faiss_index.search(q[None], k)
        faiss_idx, faiss_scores = faiss_idx[0], faiss_scores[0]

        # 全候補をまとめてリランク
        all_idx = np.unique(np.concatenate([neat_idx, faiss_idx]))
        neat_map  = dict(zip(neat_idx.tolist(),  neat_scores.tolist()))
        faiss_map = dict(zip(faiss_idx.tolist(), faiss_scores.tolist()))

        combined = {}
        for idx in all_idx:
            ns = neat_map.get(int(idx), 0.0)
            fs = faiss_map.get(int(idx), 0.0)
            combined[int(idx)] = (ns + fs) / 2.0

        sorted_idx = sorted(combined, key=combined.get, reverse=True)[:k]
        return np.array(sorted_idx), np.array([combined[i] for i in sorted_idx])
