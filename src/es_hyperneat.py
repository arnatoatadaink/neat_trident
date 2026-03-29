"""
TRIDENT — ES-HyperNEAT カスタム拡張
TensorNEAT の HyperNEAT をベースに Evolvable Substrate (ES) を実装する。

ES-HyperNEAT の核心:
  通常 HyperNEAT: 固定基板 (Substrate) で CPPN が接続重みを決定
  ES-HyperNEAT:  CPPN の出力パターンから隠れノードの配置を動的に決定
                  → 基板のトポロジ自体が進化する

TRIDENT での活用:
  固定ランダム射影行列 R (384×32) の代わりに、
  CPPN が幾何的に構造化された射影行列を学習する。

  入力基板: proj_dim ノード @ y=-1 (等間隔配置)
  出力基板: proj_dim ノード @ y=+1 (等間隔配置)
  隠れ基板: hidden_dim ノード @ y=0  (ES: 出力から動的配置)

  CPPN: (x_src, y_src, x_tgt, y_tgt) → weight
  → proj_dim × proj_dim の学習済み射影行列を生成

クラス構成:
  TRIDENTSubstrate      カスタム MLPSubstrate (入力→隠れ→出力の幾何的配置)
  ESHyperNEATProjector  CPPN ベースの射影器 (384→proj_dim)
  ESHyperNEATIndexer    A型 NeatIndexer の ES-HyperNEAT 版
"""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
import jax
import jax.numpy as jnp
from jax import vmap

from tensorneat.algorithm.hyperneat import HyperNEAT
from tensorneat.algorithm.hyperneat.hyperneat import HyperNEATNode, HyperNEATConn
from tensorneat.algorithm.hyperneat.substrate import MLPSubstrate
from tensorneat.algorithm.neat import NEAT
from tensorneat.genome import DefaultGenome, RecurrentGenome
from tensorneat.common import State, ACT, AGG
from tensorneat import Pipeline
from tensorneat.problem.base import BaseProblem


# ──────────────────────────────────────────────
# TensorNEAT v0.1.0 互換パッチ
# RecurrentGenome.forward が valid_mask を渡すが HyperNEATNode に引数がない問題を修正
# ──────────────────────────────────────────────

class _PatchedHyperNEATNode(HyperNEATNode):
    """valid_mask 引数を受け入れる HyperNEATNode の互換パッチ。

    TensorNEAT 0.1.0 では AGG 関数が (inputs, mask) を要求するため、
    valid_mask を aggregation に渡す必要がある。
    """

    def forward(self, state, attrs, inputs, is_output_node=False, valid_mask=None):  # noqa: signature-mismatch
        if valid_mask is None:
            valid_mask = ~jnp.isnan(inputs)
        agg_out = self.aggregation(inputs, valid_mask)
        return jax.lax.cond(
            is_output_node,
            lambda: agg_out,
            lambda: self.activation(agg_out),
        )


class _PatchedHyperNEAT(HyperNEAT):
    """_PatchedHyperNEATNode を使う HyperNEAT サブクラス。"""

    def __init__(self, substrate, neat, weight_threshold=0.3, max_weight=5.0,
                 aggregation=AGG.sum, activation=ACT.sigmoid,
                 activate_time=10, output_transform=ACT.sigmoid):
        super().__init__(substrate, neat, weight_threshold, max_weight,
                         aggregation, activation, activate_time, output_transform)
        # hyper_genome をパッチ済みノードで置き換える
        self.hyper_genome = RecurrentGenome(
            num_inputs=substrate.num_inputs,
            num_outputs=substrate.num_outputs,
            max_nodes=substrate.nodes_cnt,
            max_conns=substrate.conns_cnt,
            node_gene=_PatchedHyperNEATNode(aggregation, activation),
            conn_gene=HyperNEATConn(),
            activate_time=activate_time,
            output_transform=output_transform,
        )


# ──────────────────────────────────────────────
# TRIDENTSubstrate: カスタム幾何的基板
# ──────────────────────────────────────────────

def make_trident_substrate(
    proj_dim: int,
    hidden_dim: int,
) -> MLPSubstrate:
    """
    TRIDENT 用の 3 層 MLP 基板を生成する。

    レイアウト:
      Layer 0 (y=-1): proj_dim 入力ノード (等間隔 x ∈ [-1, 1])
      Layer 1 (y= 0): hidden_dim 隠れノード (等間隔 x ∈ [-1, 1])
      Layer 2 (y=+1): proj_dim 出力ノード (等間隔 x ∈ [-1, 1])

    CPPN の入力: (x_src, y_src, x_tgt, y_tgt)  4次元
    CPPN の出力: weight (スカラー)

    Parameters
    ----------
    proj_dim   : 入力・出力層のノード数 (射影次元)
    hidden_dim : 隠れ層のノード数 (ES: 動的配置の候補数)

    Returns
    -------
    MLPSubstrate
    """
    # HyperNEAT は forward 時にバイアスを 1 つ自動付加するため
    # substrate の入力層は proj_dim + 1 にして num_inputs = proj_dim を確保する
    #
    # RecurrentGenome の検証: max_conns (= substrate.conns_cnt) >= num_inputs * num_outputs
    #   num_inputs  = proj_dim + 1 (バイアス込み入力層)
    #   num_outputs = proj_dim
    #   必要 conns_cnt >= (proj_dim+1) * proj_dim
    # MLP の conns_cnt = hidden_dim * (2*proj_dim + 1)
    #   → min_hidden = ceil((proj_dim+1)*proj_dim / (2*proj_dim+1))
    min_conns_needed = (proj_dim + 1) * proj_dim
    min_hidden = (min_conns_needed + 2 * proj_dim) // (2 * proj_dim + 1)
    eff_hidden = max(hidden_dim, min_hidden)
    return MLPSubstrate(
        layers=[proj_dim + 1, eff_hidden, proj_dim],
        coor_range=(-1.0, 1.0, -1.0, 1.0),
    )


# ──────────────────────────────────────────────
# ES-HyperNEAT Projection Problem
# ──────────────────────────────────────────────

class ProjectionFitProblem(BaseProblem):
    """
    CPPN が生成する射影行列の品質を評価する適合度問題。

    HyperNEAT の forward は (proj_dim,) → (proj_dim,) を行う。
    この写像をコサイン類似度保存射影として学習する。

    fitness = mean cosine_similarity(CPPN(q), target)

    MSE ではなくコサイン類似度を使う。MSE は zero-output が
    local optimum になるが、コサイン類似度は zero-output を最悪値 (-1) とするため
    CPPN が non-zero な射影を学習するよう誘導する。
    """

    jitable = True

    def __init__(
        self,
        queries_proj: np.ndarray,   # (Q, proj_dim) 参照射影済みクエリ
        targets_proj: np.ndarray,   # (Q, proj_dim) 目標出力 (= 入力と同じ: 恒等写像に近い)
    ):
        super().__init__()
        self._queries = jnp.array(queries_proj, dtype=jnp.float32)
        self._targets = jnp.array(targets_proj, dtype=jnp.float32)
        self._proj_dim = queries_proj.shape[1]

    @property
    def input_shape(self):
        return (self._proj_dim,)

    @property
    def output_shape(self):
        return (self._proj_dim,)

    def setup(self, state: State = State()):
        return state

    def evaluate(self, state, randkey, act_func, params) -> float:
        """fitness = mean cosine similarity (CPPN_output, target)

        MSE の代わりにコサイン類似度を使う理由:
          MSE は zero-output (全結合重み = 0) を local optimum として持つ。
          単位正規化ターゲットに対して MSE(0, t) = 1/dim が達成可能で、
          初期世代ではランダム射影より良い場合がある。
          コサイン距離は zero-output を -1.0 (最悪値) とするため
          CPPN が non-zero な射影を学習するよう誘導する。
        """
        def cosine_sim(q, t):
            out = act_func(state, params, q)
            out_norm = out / (jnp.linalg.norm(out) + 1e-8)
            t_norm   = t   / (jnp.linalg.norm(t)   + 1e-8)
            return jnp.dot(out_norm, t_norm)

        sims = vmap(cosine_sim)(self._queries, self._targets)
        return jnp.mean(sims)

    def show(self, state, randkey, act_func, params, *args, **kwargs):
        fit = float(self.evaluate(state, randkey, act_func, params))
        print(f"ProjectionFitProblem fitness (neg-MSE): {fit:.4f}")


# ──────────────────────────────────────────────
# ESHyperNEATProjector: CPPN ベース射影器
# ──────────────────────────────────────────────

class ESHyperNEATProjector:
    """
    ES-HyperNEAT による幾何的射影器。

    通常の固定ランダム射影行列 R の代わりに、
    CPPN が基板幾何から学習した構造化射影を使用する。

    入力: f32[proj_dim]   (前段で 384→proj_dim にスケッチ済み)
    出力: f32[proj_dim]   (CPPN が幾何的に変換した特徴)

    Parameters
    ----------
    proj_dim          : 射影次元 (入力 = 出力次元)
    hidden_dim        : 基板の隠れ層ノード数
    cppn_pop_size     : CPPN を進化させる NEAT 集団サイズ
    cppn_species_size : 種数
    generation_limit  : 進化世代数
    weight_threshold  : HyperNEAT の重み閾値
    max_weight        : HyperNEAT の最大重み
    seed              : 乱数シード
    """

    def __init__(
        self,
        proj_dim: int = 32,
        hidden_dim: int = 16,
        cppn_pop_size: int = 30,
        cppn_species_size: int = 5,
        generation_limit: int = 50,
        weight_threshold: float = 0.3,
        max_weight: float = 3.0,
        seed: int = 0,
    ):
        self.proj_dim         = proj_dim
        self.hidden_dim       = hidden_dim
        self.cppn_pop_size    = cppn_pop_size
        self.cppn_species_size = cppn_species_size
        self.generation_limit = generation_limit
        self.weight_threshold = weight_threshold
        self.max_weight       = max_weight
        self.seed             = seed

        self._state  = None
        self._best   = None
        self._pipeline = None
        self._hyperneat = None

    def fit(
        self,
        queries_proj: np.ndarray,
        targets_proj: Optional[np.ndarray] = None,
        fitness_target: float = 0.9,
    ) -> "ESHyperNEATProjector":
        """
        CPPN を進化させて射影行列を学習する。

        Parameters
        ----------
        queries_proj : (Q, proj_dim) 入力 (前段ランダム射影済み)
        targets_proj : (Q, proj_dim) 目標出力 (None なら queries_proj をそのまま使用)
        fitness_target : 到達目標 fitness (cosine sim, range [-1, 1])
        """
        if targets_proj is None:
            targets_proj = queries_proj  # 恒等写像が初期目標

        substrate = make_trident_substrate(self.proj_dim, self.hidden_dim)

        # CPPN は基板の query_coors 次元 (4次元) を入力とする
        cppn_genome = DefaultGenome(
            num_inputs=substrate.query_coors.shape[1],   # 4
            num_outputs=1,                                 # weight スカラー
            max_nodes=max(30, substrate.query_coors.shape[1] * 2 + 10),
            max_conns=max(50, substrate.query_coors.shape[1] * 10 + 10),
        )


        neat_cppn = NEAT(
            genome=cppn_genome,
            pop_size=self.cppn_pop_size,
            species_size=self.cppn_species_size,
        )

        self._hyperneat = _PatchedHyperNEAT(
            substrate=substrate,
            neat=neat_cppn,
            weight_threshold=self.weight_threshold,
            max_weight=self.max_weight,
            activation=ACT.tanh,
            output_transform=ACT.tanh,
        )

        problem = ProjectionFitProblem(
            queries_proj=queries_proj.astype(np.float32),
            targets_proj=targets_proj.astype(np.float32),
        )

        self._pipeline = Pipeline(
            algorithm=self._hyperneat,
            problem=problem,
            seed=self.seed,
            fitness_target=fitness_target,
            generation_limit=self.generation_limit,
        )

        print(f"[ESHyperNEATProjector] 進化開始: "
              f"proj_dim={self.proj_dim}, hidden_dim={self.hidden_dim}, "
              f"cppn_pop={self.cppn_pop_size}, max_gen={self.generation_limit}")
        init_state = self._pipeline.setup()
        self._state, self._best = self._pipeline.auto_run(init_state)
        print("[ESHyperNEATProjector] 進化完了")
        return self

    def project(self, x: np.ndarray) -> np.ndarray:
        """
        入力ベクトルを CPPN 射影で変換する。

        Parameters
        ----------
        x : (proj_dim,) または (N, proj_dim)

        Returns
        -------
        projected : 同 shape
        """
        assert self._state is not None, "fit() を先に実行してください"

        single = x.ndim == 1
        xb = jnp.array(x[None] if single else x, dtype=jnp.float32)

        best_t = self._pipeline.algorithm.transform(self._state, self._best)
        out = vmap(
            lambda q: self._pipeline.algorithm.forward(self._state, best_t, q)
        )(xb)

        result = np.array(out)
        return result[0] if single else result

    def projection_matrix(self, normalize: bool = True) -> np.ndarray:
        """
        標準基底を射影して得られる重み行列 W (proj_dim, proj_dim) を返す。

        W[:, j] = CPPN(e_j)  where e_j は j 番目の標準基底ベクトル

        Parameters
        ----------
        normalize : 各列を L2 正規化するか

        Returns
        -------
        W : (proj_dim, proj_dim)
        """
        assert self._state is not None

        eye = jnp.eye(self.proj_dim, dtype=jnp.float32)  # (D, D)
        W = self.project(np.array(eye))                   # (D, D)

        if normalize:
            norms = np.linalg.norm(W, axis=0, keepdims=True) + 1e-8
            W = W / norms

        return W

    @property
    def is_fitted(self) -> bool:
        return self._state is not None


# ──────────────────────────────────────────────
# ESHyperNEATIndexer: A型 with ES-HyperNEAT 射影
# ──────────────────────────────────────────────

class ESHyperNEATIndexer:
    """
    ES-HyperNEAT を使った A型 近傍索引器。

    パイプライン:
      384 次元入力
        ↓ sketch_proj (384→proj_dim, 固定ランダム直交)
      proj_dim 次元
        ↓ ESHyperNEATProjector (CPPN 学習済み)
      proj_dim 次元 (構造化特徴)
        ↓ コサイン類似度でコーパスと比較
      k 近傍インデックス

    ESHyperNEATProjector が固定ランダム射影より
    コーパスの幾何構造に適応した射影を学習する。

    Parameters
    ----------
    input_dim   : 元の入力次元 (384)
    proj_dim    : 射影後次元 (32)
    hidden_dim  : CPPN 基板の隠れ層サイズ
    ...         : ESHyperNEATProjector の残りパラメータ
    """

    def __init__(
        self,
        input_dim: int = 384,
        proj_dim: int = 32,
        hidden_dim: int = 16,
        cppn_pop_size: int = 30,
        cppn_species_size: int = 5,
        generation_limit: int = 50,
        k: int = 5,
        seed: int = 0,
    ):
        self.input_dim  = input_dim
        self.proj_dim   = proj_dim
        self.k          = k
        self.seed       = seed

        # Stage 1: 固定ランダム直交スケッチ (384 → proj_dim)
        rng = np.random.default_rng(seed)
        R_raw = rng.standard_normal((input_dim, proj_dim)).astype(np.float32)
        self._sketch = R_raw / (np.linalg.norm(R_raw, axis=0, keepdims=True) + 1e-8)

        # Stage 2: ES-HyperNEAT 射影器 (proj_dim → proj_dim)
        self._projector = ESHyperNEATProjector(
            proj_dim=proj_dim,
            hidden_dim=hidden_dim,
            cppn_pop_size=cppn_pop_size,
            cppn_species_size=cppn_species_size,
            generation_limit=generation_limit,
            seed=seed,
        )

        self._corpus_proj: Optional[np.ndarray] = None  # (C, proj_dim) 射影済みコーパス

    def _sketch_and_norm(self, x: np.ndarray) -> np.ndarray:
        """入力を sketch 後に正規化する。"""
        out = x @ self._sketch   # (..., proj_dim)
        out = out / (np.linalg.norm(out, axis=-1, keepdims=True) + 1e-8)
        return out

    def fit(
        self,
        corpus: np.ndarray,
        queries: Optional[np.ndarray] = None,
        fitness_target: float = 0.9,
    ) -> "ESHyperNEATIndexer":
        """
        コーパスに対して ES-HyperNEAT 射影器を進化させる。

        Parameters
        ----------
        corpus  : (C, input_dim)
        queries : (Q, input_dim) または None
        """
        # Stage 1: スケッチ
        corpus_s = self._sketch_and_norm(corpus)   # (C, proj_dim)

        if queries is None:
            rng = np.random.default_rng(self.seed)
            idx = rng.choice(len(corpus), size=min(20, len(corpus)), replace=False)
            queries_s = corpus_s[idx]
        else:
            queries_s = self._sketch_and_norm(queries)

        # Stage 2: CPPN が queries_s → queries_s を恒等写像として学習
        # (コーパスの幾何構造を保持する射影を学習)
        self._projector.fit(
            queries_proj=queries_s,
            targets_proj=queries_s,
            fitness_target=fitness_target,
        )

        # コーパスを射影して保存
        self._corpus_proj = self._projector.project(corpus_s)  # (C, proj_dim)
        # 正規化
        norms = np.linalg.norm(self._corpus_proj, axis=1, keepdims=True) + 1e-8
        self._corpus_proj = self._corpus_proj / norms

        return self

    def search(
        self, query: np.ndarray, k: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        クエリに対して k 近傍を返す。

        Parameters
        ----------
        query : (input_dim,)

        Returns
        -------
        (indices, scores)
        """
        assert self._corpus_proj is not None, "fit() を先に実行してください"
        k = k or self.k

        # Stage 1: スケッチ
        q_s = self._sketch_and_norm(query[None])[0]     # (proj_dim,)
        # Stage 2: CPPN 射影
        q_p = self._projector.project(q_s)               # (proj_dim,)
        q_p = q_p / (np.linalg.norm(q_p) + 1e-8)

        scores = self._corpus_proj @ q_p                  # (C,)
        top_k  = np.argsort(-scores)[:k]
        return top_k, scores[top_k]

    def projection_matrix(self) -> np.ndarray:
        """学習済み CPPN 射影行列 W (proj_dim, proj_dim) を返す。"""
        return self._projector.projection_matrix()

    def bcs_descriptor(self, queries: np.ndarray) -> np.ndarray:
        """MAP-Elites BCS 記述子 [活性化強度, 0.0] を返す。"""
        assert self._corpus_proj is not None

        q_s = self._sketch_and_norm(queries)        # (Q, proj_dim)
        q_p = self._projector.project(q_s)          # (Q, proj_dim)

        mean_mag = float(np.mean(np.abs(q_p)))
        act_intensity = float(1.0 / (1.0 + np.exp(-mean_mag)))  # sigmoid
        return np.array([act_intensity, 0.0], dtype=np.float32)
