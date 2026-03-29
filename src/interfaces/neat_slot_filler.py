"""
TRIDENT — C型 I/F: NeatSlotFiller
NEATトポロジによる文脈ベクトル→名前付きスロット値生成。
KG操作・SQLスロット埋め・GRPO報酬スロット生成に対応。

入力 : 文脈ベクトル f32[context_dim]
出力 : {slot_name: f32}  構造化スロット値
BCS  : (スロット充填率, 出力正確性)  格子 16×16
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple
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
# スロットスキーマ定義
# ──────────────────────────────────────────────

# 用途別スキーマ
KG_SCHEMA = ("node", "relation", "weight")
SQL_SCHEMA = ("table", "col", "condition")
GRPO_SCHEMA = ("reward", "reason", "target")


# ──────────────────────────────────────────────
# カスタム適合度問題: 名前付きスロット回帰
# ──────────────────────────────────────────────

class SlotFitProblem(BaseProblem):
    """
    文脈ベクトル → スロット値ベクトルを学習する適合度問題。

    NEAT ゲノムは f32[context_dim] → f32[num_slots] を出力する。
    各スロットは tanh で [-1, 1] に正規化された連続値。
    目標スロット値との MSE を最小化する。

    fitness = -MSE(predicted_slots, target_slots)
    """

    jitable = True

    def __init__(
        self,
        contexts: np.ndarray,
        targets: np.ndarray,
        fill_threshold: float = 0.1,
    ):
        """
        Parameters
        ----------
        contexts        : (N, context_dim) 文脈ベクトル群
        targets         : (N, num_slots)   目標スロット値 ([-1, 1] 正規化済み)
        fill_threshold  : スロットが「充填済み」とみなす絶対値の閾値
        """
        super().__init__()
        assert contexts.ndim == 2 and targets.ndim == 2
        assert contexts.shape[0] == targets.shape[0]

        self._contexts = jnp.array(contexts, dtype=jnp.float32)
        self._targets  = jnp.array(targets,  dtype=jnp.float32)
        self.fill_threshold = fill_threshold
        self._context_dim = contexts.shape[1]
        self._num_slots   = targets.shape[1]

    @property
    def input_shape(self):
        return (self._context_dim,)

    @property
    def output_shape(self):
        return (self._num_slots,)

    def setup(self, state: State = State()):
        return state

    def evaluate(self, state: State, randkey, act_func, params) -> float:
        """fitness = -MSE(tanh(NEAT出力), targets)"""
        def mse_sample(context, target):
            raw = act_func(state, params, context)      # f32[S]
            out = jnp.tanh(raw)                          # [-1, 1] に圧縮
            return jnp.mean((out - target) ** 2)

        losses = vmap(mse_sample)(self._contexts, self._targets)
        return -jnp.mean(losses)

    def show(self, state, randkey, act_func, params, *args, **kwargs):
        fitness = float(self.evaluate(state, randkey, act_func, params))

        def predict(context):
            return jnp.tanh(act_func(state, params, context))

        preds = np.array(vmap(predict)(self._contexts))
        targets = np.array(self._targets)
        fill_rate = float(np.mean(np.abs(preds) > self.fill_threshold))
        print(f"SlotFitProblem — fitness (neg-MSE): {fitness:.4f}, "
              f"fill_rate: {fill_rate:.3f}")


# ──────────────────────────────────────────────
# BCS 記述子計算
# ──────────────────────────────────────────────

def compute_slot_bcs_descriptor(
    predicted_slots: np.ndarray,
    target_slots: np.ndarray,
    fill_threshold: float = 0.1,
) -> np.ndarray:
    """
    MAP-Elites BCS 記述子 (C型スロット用)。

    Parameters
    ----------
    predicted_slots : (N, S) 予測スロット値 (tanh後, [-1,1])
    target_slots    : (N, S) 目標スロット値
    fill_threshold  : 充填判定の絶対値閾値

    Returns
    -------
    descriptor : f32[2]
        [fill_rate, accuracy]

    BCS 定義
    --------
    軸1 fill_rate  = |predicted| > threshold の割合 ∈ [0, 1]
    軸2 accuracy   = 1 - normalized_MSE ∈ [0, 1]
                     nMSE = MSE / (var(targets) + eps)
    """
    preds = np.asarray(predicted_slots, dtype=np.float32)
    tgts  = np.asarray(target_slots,    dtype=np.float32)

    fill_rate = float(np.mean(np.abs(preds) > fill_threshold))

    mse = float(np.mean((preds - tgts) ** 2))
    var = float(np.var(tgts)) + 1e-8
    nmse = mse / var
    accuracy = float(np.clip(1.0 - nmse, 0.0, 1.0))

    return np.array([fill_rate, accuracy], dtype=np.float32)


# ──────────────────────────────────────────────
# NeatSlotFiller (C型 I/F メインクラス)
# ──────────────────────────────────────────────

class NeatSlotFiller:
    """
    C型 I/F: named slots ベースの NEAT スロット生成器。

    NEAT で文脈ベクトル → 名前付きスロット値を学習し、
    KG エンティティ操作・SQLスロット埋め・GRPO報酬スロット生成に使用する。

    Parameters
    ----------
    slot_names        : スロット名のリスト (例: ["node", "relation", "weight"])
    context_dim       : 入力文脈ベクトルの次元
    fill_threshold    : スロット充填判定の閾値 (|value| > threshold)
    pop_size          : NEAT 集団サイズ
    species_size      : 種数
    max_nodes         : ゲノムの最大ノード数
    max_conns         : ゲノムの最大コネクション数
    generation_limit  : 進化の最大世代数
    seed              : 乱数シード
    """

    def __init__(
        self,
        slot_names: Tuple[str, ...],
        context_dim: int = 16,
        fill_threshold: float = 0.1,
        pop_size: int = 50,
        species_size: int = 5,
        max_nodes: int = 50,
        max_conns: int = 100,
        generation_limit: int = 100,
        seed: int = 42,
    ):
        self.slot_names    = tuple(slot_names)
        self.num_slots     = len(slot_names)
        self.context_dim   = context_dim
        self.fill_threshold = fill_threshold
        self.pop_size      = pop_size
        self.species_size  = species_size
        self.max_nodes     = max_nodes
        self.max_conns     = max_conns
        self.generation_limit = generation_limit
        self.seed = seed

        self._state: Optional[State] = None
        self._best_params = None
        self._pipeline: Optional[Pipeline] = None
        self._problem: Optional[SlotFitProblem] = None

    def fit(
        self,
        contexts: np.ndarray,
        targets: np.ndarray,
        fitness_target: float = -0.05,
    ) -> "NeatSlotFiller":
        """
        文脈→スロット値のペアで NEAT を進化させる。

        Parameters
        ----------
        contexts : (N, context_dim) 学習用文脈ベクトル
        targets  : (N, num_slots)   目標スロット値 ([-1, 1] 正規化済み)
        fitness_target : 到達目標 fitness

        Returns
        -------
        self
        """
        assert contexts.shape[1] == self.context_dim, (
            f"context_dim mismatch: expected {self.context_dim}, got {contexts.shape[1]}"
        )
        assert targets.shape[1] == self.num_slots, (
            f"num_slots mismatch: expected {self.num_slots}, got {targets.shape[1]}"
        )

        self._problem = SlotFitProblem(
            contexts=contexts.astype(np.float32),
            targets=targets.astype(np.float32),
            fill_threshold=self.fill_threshold,
        )

        min_nodes = self.context_dim + self.num_slots + 10
        min_conns = self.context_dim * self.num_slots + 10
        eff_nodes = max(self.max_nodes, min_nodes)
        eff_conns = max(self.max_conns, min_conns)

        genome = DefaultGenome(
            num_inputs=self.context_dim,
            num_outputs=self.num_slots,
            max_nodes=eff_nodes,
            max_conns=eff_conns,
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

        print(f"[NeatSlotFiller] 進化開始: pop={self.pop_size}, "
              f"context_dim={self.context_dim}, slots={self.slot_names}, "
              f"max_gen={self.generation_limit}")
        init_state = self._pipeline.setup()
        self._state, self._best_params = self._pipeline.auto_run(init_state)
        print("[NeatSlotFiller] 進化完了")
        return self

    def _get_transformed(self):
        return self._pipeline.algorithm.transform(self._state, self._best_params)

    def fill(self, context: np.ndarray) -> Dict[str, float]:
        """
        文脈ベクトルからスロット辞書を生成する。

        Parameters
        ----------
        context : (context_dim,) 文脈ベクトル

        Returns
        -------
        slots : {slot_name: float}  各スロットの値 (tanh後, [-1, 1])
        """
        assert self._state is not None, "fit() を先に実行してください"
        assert context.ndim == 1

        best_transformed = self._get_transformed()
        raw = self._pipeline.algorithm.forward(
            self._state, best_transformed,
            jnp.array(context, dtype=jnp.float32)
        )
        values = np.array(jnp.tanh(raw))  # [-1, 1]
        return {name: float(values[i]) for i, name in enumerate(self.slot_names)}

    def fill_batch(self, contexts: np.ndarray) -> List[Dict[str, float]]:
        """
        バッチ処理でスロット辞書リストを生成する。

        Parameters
        ----------
        contexts : (N, context_dim)

        Returns
        -------
        slots_list : list of {slot_name: float}
        """
        assert self._state is not None, "fit() を先に実行してください"

        best_transformed = self._get_transformed()
        c = jnp.array(contexts, dtype=jnp.float32)
        raws = vmap(
            lambda x: self._pipeline.algorithm.forward(self._state, best_transformed, x)
        )(c)
        values = np.array(jnp.tanh(raws))  # (N, S)
        return [
            {name: float(values[i, j]) for j, name in enumerate(self.slot_names)}
            for i in range(len(contexts))
        ]

    def fill_rate(self, contexts: np.ndarray) -> float:
        """
        スロット充填率 (|value| > fill_threshold の割合) を返す。
        """
        assert self._state is not None

        best_transformed = self._get_transformed()
        c = jnp.array(contexts, dtype=jnp.float32)
        raws = vmap(
            lambda x: self._pipeline.algorithm.forward(self._state, best_transformed, x)
        )(c)
        values = np.array(jnp.tanh(raws))  # (N, S)
        return float(np.mean(np.abs(values) > self.fill_threshold))

    def bcs_descriptor(
        self,
        contexts: np.ndarray,
        targets: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        MAP-Elites BCS 記述子 [充填率, 正確性] を返す。

        Returns
        -------
        descriptor : f32[2]
        """
        assert self._state is not None

        best_transformed = self._get_transformed()
        c = jnp.array(contexts, dtype=jnp.float32)
        raws = vmap(
            lambda x: self._pipeline.algorithm.forward(self._state, best_transformed, x)
        )(c)
        preds = np.array(jnp.tanh(raws))  # (N, S)

        if targets is None:
            # 正確性不明の場合は充填率のみ計算
            fill_rate = float(np.mean(np.abs(preds) > self.fill_threshold))
            return np.array([fill_rate, 0.0], dtype=np.float32)

        return compute_slot_bcs_descriptor(preds, targets, self.fill_threshold)


# ──────────────────────────────────────────────
# NeatKGWriter: KG 操作への統合ラッパー
# ──────────────────────────────────────────────

class NeatKGWriter:
    """
    KG（ナレッジグラフ）操作のスロット生成を NeatSlotFiller で担う統合クラス。

    KG スキーマ: {node: f32, relation: f32, weight: f32}
    - node     : エンティティID の埋め込み次元インデックス [-1, 1]
    - relation : 関係タイプの埋め込み [-1, 1]
    - weight   : エッジ重み [0, 1] (tanh出力を 0.5 + 0.5*v で変換)

    Parameters
    ----------
    neat_filler : 学習済み NeatSlotFiller (KG_SCHEMA)
    kg_store    : 既存 KG ストア (write メソッドを持つ任意オブジェクト)
    """

    def __init__(
        self,
        neat_filler: NeatSlotFiller,
        kg_store=None,
    ):
        assert neat_filler.slot_names == KG_SCHEMA, (
            f"KG スキーマは {KG_SCHEMA} である必要があります。"
            f"受け取った: {neat_filler.slot_names}"
        )
        self.neat_filler = neat_filler
        self.kg_store    = kg_store

    def generate_triple(self, context: np.ndarray) -> Dict[str, float]:
        """
        文脈からKGトリプル {node, relation, weight} を生成する。
        weight は [0, 1] に変換済み。
        """
        slots = self.neat_filler.fill(context)
        slots["weight"] = 0.5 + 0.5 * slots["weight"]  # tanh → [0, 1]
        return slots

    def write(self, context: np.ndarray) -> Dict[str, float]:
        """
        KG トリプルを生成して kg_store に書き込む。

        Returns
        -------
        triple : {node, relation, weight}
        """
        triple = self.generate_triple(context)
        if self.kg_store is not None:
            self.kg_store.write(triple)
        return triple
