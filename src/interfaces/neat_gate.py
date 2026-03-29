"""
TRIDENT — B型 I/F: NeatGate
NEATトポロジによる文脈ベクトル→スキル有効化マスクのゲート。
GRPO報酬判定・スキル選択・信頼度フィルタに対応。

入力 : 文脈ベクトル f32[context_dim]
出力 : スキル有効化マスク bool[num_skills]
BCS  : (発火率, 特異性=エントロピー逆数)  格子 16×16
"""

from __future__ import annotations

from typing import Callable, Literal, Optional
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
# カスタム適合度問題: バイナリゲート分類
# ──────────────────────────────────────────────

class BinaryGateProblem(BaseProblem):
    """
    文脈ベクトル → スキル有効化マスクを学習する適合度問題。

    NEAT ゲノムは f32[context_dim] → f32[num_skills] を出力する。
    sigmoid(出力) > threshold で bool マスクを生成し、
    目標マスクとの Binary Cross Entropy を最小化する。

    fitness = -BCE(predicted_logits, target_mask)
    """

    jitable = True

    def __init__(
        self,
        contexts: np.ndarray,
        targets: np.ndarray,
        threshold: float = 0.5,
    ):
        """
        Parameters
        ----------
        contexts : (N, context_dim) 文脈ベクトル群
        targets  : (N, num_skills)  目標活性化マスク (0.0 or 1.0)
        threshold: 出力を bool に変換する閾値 (BCS 計算用)
        """
        super().__init__()
        assert contexts.ndim == 2 and targets.ndim == 2
        assert contexts.shape[0] == targets.shape[0]
        assert np.all((targets == 0) | (targets == 1)), "targets は 0/1 のみ"

        self._contexts = jnp.array(contexts, dtype=jnp.float32)   # (N, C)
        self._targets  = jnp.array(targets,  dtype=jnp.float32)   # (N, K)
        self.threshold = threshold
        self._context_dim = contexts.shape[1]
        self._num_skills  = targets.shape[1]

    @property
    def input_shape(self):
        return (self._context_dim,)

    @property
    def output_shape(self):
        return (self._num_skills,)

    def setup(self, state: State = State()):
        return state

    def evaluate(self, state: State, randkey, act_func, params) -> float:
        """
        fitness = -mean_BCE(sigmoid(logits), targets)

        BCE は数値安定版: -[y*log(p) + (1-y)*log(1-p)]
        """
        eps = 1e-7

        def bce_sample(context, target):
            logits = act_func(state, params, context)       # f32[K]
            probs  = jax.nn.sigmoid(logits)                  # f32[K]
            probs  = jnp.clip(probs, eps, 1.0 - eps)
            bce    = -(target * jnp.log(probs) + (1.0 - target) * jnp.log(1.0 - probs))
            return jnp.mean(bce)

        losses = vmap(bce_sample)(self._contexts, self._targets)  # (N,)
        return -jnp.mean(losses)

    def show(self, state, randkey, act_func, params, *args, **kwargs):
        fitness = float(self.evaluate(state, randkey, act_func, params))
        # 精度も計算
        def predict(context):
            logits = act_func(state, params, context)
            return (jax.nn.sigmoid(logits) > self.threshold).astype(jnp.float32)

        preds = vmap(predict)(self._contexts)             # (N, K)
        acc = float(jnp.mean(preds == self._targets))
        print(f"BinaryGateProblem — fitness (neg-BCE): {fitness:.4f}, accuracy: {acc:.4f}")


# ──────────────────────────────────────────────
# BCS 記述子計算
# ──────────────────────────────────────────────

def compute_gate_bcs_descriptor(
    activation_masks: np.ndarray,
) -> np.ndarray:
    """
    MAP-Elites BCS 記述子 (B型ゲート用)。

    Parameters
    ----------
    activation_masks : (N, K) bool/float マスク (0/1)

    Returns
    -------
    descriptor : f32[2]
        [firing_rate, specificity]
        どちらも [0, 1] に正規化済み

    BCS 定義
    --------
    軸1 firing_rate  = 全マスクの平均発火率 ∈ [0, 1]
    軸2 specificity  = 1 / (1 + H)  where H = シャノンエントロピー
                       H=0 → specificity=1 (完全特異)
                       H 大 → specificity≈0 (一様発火)
    """
    masks = np.asarray(activation_masks, dtype=np.float32)  # (N, K)

    # 発火率: 全要素の平均
    firing_rate = float(np.mean(masks))

    # 特異性: スキルごとの発火確率分布のエントロピー
    skill_prob = np.mean(masks, axis=0)  # (K,) 各スキルの発火確率
    eps = 1e-8
    skill_prob = np.clip(skill_prob, eps, 1.0 - eps)
    entropy = -np.mean(
        skill_prob * np.log2(skill_prob) + (1.0 - skill_prob) * np.log2(1.0 - skill_prob)
    )  # スカラー (最大1 bit / スキル)
    specificity = float(1.0 / (1.0 + entropy))

    return np.array([firing_rate, specificity], dtype=np.float32)


def gate_accuracy(
    act_func,
    state,
    params,
    contexts: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """ゲートの分類精度 (accuracy) を返す。"""
    preds = np.array(
        vmap(lambda c: jax.nn.sigmoid(act_func(state, params, jnp.array(c, jnp.float32))) > threshold)(
            jnp.array(contexts, jnp.float32)
        )
    )  # (N, K)
    return float(np.mean(preds == (targets > 0.5)))


# ──────────────────────────────────────────────
# NeatGate (B型 I/F メインクラス)
# ──────────────────────────────────────────────

class NeatGate:
    """
    B型 I/F: binary activation ベースの NEAT ゲートモジュール。

    NEAT で文脈ベクトル → スキル有効化マスクを学習し、
    GRPO 報酬判定・スキル選択・信頼度フィルタに使用する。

    Parameters
    ----------
    context_dim       : 入力文脈ベクトルの次元
    num_skills        : スキル数 (出力マスクのビット数)
    threshold         : sigmoid 出力を bool に変換する閾値
    pop_size          : NEAT 集団サイズ
    species_size      : 種数
    max_nodes         : ゲノムの最大ノード数
    max_conns         : ゲノムの最大コネクション数
    generation_limit  : 進化の最大世代数
    seed              : 乱数シード
    """

    def __init__(
        self,
        context_dim: int = 16,
        num_skills: int = 8,
        threshold: float = 0.5,
        pop_size: int = 50,
        species_size: int = 5,
        max_nodes: int = 50,
        max_conns: int = 100,
        generation_limit: int = 100,
        seed: int = 42,
    ):
        self.context_dim  = context_dim
        self.num_skills   = num_skills
        self.threshold    = threshold
        self.pop_size     = pop_size
        self.species_size = species_size
        self.max_nodes    = max_nodes
        self.max_conns    = max_conns
        self.generation_limit = generation_limit
        self.seed = seed

        self._state: Optional[State] = None
        self._best_params = None
        self._pipeline: Optional[Pipeline] = None
        self._problem: Optional[BinaryGateProblem] = None

    def fit(
        self,
        contexts: np.ndarray,
        targets: np.ndarray,
        fitness_target: float = -0.05,
    ) -> "NeatGate":
        """
        文脈→マスクのペアで NEAT を進化させる。

        Parameters
        ----------
        contexts : (N, context_dim) 学習用文脈ベクトル
        targets  : (N, num_skills)  目標活性化マスク (0/1)
        fitness_target : 到達目標 fitness

        Returns
        -------
        self
        """
        assert contexts.shape[1] == self.context_dim, (
            f"context_dim mismatch: expected {self.context_dim}, got {contexts.shape[1]}"
        )
        assert targets.shape[1] == self.num_skills, (
            f"num_skills mismatch: expected {self.num_skills}, got {targets.shape[1]}"
        )

        self._problem = BinaryGateProblem(
            contexts=contexts.astype(np.float32),
            targets=targets.astype(np.float32),
            threshold=self.threshold,
        )

        # 初期ノード数・コネクション数の下限を自動計算
        min_nodes = self.context_dim + self.num_skills + 10
        min_conns = self.context_dim * self.num_skills + 10
        eff_nodes = max(self.max_nodes, min_nodes)
        eff_conns = max(self.max_conns, min_conns)

        genome = DefaultGenome(
            num_inputs=self.context_dim,
            num_outputs=self.num_skills,
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

        print(f"[NeatGate] 進化開始: pop={self.pop_size}, "
              f"context_dim={self.context_dim}, num_skills={self.num_skills}, "
              f"max_gen={self.generation_limit}")
        init_state = self._pipeline.setup()
        self._state, self._best_params = self._pipeline.auto_run(init_state)
        print("[NeatGate] 進化完了")
        return self

    def _forward(self, context: jnp.ndarray) -> jnp.ndarray:
        """内部: context f32[C] → logits f32[K]"""
        best_transformed = self._pipeline.algorithm.transform(
            self._state, self._best_params
        )
        return self._pipeline.algorithm.forward(
            self._state, best_transformed, context
        )

    def activate(self, context: np.ndarray) -> np.ndarray:
        """
        文脈ベクトルをスキル有効化マスク (bool) に変換する。

        Parameters
        ----------
        context : (C,) または (N, C)

        Returns
        -------
        mask : bool[(K,)] または bool[(N, K)]
        """
        assert self._state is not None, "fit() を先に実行してください"

        single = context.ndim == 1
        c = jnp.array(context[None] if single else context, dtype=jnp.float32)

        best_transformed = self._pipeline.algorithm.transform(
            self._state, self._best_params
        )
        logits = vmap(
            lambda x: self._pipeline.algorithm.forward(self._state, best_transformed, x)
        )(c)  # (N, K)

        masks = np.array(jax.nn.sigmoid(logits)) > self.threshold  # bool (N, K)
        return masks[0] if single else masks

    def logits(self, context: np.ndarray) -> np.ndarray:
        """
        sigmoid 前の生ロジットを返す (確率的用途向け)。

        Returns
        -------
        logits : f32[(K,)] または f32[(N, K)]
        """
        assert self._state is not None, "fit() を先に実行してください"

        single = context.ndim == 1
        c = jnp.array(context[None] if single else context, dtype=jnp.float32)

        best_transformed = self._pipeline.algorithm.transform(
            self._state, self._best_params
        )
        out = vmap(
            lambda x: self._pipeline.algorithm.forward(self._state, best_transformed, x)
        )(c)  # (N, K)

        result = np.array(out)
        return result[0] if single else result

    def probs(self, context: np.ndarray) -> np.ndarray:
        """sigmoid 後の確率を返す。"""
        return 1.0 / (1.0 + np.exp(-self.logits(context)))

    def accuracy(self, contexts: np.ndarray, targets: np.ndarray) -> float:
        """分類精度を計算する。"""
        assert self._state is not None
        best_transformed = self._pipeline.algorithm.transform(
            self._state, self._best_params
        )
        return gate_accuracy(
            act_func=self._pipeline.algorithm.forward,
            state=self._state,
            params=best_transformed,
            contexts=contexts,
            targets=targets,
            threshold=self.threshold,
        )

    def bcs_descriptor(self, contexts: np.ndarray) -> np.ndarray:
        """
        MAP-Elites BCS 記述子 [発火率, 特異性] を返す。

        Returns
        -------
        descriptor : f32[2]
        """
        assert self._state is not None
        masks = self.activate(contexts)  # (N, K) bool
        return compute_gate_bcs_descriptor(masks.astype(np.float32))


# ──────────────────────────────────────────────
# NeatAugmentedReward: GRPO 報酬と NeatGate の統合
# ──────────────────────────────────────────────

class NeatAugmentedReward:
    """
    GRPO 既存報酬関数に NeatGate のスキル選択を組み込んだ拡張報酬。

    NeatGate が有効化したスキルのみ base_reward で評価し、
    スキル選択の正確性で報酬をスケーリングする。

    Parameters
    ----------
    neat_gate    : 学習済み NeatGate (B型)
    base_reward  : 既存 GRPO 報酬関数 (context, action) -> float
    gate_weight  : ゲートスコアの報酬への重み [0, 1]
    """

    def __init__(
        self,
        neat_gate: NeatGate,
        base_reward: Callable,
        gate_weight: float = 0.3,
    ):
        self.neat_gate   = neat_gate
        self.base_reward = base_reward
        self.gate_weight = gate_weight

    def __call__(self, context: np.ndarray, action: np.ndarray) -> float:
        """
        拡張報酬 = (1 - gate_weight) * base_reward
                 + gate_weight * gate_confidence_score

        gate_confidence_score: 選択されたスキルの sigmoid 確率の平均
        """
        base = float(self.base_reward(context, action))

        probs = self.neat_gate.probs(context)            # f32[K]
        mask  = self.neat_gate.activate(context)         # bool[K]
        active_probs = probs[mask] if mask.any() else np.array([0.5])
        gate_score = float(np.mean(active_probs))

        return (1.0 - self.gate_weight) * base + self.gate_weight * gate_score
