"""
TRIDENT — Novelty Search カスタム実装
JAX ネイティブで k-最近傍新規性スコアを計算し、
TRIDENTArchive の EvolutionLoop と統合する。

設計方針:
  - 行動ベクトル = BCS 記述子 f32[2] (A/B/C 型共通)
  - 新規性スコア = k-NN 平均距離 (アーカイブ + 一時バッファ 内)
  - アーカイブ追加基準:
      (a) 新規性スコア > threshold、または
      (b) 確率 p_add でランダム追加 (多様性確保)
  - 適合度 = 新規性スコア (fitness-free NS) または
             α * novelty + (1-α) * task_fitness (NS+F)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial


# ──────────────────────────────────────────────
# JAX ネイティブ k-NN 新規性スコア計算
# ──────────────────────────────────────────────

@partial(jax.jit, static_argnames=("k",))
def knn_novelty_scores(
    candidates: jnp.ndarray,
    archive: jnp.ndarray,
    k: int = 15,
) -> jnp.ndarray:
    """
    候補行動ベクトル群の k-NN 新規性スコアを JAX で一括計算する。

    Parameters
    ----------
    candidates : (N, D) 評価対象の行動ベクトル群
    archive    : (M, D) 既存アーカイブの行動ベクトル群 (M >= 1)
    k          : 近傍数 (static; archive サイズ以下である必要がある)

    Returns
    -------
    scores : (N,) 各候補の新規性スコア (k-NN 平均距離)
    """
    # (N, M) 距離行列
    diff  = candidates[:, None, :] - archive[None, :, :]    # (N, M, D)
    dists = jnp.sqrt(jnp.sum(diff ** 2, axis=-1) + 1e-12)  # (N, M)

    # lax.top_k は「最大 k 件」を返す → 負の距離で最小 k 件を選ぶ
    neg_top_k, _ = jax.lax.top_k(-dists, k)   # (N, k) 負値
    top_k_dists  = -neg_top_k                  # (N, k) 正値
    return jnp.mean(top_k_dists, axis=1)       # (N,)


@partial(jax.jit, static_argnames=("k",))
def single_novelty_score(
    candidate: jnp.ndarray,
    archive: jnp.ndarray,
    k: int = 15,
) -> jnp.ndarray:
    """
    単一候補の新規性スコアを計算する。

    Returns
    -------
    score : scalar jnp.ndarray
    """
    return knn_novelty_scores(candidate[None], archive, k)[0]


def novelty_score_with_buffer(
    candidate: np.ndarray,
    archive_vecs: np.ndarray,
    buffer_vecs: Optional[np.ndarray],
    k: int = 15,
) -> float:
    """
    アーカイブ + 一時バッファを合わせた新規性スコアを計算する。

    Parameters
    ----------
    candidate    : (D,) 評価対象の行動ベクトル
    archive_vecs : (M, D) 既存アーカイブ行動ベクトル (M >= 1)
    buffer_vecs  : (B, D) または None  一時バッファ
    k            : 近傍数

    Returns
    -------
    float
    """
    parts = [archive_vecs]
    if buffer_vecs is not None and len(buffer_vecs) > 0:
        parts.append(buffer_vecs)
    pool = np.concatenate(parts, axis=0)

    if len(pool) == 0:
        return 1.0

    k_eff  = min(k, len(pool))
    pool_j = jnp.array(pool, dtype=jnp.float32)
    cand_j = jnp.array(candidate, dtype=jnp.float32)
    return float(single_novelty_score(cand_j, pool_j, k=k_eff))


# ──────────────────────────────────────────────
# NoveltyArchive: 新規性探索専用のバッファ付きアーカイブ
# ──────────────────────────────────────────────

@dataclass
class NoveltyRecord:
    """新規性アーカイブに保存する1エントリ。"""
    behavior:      np.ndarray   # f32[D] 行動ベクトル (BCS 記述子)
    novelty_score: float
    skill_type:    str
    skill:         object = None
    task_fitness:  float  = 0.0
    metadata:      dict   = field(default_factory=dict)


class NoveltyArchive:
    """
    Novelty Search 用の行動ベクトルアーカイブ。

    内部に固定サイズのリングバッファを持ち、
    新規性の高い個体を蓄積する。

    Parameters
    ----------
    behavior_dim  : 行動ベクトルの次元 (BCS = 2)
    max_size      : アーカイブ最大サイズ (デフォルト 200)
    add_prob      : ランダム追加確率 (新規性閾値未達でも追加する確率)
    novelty_threshold : この値以上の新規性スコアは即時追加
    k_neighbors   : 新規性計算の k
    seed          : 乱数シード
    """

    def __init__(
        self,
        behavior_dim: int = 2,
        max_size: int = 200,
        add_prob: float = 0.05,
        novelty_threshold: float = 0.1,
        k_neighbors: int = 15,
        seed: int = 0,
    ):
        self.behavior_dim      = behavior_dim
        self.max_size          = max_size
        self.add_prob          = add_prob
        self.novelty_threshold = novelty_threshold
        self.k_neighbors       = k_neighbors
        self._rng              = np.random.default_rng(seed)

        self._behaviors: List[np.ndarray] = []  # 行動ベクトルのリスト
        self._records:   List[NoveltyRecord] = []

    # ─── プロパティ ───

    @property
    def size(self) -> int:
        return len(self._records)

    @property
    def behaviors_array(self) -> Optional[np.ndarray]:
        if not self._behaviors:
            return None
        return np.stack(self._behaviors, axis=0)  # (M, D)

    # ─── 新規性スコア計算 ───

    def compute_novelty(
        self,
        behavior: np.ndarray,
        extra_pool: Optional[np.ndarray] = None,
    ) -> float:
        """
        行動ベクトルの新規性スコアを計算する。

        アーカイブが空の場合は 1.0 (最大新規性) を返す。

        Parameters
        ----------
        behavior   : (D,) 行動ベクトル
        extra_pool : (B, D) 追加参照プール (現在世代の個体群など)
        """
        archive = self.behaviors_array

        if archive is None:
            return 1.0

        return novelty_score_with_buffer(
            candidate=behavior.astype(np.float32),
            archive_vecs=archive.astype(np.float32),
            buffer_vecs=extra_pool,
            k=self.k_neighbors,
        )

    def compute_novelty_batch(
        self,
        behaviors: np.ndarray,
    ) -> np.ndarray:
        """
        バッチの新規性スコアを一括計算する (JAX JIT 使用)。

        Parameters
        ----------
        behaviors : (N, D)

        Returns
        -------
        scores : (N,) float32
        """
        archive = self.behaviors_array
        if archive is None:
            return np.ones(len(behaviors), dtype=np.float32)

        k_eff = min(self.k_neighbors, len(archive))
        return np.array(knn_novelty_scores(
            jnp.array(behaviors, dtype=jnp.float32),
            jnp.array(archive,   dtype=jnp.float32),
            k=k_eff,
        ))

    # ─── アーカイブへの追加 ───

    def try_add(
        self,
        behavior: np.ndarray,
        skill_type: str,
        skill=None,
        task_fitness: float = 0.0,
        metadata: Optional[dict] = None,
        extra_pool: Optional[np.ndarray] = None,
    ) -> Tuple[bool, float]:
        """
        新規性スコアを計算し、条件を満たせばアーカイブに追加する。

        追加条件:
          (a) novelty_score >= novelty_threshold
          または
          (b) random() < add_prob

        アーカイブが満杯の場合は最古のエントリを削除 (FIFO)。

        Returns
        -------
        (added, novelty_score)
        """
        novelty = self.compute_novelty(behavior, extra_pool)

        add = (novelty >= self.novelty_threshold) or \
              (self._rng.random() < self.add_prob)

        if add:
            rec = NoveltyRecord(
                behavior=behavior.astype(np.float32),
                novelty_score=novelty,
                skill_type=skill_type,
                skill=skill,
                task_fitness=task_fitness,
                metadata=metadata or {},
            )
            if len(self._records) >= self.max_size:
                self._behaviors.pop(0)
                self._records.pop(0)
            self._behaviors.append(behavior.astype(np.float32))
            self._records.append(rec)

        return add, novelty

    # ─── 統計 ───

    @property
    def mean_novelty(self) -> float:
        if not self._records:
            return 0.0
        return float(np.mean([r.novelty_score for r in self._records]))

    @property
    def max_novelty(self) -> float:
        if not self._records:
            return 0.0
        return float(np.max([r.novelty_score for r in self._records]))

    def most_novel(self, n: int = 5) -> List[NoveltyRecord]:
        return sorted(self._records, key=lambda r: r.novelty_score, reverse=True)[:n]

    def summary(self) -> dict:
        return {
            "size":             self.size,
            "max_size":         self.max_size,
            "mean_novelty":     f"{self.mean_novelty:.4f}",
            "max_novelty":      f"{self.max_novelty:.4f}",
            "novelty_threshold": self.novelty_threshold,
            "k_neighbors":      self.k_neighbors,
        }


# ──────────────────────────────────────────────
# NoveltyFitness: NS+F 複合適合度
# ──────────────────────────────────────────────

class NoveltyFitness:
    """
    Novelty Search + Task Fitness の複合適合度計算器。

    combined = α * novelty_score + (1 - α) * task_fitness

    α=1.0: 純 Novelty Search (fitness-free)
    α=0.0: 純タスク適合度 (通常 NEAT)
    α=0.5: 均等複合

    Parameters
    ----------
    ns_archive : NoveltyArchive
    alpha      : 新規性スコアの重み [0, 1]
    """

    def __init__(self, ns_archive: NoveltyArchive, alpha: float = 0.5):
        self.ns_archive = ns_archive
        self.alpha = alpha

    def __call__(
        self,
        behavior: np.ndarray,
        task_fitness: float,
        extra_pool: Optional[np.ndarray] = None,
    ) -> float:
        """
        行動ベクトルと タスク適合度から複合スコアを返す。

        Parameters
        ----------
        behavior     : (D,) BCS 記述子
        task_fitness : NEAT 適合度 (負の MSE/BCE など)
        extra_pool   : 現世代の行動ベクトル群 (新規性計算の参照)

        Returns
        -------
        combined_fitness : float
        """
        novelty = self.ns_archive.compute_novelty(behavior, extra_pool)
        # task_fitness は通常 [-∞, 0] → [0, 1] に正規化
        norm_task = 1.0 / (1.0 + abs(task_fitness))
        return self.alpha * novelty + (1.0 - self.alpha) * norm_task


# ──────────────────────────────────────────────
# NoveltyEvolutionLoop: NS 統合進化ループ
# ──────────────────────────────────────────────

class NoveltyEvolutionLoop:
    """
    Novelty Search を組み込んだ TRIDENT 進化ループ。

    各イテレーション:
      1. skill_factory でスキルを進化 (task_fitness + BCS 取得)
      2. NoveltyFitness で複合スコアを計算
      3. NoveltyArchive に追加を試みる
      4. TRIDENTArchive に MAP-Elites として追加 (task_fitness ベース)

    Parameters
    ----------
    trident_archive  : TRIDENTArchive
    ns_archive       : NoveltyArchive
    skill_factory    : (skill_type, rng) -> (skill, task_fitness, bcs_descriptor)
    novelty_fitness  : NoveltyFitness  複合適合度計算器
    max_iterations   : 最大イテレーション数
    skill_types      : ローテーションするスキルタイプ
    """

    def __init__(
        self,
        trident_archive,
        ns_archive: NoveltyArchive,
        skill_factory: Callable,
        novelty_fitness: NoveltyFitness,
        max_iterations: int = 100,
        skill_types: Optional[List[str]] = None,
        seed: int = 0,
    ):
        self.trident_archive = trident_archive
        self.ns_archive      = ns_archive
        self.skill_factory   = skill_factory
        self.novelty_fitness = novelty_fitness
        self.max_iterations  = max_iterations
        self.skill_types     = skill_types or ["indexer", "gate", "slot_filler"]
        self._rng            = np.random.default_rng(seed)
        self.history: List[dict] = []

    def run(self) -> Tuple[object, NoveltyArchive]:
        """
        進化ループを実行する。

        Returns
        -------
        (trident_archive, ns_archive)
        """
        print(f"[NoveltyEvolutionLoop] {self.max_iterations} イテレーション開始")
        ns_added = 0
        map_adopted = 0

        for i in range(self.max_iterations):
            stype = self.skill_types[i % len(self.skill_types)]
            skill, task_fitness, bcs_desc = self.skill_factory(stype, self._rng)

            # 複合適合度
            current_behaviors = self.ns_archive.behaviors_array
            combined = self.novelty_fitness(
                behavior=bcs_desc,
                task_fitness=task_fitness,
                extra_pool=current_behaviors,
            )

            # Novelty Archive への追加
            added, novelty = self.ns_archive.try_add(
                behavior=bcs_desc,
                skill_type=stype,
                skill=skill,
                task_fitness=task_fitness,
                metadata={"iteration": i, "combined": combined},
            )
            if added:
                ns_added += 1

            # TRIDENTArchive (MAP-Elites) への追加 (task_fitness ベース)
            adopted = False
            if stype == "indexer":
                adopted = self.trident_archive.add_indexer(
                    skill, task_fitness, bcs_desc, {"iteration": i})
            elif stype == "gate":
                adopted = self.trident_archive.add_gate(
                    skill, task_fitness, bcs_desc, {"iteration": i})
            elif stype == "slot_filler":
                adopted = self.trident_archive.add_slot_filler(
                    skill, task_fitness, bcs_desc, {"iteration": i})
            if adopted:
                map_adopted += 1

            self.history.append({
                "iteration":    i,
                "skill_type":   stype,
                "task_fitness": task_fitness,
                "novelty":      novelty,
                "combined":     combined,
                "ns_added":     added,
                "map_adopted":  adopted,
            })

        total_map = self.trident_archive.total_skills
        print(
            f"[NoveltyEvolutionLoop] 完了:\n"
            f"  NS アーカイブ  : {ns_added}/{self.max_iterations} 追加 "
            f"(size={self.ns_archive.size})\n"
            f"  MAP-Elites     : {map_adopted}/{self.max_iterations} 採用 "
            f"(total={total_map})\n"
            f"  平均新規性     : {self.ns_archive.mean_novelty:.4f}"
        )
        return self.trident_archive, self.ns_archive

    def novelty_history(self) -> np.ndarray:
        """イテレーションごとの新規性スコア履歴 (N,)。"""
        return np.array([h["novelty"] for h in self.history])

    def combined_fitness_history(self) -> np.ndarray:
        """イテレーションごとの複合適合度履歴 (N,)。"""
        return np.array([h["combined"] for h in self.history])
