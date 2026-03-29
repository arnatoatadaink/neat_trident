"""
TRIDENT — MAP-Elites スキルアーカイブ
QDax MapElitesRepertoire を使って A/B/C 型スキルを BCS 空間で管理する。

各スキルタイプに独立したリポジトリを持つ:
  A型 NeatIndexer  : BCS [活性化強度, Recall@k]   格子 64×64 = 4096 セル
  B型 NeatGate     : BCS [発火率, 特異性]          格子 16×16 =  256 セル
  C型 NeatSlotFiller: BCS [充填率, 正確性]          格子 16×16 =  256 セル

アーカイブの役割:
  - 進化ループ内で発見された多様なスキルをセルごとに保持
  - fitness が高い個体のみ各セルに残す (MAP-Elites のエリート戦略)
  - セルからスキルを取り出して Novelty Search・推論に利用
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Any
import numpy as np
import jax.numpy as jnp
import jax

from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire


# ──────────────────────────────────────────────
# スキルタイプ定義
# ──────────────────────────────────────────────

SkillType = Literal["indexer", "gate", "slot_filler"]

_GRID_CONFIG: Dict[SkillType, int] = {
    "indexer":     64,   # 64×64 = 4096 セル
    "gate":        16,   # 16×16 = 256 セル
    "slot_filler": 16,   # 16×16 = 256 セル
}


# ──────────────────────────────────────────────
# グリッドセントロイド生成
# ──────────────────────────────────────────────

def make_grid_centroids(grid_size: int) -> jnp.ndarray:
    """
    [0, 1]^2 の均等格子セントロイドを生成する。

    Parameters
    ----------
    grid_size : 各軸のセル数 (総セル数 = grid_size^2)

    Returns
    -------
    centroids : jnp.ndarray  shape=(grid_size^2, 2)  dtype=float32
    """
    coords = np.linspace(0.0, 1.0, grid_size, dtype=np.float32)
    xx, yy = np.meshgrid(coords, coords)
    return jnp.array(
        np.stack([xx.ravel(), yy.ravel()], axis=1), dtype=jnp.float32
    )


# ──────────────────────────────────────────────
# SkillRecord: アーカイブに保存するスキルの記録
# ──────────────────────────────────────────────

@dataclass
class SkillRecord:
    """
    1 つのスキルインスタンスとその評価情報を保持するデータクラス。

    Attributes
    ----------
    skill_type  : "indexer" / "gate" / "slot_filler"
    skill       : 学習済みスキルオブジェクト (NeatIndexer 等)
    fitness     : 適合度スコア (高いほど良い)
    descriptor  : BCS 記述子 f32[2]  (セル割り当てに使用)
    cell_index  : 割り当てられたグリッドセルのインデックス
    metadata    : 追加情報 (学習設定など)
    """
    skill_type:  SkillType
    skill:       Any
    fitness:     float
    descriptor:  np.ndarray          # f32[2]
    cell_index:  int = -1
    metadata:    Dict[str, Any] = field(default_factory=dict)


# ──────────────────────────────────────────────
# SkillRepertoire: 単一スキルタイプのアーカイブ
# ──────────────────────────────────────────────

class SkillRepertoire:
    """
    1 つのスキルタイプ (A/B/C) の MAP-Elites アーカイブ。

    MapElitesRepertoire で fitness/descriptor を管理し、
    Python 辞書でスキルオブジェクト本体を保持する。

    Parameters
    ----------
    skill_type : "indexer" / "gate" / "slot_filler"
    grid_size  : 各軸のグリッドサイズ (デフォルトは _GRID_CONFIG から)
    """

    def __init__(
        self,
        skill_type: SkillType,
        grid_size: Optional[int] = None,
    ):
        self.skill_type = skill_type
        self.grid_size  = grid_size or _GRID_CONFIG[skill_type]
        self.num_cells  = self.grid_size ** 2

        # QDax リポジトリ (genotype = BCS 記述子 f32[2] を格納)
        centroids = make_grid_centroids(self.grid_size)
        dummy_genotype = jnp.zeros((2,), dtype=jnp.float32)
        self._repo: MapElitesRepertoire = MapElitesRepertoire.init_default(
            genotype=dummy_genotype,
            centroids=centroids,
        )

        # スキルオブジェクト本体: {cell_index: SkillRecord}
        self._skills: Dict[int, SkillRecord] = {}

    # ─── 統計プロパティ ───

    @property
    def filled_cells(self) -> int:
        return int(jnp.sum(self._repo.fitnesses > -jnp.inf))

    @property
    def coverage(self) -> float:
        return self.filled_cells / self.num_cells

    @property
    def best_fitness(self) -> Optional[float]:
        valid = self._repo.fitnesses[self._repo.fitnesses > -jnp.inf]
        if len(valid) == 0:
            return None
        return float(jnp.max(valid))

    @property
    def qd_score(self) -> float:
        """QD スコア = Σ max(fitness_per_cell, 0)"""
        clipped = jnp.maximum(self._repo.fitnesses, 0.0)
        return float(jnp.sum(clipped))

    # ─── スキルの追加 ───

    def add(self, record: SkillRecord) -> bool:
        """
        スキルをアーカイブに追加する。

        既存のセルより fitness が高い場合のみ更新される。

        Parameters
        ----------
        record : SkillRecord

        Returns
        -------
        bool : セルが更新された場合 True
        """
        desc = jnp.array(record.descriptor, dtype=jnp.float32)[None]  # (1, 2)
        fit  = jnp.array([[record.fitness]], dtype=jnp.float32)         # (1, 1)
        # genotype に descriptor 自体を格納 (セル座標として有用)
        gen  = jnp.array(record.descriptor, dtype=jnp.float32)[None]   # (1, 2)

        old_filled = self.filled_cells
        self._repo = self._repo.add(gen, desc, fit)
        new_filled = self.filled_cells

        # セルインデックスを解決
        cell_idx = self._resolve_cell(record.descriptor)
        record.cell_index = cell_idx

        current_fit = float(self._repo.fitnesses[cell_idx, 0])
        # fitness が採用されたか = 現在のセル fitness が record.fitness と一致
        adopted = abs(current_fit - record.fitness) < 1e-6

        if adopted:
            self._skills[cell_idx] = record

        return adopted

    def _resolve_cell(self, descriptor: np.ndarray) -> int:
        """descriptor が最も近いセントロイドのインデックスを返す。"""
        desc = np.asarray(descriptor, dtype=np.float32)
        centroids = np.array(self._repo.centroids)
        dists = np.sum((centroids - desc) ** 2, axis=1)
        return int(np.argmin(dists))

    # ─── スキルの取得 ───

    def get(self, descriptor: np.ndarray) -> Optional[SkillRecord]:
        """BCS 記述子に最も近いセルのスキルを返す。"""
        cell_idx = self._resolve_cell(descriptor)
        return self._skills.get(cell_idx)

    def get_by_cell(self, cell_index: int) -> Optional[SkillRecord]:
        return self._skills.get(cell_index)

    def all_skills(self) -> List[SkillRecord]:
        return list(self._skills.values())

    def best_skill(self) -> Optional[SkillRecord]:
        if not self._skills:
            return None
        return max(self._skills.values(), key=lambda r: r.fitness)

    # ─── 概要表示 ───

    def summary(self) -> Dict[str, Any]:
        return {
            "skill_type":   self.skill_type,
            "grid_size":    f"{self.grid_size}×{self.grid_size}",
            "num_cells":    self.num_cells,
            "filled_cells": self.filled_cells,
            "coverage":     f"{self.coverage:.1%}",
            "best_fitness": self.best_fitness,
            "qd_score":     f"{self.qd_score:.4f}",
        }


# ──────────────────────────────────────────────
# TRIDENTArchive: A/B/C 型を束ねる統合アーカイブ
# ──────────────────────────────────────────────

class TRIDENTArchive:
    """
    A/B/C 型スキルを一元管理する MAP-Elites スキルライブラリ。

    各スキルタイプに独立した SkillRepertoire を持つ。
    スキルの追加・取得・統計確認を統一 I/F で提供する。

    Parameters
    ----------
    grid_sizes : スキルタイプごとのグリッドサイズ上書き (省略時はデフォルト)
    """

    def __init__(
        self,
        grid_sizes: Optional[Dict[SkillType, int]] = None,
    ):
        gs = dict(_GRID_CONFIG)
        if grid_sizes:
            gs.update(grid_sizes)

        self.repertoires: Dict[SkillType, SkillRepertoire] = {
            "indexer":     SkillRepertoire("indexer",     gs["indexer"]),
            "gate":        SkillRepertoire("gate",        gs["gate"]),
            "slot_filler": SkillRepertoire("slot_filler", gs["slot_filler"]),
        }

    # ─── スキルの追加 ───

    def add_indexer(
        self,
        skill,
        fitness: float,
        descriptor: np.ndarray,
        metadata: Optional[Dict] = None,
    ) -> bool:
        """NeatIndexer をアーカイブに追加する。"""
        record = SkillRecord(
            skill_type="indexer",
            skill=skill,
            fitness=fitness,
            descriptor=np.asarray(descriptor, dtype=np.float32),
            metadata=metadata or {},
        )
        return self.repertoires["indexer"].add(record)

    def add_gate(
        self,
        skill,
        fitness: float,
        descriptor: np.ndarray,
        metadata: Optional[Dict] = None,
    ) -> bool:
        """NeatGate をアーカイブに追加する。"""
        record = SkillRecord(
            skill_type="gate",
            skill=skill,
            fitness=fitness,
            descriptor=np.asarray(descriptor, dtype=np.float32),
            metadata=metadata or {},
        )
        return self.repertoires["gate"].add(record)

    def add_slot_filler(
        self,
        skill,
        fitness: float,
        descriptor: np.ndarray,
        metadata: Optional[Dict] = None,
    ) -> bool:
        """NeatSlotFiller をアーカイブに追加する。"""
        record = SkillRecord(
            skill_type="slot_filler",
            skill=skill,
            fitness=fitness,
            descriptor=np.asarray(descriptor, dtype=np.float32),
            metadata=metadata or {},
        )
        return self.repertoires["slot_filler"].add(record)

    # ─── スキルの取得 ───

    def get_indexer(self, descriptor: np.ndarray) -> Optional[SkillRecord]:
        return self.repertoires["indexer"].get(descriptor)

    def get_gate(self, descriptor: np.ndarray) -> Optional[SkillRecord]:
        return self.repertoires["gate"].get(descriptor)

    def get_slot_filler(self, descriptor: np.ndarray) -> Optional[SkillRecord]:
        return self.repertoires["slot_filler"].get(descriptor)

    def best_indexer(self) -> Optional[SkillRecord]:
        return self.repertoires["indexer"].best_skill()

    def best_gate(self) -> Optional[SkillRecord]:
        return self.repertoires["gate"].best_skill()

    def best_slot_filler(self) -> Optional[SkillRecord]:
        return self.repertoires["slot_filler"].best_skill()

    # ─── 統計 ───

    @property
    def total_skills(self) -> int:
        return sum(r.filled_cells for r in self.repertoires.values())

    def summary(self) -> Dict[str, Any]:
        return {
            stype: repo.summary()
            for stype, repo in self.repertoires.items()
        }

    def print_summary(self) -> None:
        print("=" * 55)
        print("TRIDENT MAP-Elites アーカイブ統計")
        print("=" * 55)
        for stype, info in self.summary().items():
            label = {"indexer": "A型 NeatIndexer",
                     "gate":    "B型 NeatGate",
                     "slot_filler": "C型 NeatSlotFiller"}[stype]
            print(f"\n  [{label}]")
            for k, v in info.items():
                if k != "skill_type":
                    print(f"    {k:15s}: {v}")
        print()


# ──────────────────────────────────────────────
# EvolutionLoop: 複数世代にわたるアーカイブ更新ループ
# ──────────────────────────────────────────────

class EvolutionLoop:
    """
    NEAT 進化 × MAP-Elites アーカイブの更新ループ。

    1 イテレーション = 1 スキルの進化 + アーカイブへの追加。
    各イテレーションでランダムに A/B/C 型のいずれかを進化させ、
    結果を TRIDENTArchive に蓄積する。

    Parameters
    ----------
    archive          : TRIDENTArchive
    skill_factory    : スキルタイプに応じたスキルを生成する callable
                       signature: (skill_type, rng) -> (skill, fitness, descriptor)
    max_iterations   : 最大イテレーション数
    """

    def __init__(
        self,
        archive: TRIDENTArchive,
        skill_factory,
        max_iterations: int = 50,
        seed: int = 0,
    ):
        self.archive        = archive
        self.skill_factory  = skill_factory
        self.max_iterations = max_iterations
        self.rng            = np.random.default_rng(seed)
        self.history: List[Dict] = []

    def run(self, skill_types: Optional[List[SkillType]] = None) -> TRIDENTArchive:
        """
        進化ループを実行する。

        Parameters
        ----------
        skill_types : 進化させるスキルタイプのリスト
                      None の場合は ["indexer", "gate", "slot_filler"] を均等にサイクル

        Returns
        -------
        archive : 更新済み TRIDENTArchive
        """
        types = skill_types or ["indexer", "gate", "slot_filler"]

        print(f"[EvolutionLoop] {self.max_iterations} イテレーション開始")
        adopted_count = 0

        for i in range(self.max_iterations):
            stype = types[i % len(types)]
            skill, fitness, descriptor = self.skill_factory(stype, self.rng)

            if stype == "indexer":
                adopted = self.archive.add_indexer(skill, fitness, descriptor,
                                                   {"iteration": i})
            elif stype == "gate":
                adopted = self.archive.add_gate(skill, fitness, descriptor,
                                                {"iteration": i})
            else:
                adopted = self.archive.add_slot_filler(skill, fitness, descriptor,
                                                       {"iteration": i})

            if adopted:
                adopted_count += 1

            self.history.append({
                "iteration": i,
                "skill_type": stype,
                "fitness": fitness,
                "descriptor": descriptor.tolist(),
                "adopted": adopted,
            })

        total = self.archive.total_skills
        print(f"[EvolutionLoop] 完了: {adopted_count}/{self.max_iterations} 採用, "
              f"総スキル数={total}")
        return self.archive
