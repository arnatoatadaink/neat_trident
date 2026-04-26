"""
TRIDENT — MED統合 抽象インターフェース定義
MED 側に期待するコンポーネントを typing.Protocol で宣言する。

MED実装準拠 (MED/src/memory/faiss_index.py DomainIndex):
  search(query: (dim,), k) → list[tuple[doc_id: str, score: float]]
  add(doc_ids: list[str], embeddings: (n, dim)) → None
  ベクトルは L2 正規化済み前提 (inner_product = cosine similarity)

実装差し替えフロー:
  開発中  : stub_med.py の StubMEDIndexer / StubMEDSkillStore を使用
  統合時  : MED の DomainIndex をそのまま渡す (Protocol 互換)
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable
import numpy as np


@runtime_checkable
class MEDIndexerProtocol(Protocol):
    """
    MED の DomainIndex に準拠した FAISS インデクサ I/F。

    MED 実装: MED/src/memory/faiss_index.py — DomainIndex
    """

    def search(self, query: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
        """
        クエリに対して k 近傍を返す。

        MED 準拠: DomainIndex.search(query, k) → list[tuple[doc_id, score]]

        Parameters
        ----------
        query : (D,) L2正規化済みクエリベクトル (float32)
        k     : 近傍数 (デフォルト 5)

        Returns
        -------
        list[tuple[doc_id: str, score: float]]  降順ソート済み
        """
        ...

    def add(self, doc_ids: list[str], embeddings: np.ndarray) -> None:
        """
        ベクトル群をインデックスに追加する。

        MED 準拠: DomainIndex.add(doc_ids, embeddings)

        Parameters
        ----------
        doc_ids    : ドキュメントID のリスト (長さ N)
        embeddings : (N, D) L2正規化済みベクトル群 (float32)
        """
        ...

    @property
    def dimension(self) -> int:
        """インデックスのベクトル次元数。"""
        ...

    @property
    def ntotal(self) -> int:
        """インデックスに登録済みのベクトル数。"""
        ...


@runtime_checkable
class MEDSkillStoreProtocol(Protocol):
    """
    MED のスキルストアに期待するインターフェース。

    A/B/C 型スキルを dict 形式で保存・取得する。
    """

    def store_skill(self, skill: dict) -> int:
        """
        スキルを保存してスキルIDを返す。

        Parameters
        ----------
        skill : スキル情報の辞書
            必須キー: "type" ("indexer"/"gate"/"slot_filler"), "fitness", "descriptor"

        Returns
        -------
        skill_id : int
        """
        ...

    def get_skill(self, skill_id: int) -> dict:
        """
        スキルIDからスキル情報を取得する。

        Parameters
        ----------
        skill_id : store_skill() が返したID

        Returns
        -------
        skill dict (存在しない場合は空 dict)
        """
        ...

    def list_skills(self, skill_type: str | None = None) -> list[dict]:
        """
        保存済みスキルの一覧を返す。

        Parameters
        ----------
        skill_type : "indexer"/"gate"/"slot_filler" でフィルタ (None は全件)
        """
        ...

    @property
    def size(self) -> int:
        """保存済みスキル数。"""
        ...
