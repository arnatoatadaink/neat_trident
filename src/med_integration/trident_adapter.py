"""
TRIDENT — MED アダプタ層
TRIDENT の HybridIndexer を MED の DomainIndex I/F に適合させる。

MED準拠シグネチャ (MED/src/memory/faiss_index.py DomainIndex):
  search(query: (dim,), k=5) → list[tuple[doc_id: str, score: float]]
  add(doc_ids: list[str], embeddings: (n, dim)) → None

使用例:
    from src.med_integration.trident_adapter import TRIDENTMEDAdapter
    from src.med_integration.stub_med import StubMEDIndexer, StubMEDSkillStore

    adapter = TRIDENTMEDAdapter(
        hybrid_indexer=hi,
        med_indexer=StubMEDIndexer(dimension=384),
        med_skill_store=StubMEDSkillStore(),
    )
    adapter.sync_indexer(doc_ids=["d0", "d1", ...], corpus=corpus)
    results = adapter.search(query, k=5)  # → list[tuple[str, float]]
"""

from __future__ import annotations

from typing import Optional
import numpy as np

from .interfaces import MEDIndexerProtocol, MEDSkillStoreProtocol


class HybridIndexerMEDAdapter:
    """
    HybridIndexer を MEDIndexerProtocol (MED DomainIndex準拠) に適合させるアダプタ。

    変換内容:
      HybridIndexer.search(q, k) → (indices: ndarray, scores: ndarray)
      ↓ このアダプタが変換
      MED準拠: search(q, k) → list[tuple[doc_id: str, score: float]]

      MED準拠: add(doc_ids, embeddings) → None
      ↓ doc_ids を内部マップに保存し corpus を sync

    Parameters
    ----------
    hybrid_indexer : 学習済み HybridIndexer
    dimension      : ベクトル次元数
    """

    def __init__(self, hybrid_indexer, dimension: int):
        self._hi = hybrid_indexer
        self._dimension = dimension
        self._doc_ids: list[str] = []      # corpus インデックス → doc_id マップ

    # ─── MEDIndexerProtocol 実装 ───

    def search(self, query: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
        """
        HybridIndexer で k 近傍検索し、MED準拠の戻値形式で返す。

        Parameters
        ----------
        query : (D,) L2正規化済みクエリベクトル

        Returns
        -------
        list[tuple[doc_id: str, score: float]]  降順ソート済み
        """
        q = np.asarray(query, dtype=np.float32).ravel()
        indices, scores = self._hi.search(q, k=k)

        results = []
        for idx, score in zip(indices.tolist(), scores.tolist()):
            doc_id = self._doc_ids[idx] if idx < len(self._doc_ids) else str(idx)
            results.append((doc_id, float(score)))
        return results

    def add(self, doc_ids: list[str], embeddings: np.ndarray) -> None:
        """
        doc_ids を内部マップに登録する。

        MED準拠シグネチャ。HybridIndexer は fit() でコーパスを受け取るため
        実際の FAISS/NEAT へのベクトル追加は sync_corpus() で行う。
        ここでは doc_id マッピングのみ更新する。

        Parameters
        ----------
        doc_ids    : ドキュメントID リスト (長さ N)
        embeddings : (N, D) ベクトル群 (シグネチャ互換のため受け取るが内部保存は省略)
        """
        assert len(doc_ids) == len(embeddings), (
            f"doc_ids length {len(doc_ids)} != embeddings rows {len(embeddings)}"
        )
        self._doc_ids.extend(doc_ids)

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def ntotal(self) -> int:
        return len(self._doc_ids)

    # ─── TRIDENT 拡張メソッド ───

    def sync_corpus(self, doc_ids: list[str], corpus: np.ndarray) -> None:
        """
        コーパスと doc_ids を一括登録する (fit() 後に呼ぶ)。

        Parameters
        ----------
        doc_ids : ドキュメントID リスト (長さ = corpus 行数)
        corpus  : (N, D) コーパスベクトル
        """
        assert len(doc_ids) == len(corpus), (
            f"doc_ids length {len(doc_ids)} != corpus rows {len(corpus)}"
        )
        self._doc_ids = list(doc_ids)

    def get_doc_id(self, index: int) -> str:
        """コーパスインデックスから doc_id を返す。"""
        if index < len(self._doc_ids):
            return self._doc_ids[index]
        return str(index)

    def get_hybrid_indexer(self):
        """内部の HybridIndexer を返す (TRIDENT 内部用)。"""
        return self._hi


class TRIDENTMEDAdapter:
    """
    TRIDENT ↔ MED の統合アダプタ。

    - HybridIndexer を MEDIndexerProtocol (DomainIndex準拠) として MED に提供
    - A/B/C 型スキルを MEDSkillStoreProtocol に書き込む
    - TRIDENTArchive のベストスキルを一括エクスポート

    Parameters
    ----------
    hybrid_indexer  : 学習済み HybridIndexer
    med_indexer     : MEDIndexerProtocol 実装 (StubMEDIndexer or MED DomainIndex)
    med_skill_store : MEDSkillStoreProtocol 実装
    dimension       : ベクトル次元数 (デフォルト 384)
    """

    def __init__(
        self,
        hybrid_indexer,
        med_indexer: MEDIndexerProtocol,
        med_skill_store: MEDSkillStoreProtocol,
        dimension: int = 384,
    ):
        self._hi = hybrid_indexer
        self._med_indexer = med_indexer
        self._med_store = med_skill_store
        self._dimension = dimension
        self._hi_adapter = HybridIndexerMEDAdapter(hybrid_indexer, dimension)

    # ─── インデクサ操作 ───

    def sync_indexer(self, doc_ids: list[str], corpus: np.ndarray) -> None:
        """
        コーパスを MED インデクサと TRIDENT の両方に登録する。

        Parameters
        ----------
        doc_ids : ドキュメントID リスト (長さ = corpus 行数)
        corpus  : (N, D) L2正規化済みコーパスベクトル
        """
        # MED インデクサ (DomainIndex等) に追加
        self._med_indexer.add(doc_ids, corpus)
        # HybridIndexer アダプタの doc_id マップ同期
        self._hi_adapter.sync_corpus(doc_ids, corpus)

    def search(self, query: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
        """
        HybridIndexer (TRIDENT) で検索する。

        Returns
        -------
        list[tuple[doc_id: str, score: float]]  降順ソート済み (MED準拠)
        """
        return self._hi_adapter.search(query, k=k)

    def search_med(self, query: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
        """
        MED インデクサ (DomainIndex等) で検索する (比較・フォールバック用)。

        Returns
        -------
        list[tuple[doc_id: str, score: float]]  (MED準拠)
        """
        return self._med_indexer.search(query, k=k)

    # ─── スキルストア操作 ───

    def export_skill(
        self,
        skill_obj,
        skill_type: str,
        fitness: float,
        descriptor: np.ndarray,
        metadata: Optional[dict] = None,
    ) -> int:
        """
        スキルを MED スキルストアに書き込む。

        Returns
        -------
        skill_id : int
        """
        skill_dict = {
            "type":       skill_type,
            "fitness":    float(fitness),
            "descriptor": np.asarray(descriptor, dtype=np.float32).tolist(),
            "skill_obj":  skill_obj,
            "metadata":   metadata or {},
        }
        return self._med_store.store_skill(skill_dict)

    def export_archive(self, trident_archive) -> dict[str, list[int]]:
        """
        TRIDENTArchive の全スキルを MED スキルストアに一括エクスポートする。

        Returns
        -------
        exported_ids : {"indexer": [...], "gate": [...], "slot_filler": [...]}
        """
        exported: dict[str, list[int]] = {
            "indexer": [], "gate": [], "slot_filler": []
        }

        for stype, repo in trident_archive.repertoires.items():
            for record in repo.all_skills():
                sid = self.export_skill(
                    skill_obj=record.skill,
                    skill_type=stype,
                    fitness=record.fitness,
                    descriptor=record.descriptor,
                    metadata=record.metadata,
                )
                exported[stype].append(sid)

        total = sum(len(ids) for ids in exported.values())
        print(
            f"[TRIDENTMEDAdapter] エクスポート完了: {total} スキル → MEDSkillStore\n"
            + "\n".join(f"  {t}: {len(ids)} 件" for t, ids in exported.items())
        )
        return exported

    # ─── プロパティ ───

    @property
    def indexer_adapter(self) -> HybridIndexerMEDAdapter:
        """MEDIndexerProtocol として使える HybridIndexer アダプタ。"""
        return self._hi_adapter

    @property
    def med_indexer(self) -> MEDIndexerProtocol:
        return self._med_indexer

    @property
    def med_skill_store(self) -> MEDSkillStoreProtocol:
        return self._med_store

    def summary(self) -> dict:
        return {
            "dimension":        self._dimension,
            "trident_ntotal":   self._hi_adapter.ntotal,
            "med_ntotal":       self._med_indexer.ntotal,
            "skill_store_size": self._med_store.size,
        }
