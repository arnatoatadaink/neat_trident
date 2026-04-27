"""
TRIDENT — DomainIndexAdapter
MED の DomainIndex を MEDIndexerProtocol に適合させるアダプタ。

DomainIndex は count / search / add を持つが、
MEDIndexerProtocol が要求する ntotal / dimension プロパティを持たない。
このアダプタが両者を埋める。

使用例:
    import sys; sys.path.insert(0, MED_ROOT)
    from src.memory.faiss_index import DomainIndex
    from src.common.config import FAISSIndexConfig

    cfg = FAISSIndexConfig(dim=384, initial_type="Flat", metric="inner_product")
    di  = DomainIndex(cfg)
    adapter = DomainIndexAdapter(di, dimension=384)

    # → MEDIndexerProtocol 準拠で TRIDENTMEDAdapter に渡せる
"""

from __future__ import annotations

import numpy as np

from .interfaces import MEDIndexerProtocol


class DomainIndexAdapter:
    """
    MED DomainIndex → MEDIndexerProtocol アダプタ。

    DomainIndex の API ギャップを埋める:
      count    → ntotal  (プロパティ名変換)
      なし      → dimension (コンストラクタで注入)

    Parameters
    ----------
    domain_index : MED DomainIndex インスタンス
    dimension    : ベクトル次元数 (DomainIndex 構築時の FAISSIndexConfig.dim と一致させること)
    """

    def __init__(self, domain_index: object, dimension: int) -> None:
        self._di = domain_index
        self._dimension = dimension

    # ─── MEDIndexerProtocol 実装 ───────────────

    def search(self, query: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
        """DomainIndex.search() に委譲する。"""
        return self._di.search(np.asarray(query, dtype=np.float32), k=k)

    def add(self, doc_ids: list[str], embeddings: np.ndarray) -> None:
        """DomainIndex.add() に委譲する。"""
        self._di.add(doc_ids, np.asarray(embeddings, dtype=np.float32))

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def ntotal(self) -> int:
        return self._di.count


def _check_protocol(adapter: DomainIndexAdapter) -> bool:
    """MEDIndexerProtocol 実装チェック (検証スクリプト用)。"""
    return isinstance(adapter, MEDIndexerProtocol)
