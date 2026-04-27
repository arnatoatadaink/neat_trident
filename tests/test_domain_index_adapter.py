"""
tests/test_domain_index_adapter.py
DomainIndexAdapter のユニットテスト。

MED / pydantic 不要: _FakeDomainIndex で DomainIndex の振る舞いを再現。
"""

import numpy as np
import pytest

from src.med_integration.domain_index_adapter import DomainIndexAdapter, _check_protocol
from src.med_integration.interfaces import MEDIndexerProtocol
from src.med_integration.trident_adapter import TRIDENTMEDAdapter, HybridIndexerMEDAdapter
from src.med_integration.stub_med import StubMEDSkillStore

DIM = 16
RNG = np.random.default_rng(7)


class _FakeDomainIndex:
    """DomainIndex の最小互換フェイク (MED 依存なし)。"""

    def __init__(self, dim: int = DIM) -> None:
        self._dim = dim
        self._ids: list[str] = []
        self._vecs: list[np.ndarray] = []

    @property
    def count(self) -> int:
        return len(self._ids)

    def add(self, doc_ids: list[str], embeddings: np.ndarray) -> None:
        for did, emb in zip(doc_ids, embeddings):
            if did not in self._ids:
                self._ids.append(did)
                self._vecs.append(emb.astype(np.float32))

    def search(self, query: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
        if not self._ids:
            return []
        mat = np.stack(self._vecs)
        q = query.astype(np.float32)
        scores = mat @ q / (np.linalg.norm(mat, axis=1) + 1e-8)
        n = min(k, len(self._ids))
        order = np.argsort(-scores)[:n]
        return [(self._ids[i], float(scores[i])) for i in order]


@pytest.fixture
def adapter() -> DomainIndexAdapter:
    return DomainIndexAdapter(_FakeDomainIndex(), dimension=DIM)


class TestDomainIndexAdapterBasic:

    def test_protocol_compliance(self, adapter):
        assert _check_protocol(adapter)
        assert isinstance(adapter, MEDIndexerProtocol)

    def test_dimension_property(self, adapter):
        assert adapter.dimension == DIM

    def test_ntotal_empty(self, adapter):
        assert adapter.ntotal == 0

    def test_ntotal_after_add(self, adapter):
        embs = RNG.standard_normal((5, DIM)).astype(np.float32)
        adapter.add([f"d{i}" for i in range(5)], embs)
        assert adapter.ntotal == 5

    def test_add_accepts_float64(self, adapter):
        embs = RNG.standard_normal((3, DIM))   # float64
        adapter.add(["a", "b", "c"], embs)
        assert adapter.ntotal == 3

    def test_search_returns_list_of_tuples(self, adapter):
        embs = RNG.standard_normal((10, DIM)).astype(np.float32)
        adapter.add([f"d{i}" for i in range(10)], embs)
        q = RNG.standard_normal(DIM).astype(np.float32)
        results = adapter.search(q, k=5)
        assert isinstance(results, list)
        assert len(results) == 5
        assert all(isinstance(r[0], str) for r in results)
        assert all(isinstance(r[1], float) for r in results)

    def test_search_sorted_descending(self, adapter):
        embs = RNG.standard_normal((10, DIM)).astype(np.float32)
        adapter.add([f"d{i}" for i in range(10)], embs)
        q = RNG.standard_normal(DIM).astype(np.float32)
        results = adapter.search(q, k=5)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_empty_index(self, adapter):
        q = RNG.standard_normal(DIM).astype(np.float32)
        assert adapter.search(q, k=3) == []

    def test_search_k_clipped_to_ntotal(self, adapter):
        embs = RNG.standard_normal((3, DIM)).astype(np.float32)
        adapter.add(["a", "b", "c"], embs)
        q = RNG.standard_normal(DIM).astype(np.float32)
        results = adapter.search(q, k=10)
        assert len(results) == 3

    def test_highest_score_is_self(self):
        fake = _FakeDomainIndex()
        embs = RNG.standard_normal((5, DIM)).astype(np.float32)
        ids = [f"d{i}" for i in range(5)]
        fake.add(ids, embs)
        adapter = DomainIndexAdapter(fake, dimension=DIM)
        # embs[0] に完全一致クエリを送ると d0 が最上位のはず
        q = embs[0] / (np.linalg.norm(embs[0]) + 1e-8)
        results = adapter.search(q, k=5)
        assert results[0][0] == "d0"


class TestDomainIndexAdapterInTRIDENTAdapter:

    def test_used_as_med_indexer(self):
        """DomainIndexAdapter を TRIDENTMEDAdapter の med_indexer として使える。"""
        from src.interfaces.neat_indexer import NeatIndexer, HybridIndexer

        ni = NeatIndexer(
            input_dim=DIM, pop_size=10, species_size=3,
            max_nodes=30, max_conns=50, generation_limit=3, seed=0,
        )
        corpus = RNG.standard_normal((20, DIM)).astype(np.float32)
        ni.fit(corpus)
        hi = HybridIndexer(neat_indexer=ni, mode="neat")

        fake = _FakeDomainIndex()
        di_adapter = DomainIndexAdapter(fake, dimension=DIM)
        store = StubMEDSkillStore()

        full = TRIDENTMEDAdapter(
            hybrid_indexer=hi,
            med_indexer=di_adapter,
            med_skill_store=store,
            dimension=DIM,
        )

        doc_ids = [f"doc_{i}" for i in range(20)]
        full.sync_indexer(doc_ids=doc_ids, corpus=corpus)

        # MED (DomainIndexAdapter 経由) の検索結果確認
        q = RNG.standard_normal(DIM).astype(np.float32)
        med_res = full.search_med(q, k=5)
        assert len(med_res) == 5
        assert isinstance(med_res[0][0], str)

        # ntotal が DomainIndexAdapter 経由で正しく反映されている
        assert di_adapter.ntotal == 20
        assert full.summary()["med_ntotal"] == 20
