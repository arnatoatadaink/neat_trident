"""
tests/test_context_search.py
仮説E: AssociationFn + ContextSensitiveSearch のテスト
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.med_integration.context_search import (
    AssociationFn,
    ContextSensitiveSearch,
    SearchResult,
)

DIM = 16
N   = 30
RNG = np.random.default_rng(0)

# ──────────────────────────────────────────────
# フィクスチャ
# ──────────────────────────────────────────────

@pytest.fixture
def emb():
    e = RNG.standard_normal((N, DIM)).astype(np.float32)
    return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12)

@pytest.fixture
def texts():
    return [f"doc_{i}" for i in range(N)]

@pytest.fixture
def searcher(emb, texts):
    s = ContextSensitiveSearch()
    s.build_index(emb, texts)
    return s

@pytest.fixture
def assoc_fn():
    return AssociationFn()


# ──────────────────────────────────────────────
# AssociationFn テスト
# ──────────────────────────────────────────────

class TestAssociationFn:

    def test_default_weights_sum_one(self, assoc_fn):
        assert abs(assoc_fn.weights.sum() - 1.0) < 1e-9

    def test_score_returns_float(self, assoc_fn):
        q = RNG.standard_normal(DIM)
        c = RNG.standard_normal(DIM)
        ctx = RNG.standard_normal(DIM)
        s = assoc_fn.score(q, c, ctx)
        assert isinstance(s, float)
        assert np.isfinite(s)

    def test_score_no_context(self, assoc_fn):
        q = RNG.standard_normal(DIM)
        c = RNG.standard_normal(DIM)
        s = assoc_fn.score(q, c, context=None)
        assert np.isfinite(s)

    def test_identical_query_candidate_high_score(self, assoc_fn):
        """同一ベクトルは高スコアになる"""
        v = RNG.standard_normal(DIM)
        v = v / np.linalg.norm(v)
        ctx = RNG.standard_normal(DIM)
        s_same = assoc_fn.score(v, v.copy(), ctx)
        s_rand = assoc_fn.score(v, RNG.standard_normal(DIM), ctx)
        assert s_same > s_rand

    def test_score_batch_shape(self, assoc_fn):
        q = RNG.standard_normal(DIM)
        cands = RNG.standard_normal((10, DIM))
        ctx = RNG.standard_normal(DIM)
        scores = assoc_fn.score_batch(q, cands, ctx)
        assert scores.shape == (10,)
        assert np.all(np.isfinite(scores))

    def test_score_batch_no_context(self, assoc_fn):
        q = RNG.standard_normal(DIM)
        cands = RNG.standard_normal((5, DIM))
        scores = assoc_fn.score_batch(q, cands, context=None)
        assert scores.shape == (5,)

    def test_custom_weights(self):
        fn = AssociationFn(weights=[1.0, 0.0, 0.0, 0.0])
        # w0=1 のみ → score ≈ cosine(q, c) に等しい
        q = RNG.standard_normal(DIM)
        c = RNG.standard_normal(DIM)
        ctx = RNG.standard_normal(DIM)
        expected = AssociationFn._cosine(q, c)
        assert abs(fn.score(q, c, ctx) - expected) < 1e-9

    def test_fit_updates_weights(self, assoc_fn):
        w_before = assoc_fn.weights.copy()
        pairs = []
        for _ in range(10):
            pairs.append({
                "query":     RNG.standard_normal(DIM),
                "candidate": RNG.standard_normal(DIM),
                "context":   RNG.standard_normal(DIM),
                "label":     float(RNG.integers(0, 2)),
            })
        assoc_fn.fit(pairs, lr=0.1)
        assert not np.allclose(assoc_fn.weights, w_before)
        assert abs(assoc_fn.weights.sum() - 1.0) < 1e-9

    def test_fit_empty_no_error(self, assoc_fn):
        w_before = assoc_fn.weights.copy()
        assoc_fn.fit([])
        assert np.allclose(assoc_fn.weights, w_before)

    def test_fit_increments_generation(self, assoc_fn):
        pairs = [{"query": RNG.standard_normal(DIM),
                  "candidate": RNG.standard_normal(DIM),
                  "context": RNG.standard_normal(DIM),
                  "label": 1.0}]
        assoc_fn.fit(pairs)
        assert assoc_fn.arch_meta["generation"] == 1

    def test_to_dict_roundtrip(self, assoc_fn):
        d = assoc_fn.to_dict()
        fn2 = AssociationFn.from_dict(d)
        assert np.allclose(assoc_fn.weights, fn2.weights)

    def test_save_load(self, assoc_fn):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        try:
            assoc_fn.save(path)
            fn2 = AssociationFn.load(path)
            assert np.allclose(assoc_fn.weights, fn2.weights)
        finally:
            path.unlink(missing_ok=True)

    def test_saved_json_has_required_keys(self, assoc_fn):
        d = assoc_fn.to_dict()
        for key in ["arch_type", "weights", "fitness_history", "generation"]:
            assert key in d


# ──────────────────────────────────────────────
# ContextSensitiveSearch テスト
# ──────────────────────────────────────────────

class TestContextSensitiveSearch:

    def test_build_and_ntotal(self, searcher):
        assert searcher.ntotal == N

    def test_dimension(self, searcher):
        assert searcher.dimension == DIM

    def test_search_returns_list_of_searchresult(self, searcher, emb):
        q = emb[0]
        results = searcher.search(q, k=5)
        assert isinstance(results, list)
        assert len(results) == 5
        assert all(isinstance(r, SearchResult) for r in results)

    def test_search_with_context(self, searcher, emb):
        q   = emb[0]
        ctx = emb[1]
        results = searcher.search(q, context_emb=ctx, k=5)
        assert len(results) == 5

    def test_search_scores_finite(self, searcher, emb):
        q = emb[0]
        results = searcher.search(q, k=5)
        for r in results:
            assert np.isfinite(r.base_score)
            assert np.isfinite(r.assoc_score)
            assert np.isfinite(r.final_score)

    def test_search_sorted_descending(self, searcher, emb):
        q = emb[0]
        results = searcher.search(q, context_emb=emb[1], k=8)
        scores = [r.final_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_k_leq_n(self, searcher, emb):
        results = searcher.search(emb[0], k=N + 100)
        assert len(results) <= N

    def test_search_result_embedding_shape(self, searcher, emb):
        results = searcher.search(emb[0], k=3)
        for r in results:
            assert r.embedding.shape == (DIM,)

    def test_search_result_text_is_string(self, searcher, emb):
        results = searcher.search(emb[0], k=3)
        for r in results:
            assert isinstance(r.text, str)

    def test_context_changes_ranking(self, emb, texts):
        """文脈ありとなしで結果順位が変わることを確認"""
        s = ContextSensitiveSearch()
        s.build_index(emb, texts)
        q   = emb[0]
        ctx = emb[N // 2]

        res_no_ctx  = s.search(q, context_emb=None,  k=10, alpha=0.5)
        res_with_ctx = s.search(q, context_emb=ctx, k=10, alpha=0.5)

        indices_no  = [r.index for r in res_no_ctx]
        indices_ctx = [r.index for r in res_with_ctx]
        # 全く同じ順位にはならないはず
        assert indices_no != indices_ctx

    def test_alpha_zero_uses_only_assoc(self, searcher, emb):
        """alpha=0 のとき final_score = assoc_score"""
        results = searcher.search(emb[0], context_emb=emb[1], k=5, alpha=0.0)
        for r in results:
            assert abs(r.final_score - r.assoc_score) < 1e-9

    def test_alpha_one_uses_only_base(self, searcher, emb):
        """alpha=1 のとき final_score = base_score"""
        results = searcher.search(emb[0], k=5, alpha=1.0)
        for r in results:
            assert abs(r.final_score - r.base_score) < 1e-9

    def test_no_build_raises(self):
        s = ContextSensitiveSearch()
        with pytest.raises(RuntimeError):
            s.search(np.zeros(DIM), k=1)

    def test_numpy_fallback(self, emb, texts):
        """faiss なし (numpy fallback) でも動作する"""
        s = ContextSensitiveSearch()
        s.build_index(emb, texts)
        s._faiss_index = None   # 強制 fallback
        results = s.search(emb[0], k=5)
        assert len(results) == 5

    def test_custom_assoc_fn(self, emb, texts):
        fn = AssociationFn(weights=[1.0, 0.0, 0.0, 0.0])
        s  = ContextSensitiveSearch(association_fn=fn)
        s.build_index(emb, texts)
        results = s.search(emb[0], k=3)
        assert len(results) == 3
