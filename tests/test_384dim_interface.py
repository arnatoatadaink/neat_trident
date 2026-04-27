"""
tests/test_384dim_interface.py
384次元 embedding での ContextSensitiveSearch + HyperbolicAssociationFn 動作確認
(ランダム numpy ベクトル使用 — モデル不要・高速)
"""

import numpy as np
import pytest

from src.med_integration.context_search import AssociationFn, ContextSensitiveSearch, SearchResult
from src.med_integration.hyperbolic_association import HyperbolicAssociationFn

DIM = 384
N   = 50
RNG = np.random.default_rng(7)


@pytest.fixture(scope="module")
def emb_384():
    e = RNG.standard_normal((N, DIM)).astype(np.float32)
    return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12)

@pytest.fixture(scope="module")
def texts_50():
    return [f"doc_{i}" for i in range(N)]

@pytest.fixture(scope="module")
def searcher_384(emb_384, texts_50):
    s = ContextSensitiveSearch()
    s.build_index(emb_384, texts_50)
    return s


class TestContextSearch384Dim:

    def test_dimension_is_384(self, searcher_384):
        assert searcher_384.dimension == DIM

    def test_ntotal(self, searcher_384):
        assert searcher_384.ntotal == N

    def test_search_returns_k_results(self, searcher_384, emb_384):
        results = searcher_384.search(emb_384[0], k=10)
        assert len(results) == 10

    def test_search_with_context_384(self, searcher_384, emb_384):
        q   = emb_384[0]
        ctx = emb_384[1]
        results = searcher_384.search(q, context_emb=ctx, k=5, alpha=0.5)
        assert len(results) == 5
        assert all(isinstance(r, SearchResult) for r in results)

    def test_scores_finite(self, searcher_384, emb_384):
        q   = emb_384[0]
        ctx = emb_384[2]
        results = searcher_384.search(q, context_emb=ctx, k=10)
        for r in results:
            assert np.isfinite(r.base_score)
            assert np.isfinite(r.assoc_score)
            assert np.isfinite(r.final_score)

    def test_sorted_descending(self, searcher_384, emb_384):
        results = searcher_384.search(emb_384[0], context_emb=emb_384[3], k=10)
        scores = [r.final_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_context_changes_ranking_384(self, emb_384, texts_50):
        """384次元でも文脈ありとなしで結果が変わる"""
        s = ContextSensitiveSearch()
        s.build_index(emb_384, texts_50)
        q   = emb_384[0]
        ctx = emb_384[N // 2]

        res_none = s.search(q, context_emb=None,  k=15, alpha=0.5)
        res_ctx  = s.search(q, context_emb=ctx,   k=15, alpha=0.5)

        ids_none = [r.index for r in res_none]
        ids_ctx  = [r.index for r in res_ctx]
        assert ids_none != ids_ctx

    def test_assoc_score_batch_384(self):
        """384次元での score_batch が正しく動作する"""
        fn    = AssociationFn()
        q     = RNG.standard_normal(DIM)
        cands = RNG.standard_normal((20, DIM))
        ctx   = RNG.standard_normal(DIM)
        scores = fn.score_batch(q, cands, ctx)
        assert scores.shape == (20,)
        assert np.all(np.isfinite(scores))


class TestHyperbolic384Dim:

    def test_hyp_score_384(self):
        fn  = HyperbolicAssociationFn()
        q   = RNG.standard_normal(DIM)
        c   = RNG.standard_normal(DIM)
        ctx = RNG.standard_normal(DIM)
        s   = fn.score(q, c, ctx)
        assert isinstance(s, float)
        assert np.isfinite(s)

    def test_hyp_batch_384(self):
        fn    = HyperbolicAssociationFn()
        q     = RNG.standard_normal(DIM)
        cands = RNG.standard_normal((10, DIM))
        ctx   = RNG.standard_normal(DIM)
        scores = fn.score_batch(q, cands, ctx)
        assert scores.shape == (10,)
        assert np.all(np.isfinite(scores))

    def test_hyp_swap_384(self, emb_384, texts_50):
        """HyperbolicAssociationFn を 384次元 searcher に差し込める"""
        s = ContextSensitiveSearch()
        s.build_index(emb_384, texts_50)
        s.swap_association_fn(HyperbolicAssociationFn())
        results = s.search(emb_384[0], context_emb=emb_384[1], k=5)
        assert len(results) == 5
        assert all(np.isfinite(r.final_score) for r in results)
