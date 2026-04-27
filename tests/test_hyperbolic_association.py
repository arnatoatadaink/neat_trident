"""
tests/test_hyperbolic_association.py
仮説E Phase 3: HyperbolicAssociationFn のテスト
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

geoopt = pytest.importorskip("geoopt", reason="geoopt not installed")

from src.med_integration.hyperbolic_association import HyperbolicAssociationFn
from src.med_integration.context_search import AssociationFn, ContextSensitiveSearch

DIM = 16
N   = 20
RNG = np.random.default_rng(42)


@pytest.fixture
def fn():
    return HyperbolicAssociationFn()

@pytest.fixture
def emb():
    e = RNG.standard_normal((N, DIM)).astype(np.float32)
    return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12)


class TestHyperbolicAssociationFn:

    def test_score_returns_float(self, fn):
        q   = RNG.standard_normal(DIM)
        c   = RNG.standard_normal(DIM)
        ctx = RNG.standard_normal(DIM)
        s = fn.score(q, c, ctx)
        assert isinstance(s, float)
        assert np.isfinite(s)

    def test_score_no_context(self, fn):
        q = RNG.standard_normal(DIM)
        c = RNG.standard_normal(DIM)
        s = fn.score(q, c, context=None)
        assert np.isfinite(s)
        assert s > 0.0

    def test_score_positive(self, fn):
        """スコアは常に正 (距離ベースの逆数なので)"""
        for _ in range(10):
            q   = RNG.standard_normal(DIM)
            c   = RNG.standard_normal(DIM)
            ctx = RNG.standard_normal(DIM)
            assert fn.score(q, c, ctx) > 0.0

    def test_identical_vectors_highest_score(self, fn):
        """同一ベクトルはランダムより高スコア"""
        v   = RNG.standard_normal(DIM)
        v   = v / np.linalg.norm(v)
        ctx = RNG.standard_normal(DIM)
        s_same = fn.score(v, v.copy(), ctx)
        s_rand = fn.score(v, RNG.standard_normal(DIM), ctx)
        assert s_same > s_rand

    def test_context_increases_score(self, fn):
        """文脈あり (ctx_weight>0) はなしより高スコア"""
        q   = RNG.standard_normal(DIM)
        c   = RNG.standard_normal(DIM)
        ctx = RNG.standard_normal(DIM)
        s_no_ctx   = fn.score(q, c, context=None)
        s_with_ctx = fn.score(q, c, context=ctx)
        # ctx_weight=0.3 > 0 なので文脈ありの方が大きい
        assert s_with_ctx > s_no_ctx

    def test_score_batch_shape(self, fn):
        q    = RNG.standard_normal(DIM)
        cands = RNG.standard_normal((10, DIM))
        ctx  = RNG.standard_normal(DIM)
        scores = fn.score_batch(q, cands, ctx)
        assert scores.shape == (10,)
        assert np.all(np.isfinite(scores))

    def test_score_batch_no_context(self, fn):
        q    = RNG.standard_normal(DIM)
        cands = RNG.standard_normal((5, DIM))
        scores = fn.score_batch(q, cands, context=None)
        assert scores.shape == (5,)
        assert np.all(np.isfinite(scores))

    def test_score_batch_consistent_with_score(self, fn):
        """score_batch は score の繰り返しと一致する"""
        q    = RNG.standard_normal(DIM)
        cands = RNG.standard_normal((6, DIM))
        ctx  = RNG.standard_normal(DIM)
        batch = fn.score_batch(q, cands, ctx)
        singles = np.array([fn.score(q, c, ctx) for c in cands])
        np.testing.assert_allclose(batch, singles, rtol=1e-10)

    def test_float64_precision(self, fn):
        """float64 で動作し、精度が保たれる"""
        q = RNG.standard_normal(DIM).astype(np.float64)
        c = RNG.standard_normal(DIM).astype(np.float64)
        s = fn.score(q, c, context=None)
        # 結果が double 精度で有限
        assert isinstance(s, float)
        assert np.isfinite(s)

    def test_to_dict_has_required_keys(self, fn):
        d = fn.to_dict()
        for key in ["arch_type", "c", "ctx_weight", "scale", "generation"]:
            assert key in d
        assert d["arch_type"] == "hyperbolic"

    def test_to_dict_from_dict_roundtrip(self, fn):
        d   = fn.to_dict()
        fn2 = HyperbolicAssociationFn.from_dict(d)
        assert fn2.c          == fn.c
        assert fn2.ctx_weight == fn.ctx_weight
        assert fn2.scale      == fn.scale

    def test_save_load(self, fn):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        try:
            fn.save(path)
            fn2 = HyperbolicAssociationFn.load(path)
            assert fn2.c == fn.c
            assert fn2.ctx_weight == fn.ctx_weight
        finally:
            path.unlink(missing_ok=True)

    def test_custom_ctx_weight_zero(self):
        """ctx_weight=0 → score は文脈なしと同じ"""
        fn = HyperbolicAssociationFn(ctx_weight=0.0)
        q   = RNG.standard_normal(DIM)
        c   = RNG.standard_normal(DIM)
        ctx = RNG.standard_normal(DIM)
        assert abs(fn.score(q, c, ctx) - fn.score(q, c, None)) < 1e-10

    def test_swap_into_context_sensitive_search(self, emb):
        """swap_association_fn で ContextSensitiveSearch に差し込める"""
        texts = [f"doc_{i}" for i in range(N)]
        searcher = ContextSensitiveSearch()
        searcher.build_index(emb, texts)

        hyp_fn = HyperbolicAssociationFn()
        searcher.swap_association_fn(hyp_fn)

        results = searcher.search(emb[0], context_emb=emb[1], k=5)
        assert len(results) == 5
        scores = [r.final_score for r in results]
        assert scores == sorted(scores, reverse=True)
        assert all(np.isfinite(r.final_score) for r in results)


class TestHyperbolicVsMLP:
    """MLP版との比較テスト"""

    def test_both_rank_identical_higher(self):
        """MLP版・双曲版ともに同一ベクトルをランダムより高くランク付けする"""
        mlp = AssociationFn()
        hyp = HyperbolicAssociationFn()
        v   = RNG.standard_normal(DIM)
        v   = v / np.linalg.norm(v)
        ctx = RNG.standard_normal(DIM)
        rand = RNG.standard_normal(DIM)

        assert mlp.score(v, v.copy(), ctx) > mlp.score(v, rand, ctx)
        assert hyp.score(v, v.copy(), ctx) > hyp.score(v, rand, ctx)

    def test_scores_in_reasonable_range(self):
        """両実装でスコアが有限で異常値なし"""
        mlp = AssociationFn()
        hyp = HyperbolicAssociationFn()
        for _ in range(20):
            q   = RNG.standard_normal(DIM)
            c   = RNG.standard_normal(DIM)
            ctx = RNG.standard_normal(DIM)
            assert np.isfinite(mlp.score(q, c, ctx))
            assert np.isfinite(hyp.score(q, c, ctx))
