"""
tests/test_neat_assoc_evolver.py
NEAT → AssociationFn 進化ループのテスト
"""

import pickle
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.med_integration.neat_assoc_evolver import (
    AssociationFnEvolver,
    AssociationFnProblem,
    NEATAssociationFn,
    compute_features,
)
from src.med_integration.context_search import ContextSensitiveSearch

DIM = 8
RNG = np.random.default_rng(0)


# ──────────────────────────────────────────────
# フィクスチャ
# ──────────────────────────────────────────────

@pytest.fixture(scope="module")
def small_pairs():
    """小規模フィードバックペア (pos=5, neg=5)"""
    pairs = []
    q = RNG.standard_normal(DIM).astype(np.float32)
    q = q / np.linalg.norm(q)
    ctx = RNG.standard_normal(DIM).astype(np.float32)
    ctx = ctx / np.linalg.norm(ctx)

    # 正例: q と同じ方向の候補
    for _ in range(5):
        c_pos = q + RNG.standard_normal(DIM).astype(np.float32) * 0.2
        pairs.append({"query": q, "candidate": c_pos, "context": ctx, "label": 1.0})

    # 負例: q と逆方向の候補
    for _ in range(5):
        c_neg = -q + RNG.standard_normal(DIM).astype(np.float32) * 0.2
        pairs.append({"query": q, "candidate": c_neg, "context": ctx, "label": 0.0})

    return pairs


@pytest.fixture(scope="module")
def evolved_fn(small_pairs):
    """pop=10, gen=3 で進化させた NEATAssociationFn (モジュールスコープ: 一度だけ実行)"""
    evolver = AssociationFnEvolver(
        pop_size=10,
        species_size=3,
        max_nodes=15,
        max_conns=20,
        seed=0,
    )
    return evolver.evolve(small_pairs, generation_limit=3, verbose=False)


# ──────────────────────────────────────────────
# compute_features テスト
# ──────────────────────────────────────────────

class TestComputeFeatures:

    def test_shape(self):
        q   = RNG.standard_normal(DIM)
        c   = RNG.standard_normal(DIM)
        ctx = RNG.standard_normal(DIM)
        feat = compute_features(q, c, ctx)
        assert feat.shape == (4,)
        assert feat.dtype == np.float32

    def test_no_context_zeros(self):
        """context=None のとき f[1], f[2], f[3] は 0"""
        q = RNG.standard_normal(DIM)
        c = RNG.standard_normal(DIM)
        feat = compute_features(q, c, None)
        assert feat[1] == 0.0
        assert feat[2] == 0.0
        assert feat[3] == 0.0

    def test_f0_is_cosine(self):
        """f[0] = cos(q, c)"""
        q = RNG.standard_normal(DIM).astype(np.float64)
        c = RNG.standard_normal(DIM).astype(np.float64)
        feat = compute_features(q, c, None)
        expected = float(np.dot(q, c) / (np.linalg.norm(q) * np.linalg.norm(c)))
        assert abs(feat[0] - expected) < 1e-6

    def test_finite(self):
        q   = RNG.standard_normal(DIM)
        c   = RNG.standard_normal(DIM)
        ctx = RNG.standard_normal(DIM)
        feat = compute_features(q, c, ctx)
        assert np.all(np.isfinite(feat))

    def test_zero_vector_safe(self):
        """ゼロベクトルでも例外なし"""
        q    = np.zeros(DIM, dtype=np.float32)
        c    = RNG.standard_normal(DIM)
        feat = compute_features(q, c, None)
        assert np.isfinite(feat[0])


# ──────────────────────────────────────────────
# AssociationFnProblem テスト
# ──────────────────────────────────────────────

class TestAssociationFnProblem:

    def test_input_output_shape(self):
        feats  = RNG.standard_normal((10, 4)).astype(np.float32)
        labels = RNG.integers(0, 2, size=10).astype(np.float32)
        prob   = AssociationFnProblem(feats, labels)
        assert prob.input_shape  == (4,)
        assert prob.output_shape == (1,)

    def test_wrong_feature_dim_raises(self):
        with pytest.raises(AssertionError):
            AssociationFnProblem(
                RNG.standard_normal((10, 3)).astype(np.float32),
                np.zeros(10, dtype=np.float32),
            )


# ──────────────────────────────────────────────
# NEATAssociationFn テスト
# ──────────────────────────────────────────────

class TestNEATAssociationFn:

    def test_score_returns_float(self, evolved_fn):
        q   = RNG.standard_normal(DIM)
        c   = RNG.standard_normal(DIM)
        ctx = RNG.standard_normal(DIM)
        s   = evolved_fn.score(q, c, ctx)
        assert isinstance(s, float)
        assert np.isfinite(s)

    def test_score_no_context(self, evolved_fn):
        q = RNG.standard_normal(DIM)
        c = RNG.standard_normal(DIM)
        s = evolved_fn.score(q, c, context=None)
        assert np.isfinite(s)

    def test_score_batch_shape(self, evolved_fn):
        q    = RNG.standard_normal(DIM)
        cands = RNG.standard_normal((8, DIM))
        ctx  = RNG.standard_normal(DIM)
        scores = evolved_fn.score_batch(q, cands, ctx)
        assert scores.shape == (8,)
        assert np.all(np.isfinite(scores))

    def test_score_batch_consistent(self, evolved_fn):
        q    = RNG.standard_normal(DIM)
        cands = RNG.standard_normal((5, DIM))
        ctx  = RNG.standard_normal(DIM)
        batch   = evolved_fn.score_batch(q, cands, ctx)
        singles = np.array([evolved_fn.score(q, c, ctx) for c in cands])
        np.testing.assert_allclose(batch, singles, rtol=1e-5)

    def test_to_dict_keys(self, evolved_fn):
        d = evolved_fn.to_dict()
        assert d["arch_type"] == "neat_cppn"
        assert "generation" in d

    def test_from_dict_raises(self):
        with pytest.raises(NotImplementedError):
            NEATAssociationFn.from_dict({"arch_type": "neat_cppn"})

    def test_save_load_pickle(self, evolved_fn):
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = Path(f.name)
        try:
            evolved_fn.save(path)
            loaded = NEATAssociationFn.load(path)
            q   = RNG.standard_normal(DIM)
            c   = RNG.standard_normal(DIM)
            ctx = RNG.standard_normal(DIM)
            assert abs(evolved_fn.score(q, c, ctx) - loaded.score(q, c, ctx)) < 1e-5
        finally:
            path.unlink(missing_ok=True)

    def test_swap_into_context_sensitive_search(self, evolved_fn):
        """進化済み fn を ContextSensitiveSearch に差し込める"""
        embs  = RNG.standard_normal((20, DIM)).astype(np.float32)
        embs  = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
        texts = [f"doc_{i}" for i in range(20)]

        searcher = ContextSensitiveSearch()
        searcher.build_index(embs, texts)
        searcher.swap_association_fn(evolved_fn)

        results = searcher.search(embs[0], context_emb=embs[1], k=5)
        assert len(results) == 5
        scores = [r.final_score for r in results]
        assert scores == sorted(scores, reverse=True)


# ──────────────────────────────────────────────
# AssociationFnEvolver テスト
# ──────────────────────────────────────────────

class TestAssociationFnEvolver:

    def test_evolve_returns_neat_fn(self, evolved_fn):
        assert isinstance(evolved_fn, NEATAssociationFn)

    def test_evolve_empty_pairs_raises(self):
        evolver = AssociationFnEvolver(pop_size=10, seed=0)
        with pytest.raises(ValueError):
            evolver.evolve([])

    def test_evolve_no_context(self):
        """context=None のペアでも進化できる"""
        pairs = []
        q = RNG.standard_normal(DIM).astype(np.float32)
        for _ in range(4):
            c_pos = q + RNG.standard_normal(DIM).astype(np.float32) * 0.1
            pairs.append({"query": q, "candidate": c_pos, "context": None, "label": 1.0})
            c_neg = -q + RNG.standard_normal(DIM).astype(np.float32) * 0.1
            pairs.append({"query": q, "candidate": c_neg, "context": None, "label": 0.0})

        evolver = AssociationFnEvolver(pop_size=10, species_size=3, max_nodes=15, max_conns=20, seed=1)
        fn = evolver.evolve(pairs, generation_limit=3, verbose=False)
        assert isinstance(fn, NEATAssociationFn)

    def test_build_dataset_shape(self, small_pairs):
        feats, labels = AssociationFnEvolver._build_dataset(small_pairs)
        assert feats.shape  == (len(small_pairs), 4)
        assert labels.shape == (len(small_pairs),)
        assert np.all(np.isfinite(feats))
