"""
pytest — NeatIndexer / HybridIndexer (A型 I/F)
小規模パラメータで進化ループまで通し動作確認する。
"""

import numpy as np
import pytest

# ─── フィクスチャ ───

DIM = 4
N_CORPUS = 20
N_QUERY = 5
POP = 10
GEN = 5
SEED = 0


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(SEED)


@pytest.fixture(scope="module")
def corpus(rng):
    c = rng.standard_normal((N_CORPUS, DIM)).astype(np.float32)
    return c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-8)


@pytest.fixture(scope="module")
def queries(rng, corpus):
    q = rng.standard_normal((N_QUERY, DIM)).astype(np.float32)
    return q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)


@pytest.fixture(scope="module")
def fitted_indexer(corpus, queries):
    from src.interfaces.neat_indexer import NeatIndexer
    ni = NeatIndexer(
        input_dim=DIM,
        pop_size=POP,
        species_size=3,
        max_nodes=30,
        max_conns=50,
        generation_limit=GEN,
        k=3,
        seed=SEED,
    )
    ni.fit(corpus, queries=queries)
    return ni


# ─── NeatIndexer テスト ───

class TestNeatIndexer:

    def test_import(self):
        from src.interfaces.neat_indexer import NeatIndexer
        assert NeatIndexer is not None

    def test_fit_returns_self(self, corpus, queries):
        from src.interfaces.neat_indexer import NeatIndexer
        ni = NeatIndexer(input_dim=DIM, pop_size=POP, generation_limit=GEN, seed=1)
        result = ni.fit(corpus, queries=queries)
        assert result is ni

    def test_fit_sets_state(self, fitted_indexer):
        assert fitted_indexer._state is not None
        assert fitted_indexer._best_params is not None
        assert fitted_indexer._corpus is not None

    def test_transform_single(self, fitted_indexer, queries):
        out = fitted_indexer.transform(queries[0])
        assert out.shape == (DIM,)
        assert np.isfinite(out).all()

    def test_transform_batch(self, fitted_indexer, queries):
        out = fitted_indexer.transform(queries)
        assert out.shape == (N_QUERY, DIM)
        assert np.isfinite(out).all()

    def test_search_returns_k_results(self, fitted_indexer, queries):
        k = 3
        indices, scores = fitted_indexer.search(queries[0], k=k)
        assert indices.shape == (k,)
        assert scores.shape == (k,)

    def test_search_indices_in_range(self, fitted_indexer, queries):
        indices, _ = fitted_indexer.search(queries[0], k=3)
        assert all(0 <= i < N_CORPUS for i in indices)

    def test_search_scores_descending(self, fitted_indexer, queries):
        _, scores = fitted_indexer.search(queries[0], k=3)
        assert list(scores) == sorted(scores, reverse=True)

    def test_bcs_descriptor_shape(self, fitted_indexer, queries, corpus):
        # true_neighbors なし
        desc = fitted_indexer.bcs_descriptor(queries)
        assert desc.shape == (2,)
        assert np.all((desc >= 0) & (desc <= 1)), f"BCS out of [0,1]: {desc}"

    def test_bcs_descriptor_with_neighbors(self, fitted_indexer, queries, corpus):
        # 正解近傍を与えるケース
        true_nn = np.tile(np.arange(3), (N_QUERY, 1))
        desc = fitted_indexer.bcs_descriptor(queries, true_neighbors=true_nn)
        assert desc.shape == (2,)

    def test_corpus_dim_mismatch_raises(self, queries):
        from src.interfaces.neat_indexer import NeatIndexer
        ni = NeatIndexer(input_dim=DIM, pop_size=POP, generation_limit=GEN)
        bad_corpus = np.random.randn(10, DIM + 1).astype(np.float32)
        with pytest.raises(AssertionError):
            ni.fit(bad_corpus, queries=queries)

    def test_transform_before_fit_raises(self, queries):
        from src.interfaces.neat_indexer import NeatIndexer
        ni = NeatIndexer(input_dim=DIM)
        with pytest.raises(AssertionError):
            ni.transform(queries[0])


# ─── HybridIndexer テスト ───

class TestHybridIndexer:

    def test_import(self):
        from src.interfaces.neat_indexer import HybridIndexer
        assert HybridIndexer is not None

    def test_mode_neat(self, fitted_indexer, queries):
        from src.interfaces.neat_indexer import HybridIndexer
        hi = HybridIndexer(neat_indexer=fitted_indexer, mode="neat")
        idx, scores = hi.search(queries[0], k=3)
        assert idx.shape == (3,)

    def test_mode_faiss(self, fitted_indexer, corpus, queries):
        import faiss
        from src.interfaces.neat_indexer import HybridIndexer
        index = faiss.IndexFlatIP(DIM)
        c_norm = corpus / (np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-8)
        index.add(c_norm)
        hi = HybridIndexer(neat_indexer=fitted_indexer, faiss_index=index, mode="faiss")
        idx, scores = hi.search(queries[0], k=3)
        assert idx.shape == (3,)

    def test_mode_hybrid(self, fitted_indexer, corpus, queries):
        import faiss
        from src.interfaces.neat_indexer import HybridIndexer
        index = faiss.IndexFlatIP(DIM)
        c_norm = corpus / (np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-8)
        index.add(c_norm)
        hi = HybridIndexer(neat_indexer=fitted_indexer, faiss_index=index, mode="hybrid")
        idx, scores = hi.search(queries[0], k=3)
        assert idx.shape == (3,)
        assert all(0 <= i < N_CORPUS for i in idx)

    def test_faiss_mode_without_index_raises(self, fitted_indexer):
        from src.interfaces.neat_indexer import HybridIndexer
        with pytest.raises(ValueError):
            HybridIndexer(neat_indexer=fitted_indexer, mode="faiss")
