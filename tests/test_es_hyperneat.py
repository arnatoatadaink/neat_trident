"""
pytest — ESHyperNEATProjector / ESHyperNEATIndexer
小規模パラメータで CPPN 進化まで通し動作確認する。
"""

import numpy as np
import pytest

PROJ_DIM = 4
INPUT_DIM = 8
HIDDEN_DIM = 4
N_CORPUS = 15
N_QUERY = 5
CPPN_POP = 8
GEN = 5
SEED = 0


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(SEED)


@pytest.fixture(scope="module")
def queries_proj(rng):
    q = rng.standard_normal((N_QUERY, PROJ_DIM)).astype(np.float32)
    return q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)


@pytest.fixture(scope="module")
def corpus(rng):
    c = rng.standard_normal((N_CORPUS, INPUT_DIM)).astype(np.float32)
    return c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-8)


@pytest.fixture(scope="module")
def fitted_projector(queries_proj):
    from src.es_hyperneat import ESHyperNEATProjector
    proj = ESHyperNEATProjector(
        proj_dim=PROJ_DIM,
        hidden_dim=HIDDEN_DIM,
        cppn_pop_size=CPPN_POP,
        cppn_species_size=2,
        generation_limit=GEN,
        seed=SEED,
    )
    proj.fit(queries_proj)
    return proj


@pytest.fixture(scope="module")
def fitted_indexer(corpus):
    from src.es_hyperneat import ESHyperNEATIndexer
    idx = ESHyperNEATIndexer(
        input_dim=INPUT_DIM,
        proj_dim=PROJ_DIM,
        hidden_dim=HIDDEN_DIM,
        cppn_pop_size=CPPN_POP,
        cppn_species_size=2,
        generation_limit=GEN,
        k=3,
        seed=SEED,
    )
    idx.fit(corpus)
    return idx


# ─── make_trident_substrate テスト ───

class TestTRIDENTSubstrate:

    def test_import(self):
        from src.es_hyperneat import make_trident_substrate
        assert make_trident_substrate is not None

    def test_substrate_creation(self):
        from src.es_hyperneat import make_trident_substrate
        substrate = make_trident_substrate(proj_dim=PROJ_DIM, hidden_dim=HIDDEN_DIM)
        assert substrate is not None

    def test_substrate_has_query_coors(self):
        from src.es_hyperneat import make_trident_substrate
        substrate = make_trident_substrate(proj_dim=PROJ_DIM, hidden_dim=HIDDEN_DIM)
        assert hasattr(substrate, "query_coors")
        assert substrate.query_coors.shape[1] == 4  # (x_src, y_src, x_tgt, y_tgt)


# ─── ESHyperNEATProjector テスト ───

class TestESHyperNEATProjector:

    def test_import(self):
        from src.es_hyperneat import ESHyperNEATProjector
        assert ESHyperNEATProjector is not None

    def test_is_fitted_before_fit(self):
        from src.es_hyperneat import ESHyperNEATProjector
        proj = ESHyperNEATProjector(proj_dim=PROJ_DIM, generation_limit=GEN)
        assert not proj.is_fitted

    def test_fit_sets_state(self, fitted_projector):
        assert fitted_projector.is_fitted
        assert fitted_projector._state is not None

    def test_project_single_shape(self, fitted_projector, queries_proj):
        out = fitted_projector.project(queries_proj[0])
        assert out.shape == (PROJ_DIM,)

    def test_project_batch_shape(self, fitted_projector, queries_proj):
        out = fitted_projector.project(queries_proj)
        assert out.shape == (N_QUERY, PROJ_DIM)

    def test_project_output_finite(self, fitted_projector, queries_proj):
        out = fitted_projector.project(queries_proj)
        assert np.isfinite(out).all()

    def test_projection_matrix_shape(self, fitted_projector):
        W = fitted_projector.projection_matrix()
        assert W.shape == (PROJ_DIM, PROJ_DIM)

    def test_projection_matrix_normalized(self, fitted_projector):
        W = fitted_projector.projection_matrix(normalize=True)
        col_norms = np.linalg.norm(W, axis=0)
        # 非ゼロ列のみ検証（少世代ではCPPNがゼロ出力を返す列が生じうる）
        nonzero = col_norms > 1e-4
        if nonzero.any():
            assert np.allclose(col_norms[nonzero], 1.0, atol=1e-4)

    def test_project_before_fit_raises(self, queries_proj):
        from src.es_hyperneat import ESHyperNEATProjector
        proj = ESHyperNEATProjector(proj_dim=PROJ_DIM, generation_limit=GEN)
        with pytest.raises(AssertionError):
            proj.project(queries_proj[0])


# ─── ESHyperNEATIndexer テスト ───

class TestESHyperNEATIndexer:

    def test_import(self):
        from src.es_hyperneat import ESHyperNEATIndexer
        assert ESHyperNEATIndexer is not None

    def test_fit_sets_corpus_proj(self, fitted_indexer):
        assert fitted_indexer._corpus_proj is not None
        assert fitted_indexer._corpus_proj.shape == (N_CORPUS, PROJ_DIM)

    def test_search_returns_k_results(self, fitted_indexer, corpus):
        k = 3
        indices, scores = fitted_indexer.search(corpus[0], k=k)
        assert indices.shape == (k,)
        assert scores.shape == (k,)

    def test_search_indices_in_range(self, fitted_indexer, corpus):
        indices, _ = fitted_indexer.search(corpus[0], k=3)
        assert all(0 <= i < N_CORPUS for i in indices)

    def test_search_scores_descending(self, fitted_indexer, corpus):
        _, scores = fitted_indexer.search(corpus[0], k=3)
        assert list(scores) == sorted(scores, reverse=True)

    def test_projection_matrix_shape(self, fitted_indexer):
        W = fitted_indexer.projection_matrix()
        assert W.shape == (PROJ_DIM, PROJ_DIM)

    def test_bcs_descriptor_shape(self, fitted_indexer, corpus):
        desc = fitted_indexer.bcs_descriptor(corpus[:5])
        assert desc.shape == (2,)

    def test_search_before_fit_raises(self, corpus):
        from src.es_hyperneat import ESHyperNEATIndexer
        idx = ESHyperNEATIndexer(input_dim=INPUT_DIM, proj_dim=PROJ_DIM,
                                  generation_limit=GEN)
        with pytest.raises(AssertionError):
            idx.search(corpus[0])
