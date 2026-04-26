"""
pytest — NeatSlotFiller / NeatKGWriter (C型 I/F)
"""

import numpy as np
import pytest

CONTEXT_DIM = 4
N = 20
POP = 10
GEN = 5
SEED = 0
KG_SCHEMA = ("node", "relation", "weight")
CUSTOM_SCHEMA = ("reward", "reason", "target")


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(SEED)


@pytest.fixture(scope="module")
def contexts(rng):
    return rng.standard_normal((N, CONTEXT_DIM)).astype(np.float32)


@pytest.fixture(scope="module")
def targets(rng):
    # [-1, 1] に収まるターゲットスロット値
    return np.tanh(rng.standard_normal((N, len(KG_SCHEMA)))).astype(np.float32)


@pytest.fixture(scope="module")
def fitted_filler(contexts, targets):
    from src.interfaces.neat_slot_filler import NeatSlotFiller
    sf = NeatSlotFiller(
        slot_names=KG_SCHEMA,
        context_dim=CONTEXT_DIM,
        pop_size=POP,
        species_size=3,
        max_nodes=30,
        max_conns=50,
        generation_limit=GEN,
        seed=SEED,
    )
    sf.fit(contexts, targets)
    return sf


# ─── NeatSlotFiller テスト ───

class TestNeatSlotFiller:

    def test_import(self):
        from src.interfaces.neat_slot_filler import NeatSlotFiller
        assert NeatSlotFiller is not None

    def test_fit_returns_self(self, contexts, targets):
        from src.interfaces.neat_slot_filler import NeatSlotFiller
        sf = NeatSlotFiller(slot_names=KG_SCHEMA, context_dim=CONTEXT_DIM,
                             pop_size=POP, generation_limit=GEN, seed=1)
        assert sf.fit(contexts, targets) is sf

    def test_fit_sets_state(self, fitted_filler):
        assert fitted_filler._state is not None
        assert fitted_filler._best_params is not None

    def test_fill_returns_dict(self, fitted_filler, contexts):
        result = fitted_filler.fill(contexts[0])
        assert isinstance(result, dict)
        assert set(result.keys()) == set(KG_SCHEMA)

    def test_fill_values_in_range(self, fitted_filler, contexts):
        result = fitted_filler.fill(contexts[0])
        for v in result.values():
            assert -1.0 <= v <= 1.0, f"tanh出力が[-1,1]外: {v}"

    def test_fill_batch_length(self, fitted_filler, contexts):
        results = fitted_filler.fill_batch(contexts)
        assert len(results) == N
        assert all(isinstance(r, dict) for r in results)

    def test_fill_batch_keys(self, fitted_filler, contexts):
        results = fitted_filler.fill_batch(contexts)
        for r in results:
            assert set(r.keys()) == set(KG_SCHEMA)

    def test_fill_rate_range(self, fitted_filler, contexts):
        rate = fitted_filler.fill_rate(contexts)
        assert 0.0 <= rate <= 1.0

    def test_bcs_descriptor_no_targets(self, fitted_filler, contexts):
        desc = fitted_filler.bcs_descriptor(contexts)
        assert desc.shape == (2,)

    def test_bcs_descriptor_with_targets(self, fitted_filler, contexts, targets):
        desc = fitted_filler.bcs_descriptor(contexts, targets=targets)
        assert desc.shape == (2,)
        assert np.all((desc >= 0) & (desc <= 1))

    def test_slot_mismatch_raises(self, contexts, targets):
        from src.interfaces.neat_slot_filler import NeatSlotFiller
        sf = NeatSlotFiller(slot_names=("a", "b"), context_dim=CONTEXT_DIM,
                             pop_size=POP, generation_limit=GEN)
        with pytest.raises(AssertionError):
            sf.fit(contexts, targets)  # targets は 3列だが slots は 2つ

    def test_fill_before_fit_raises(self, contexts):
        from src.interfaces.neat_slot_filler import NeatSlotFiller
        sf = NeatSlotFiller(slot_names=KG_SCHEMA, context_dim=CONTEXT_DIM)
        with pytest.raises(AssertionError):
            sf.fill(contexts[0])


# ─── NeatKGWriter テスト ───

class TestNeatKGWriter:

    def test_import(self):
        from src.interfaces.neat_slot_filler import NeatKGWriter, KG_SCHEMA
        assert NeatKGWriter is not None

    def test_generate_triple_keys(self, fitted_filler, contexts):
        from src.interfaces.neat_slot_filler import NeatKGWriter
        writer = NeatKGWriter(neat_filler=fitted_filler)
        triple = writer.generate_triple(contexts[0])
        assert set(triple.keys()) == {"node", "relation", "weight"}

    def test_weight_in_zero_one(self, fitted_filler, contexts):
        from src.interfaces.neat_slot_filler import NeatKGWriter
        writer = NeatKGWriter(neat_filler=fitted_filler)
        for ctx in contexts[:5]:
            triple = writer.generate_triple(ctx)
            assert 0.0 <= triple["weight"] <= 1.0

    def test_write_without_store(self, fitted_filler, contexts):
        from src.interfaces.neat_slot_filler import NeatKGWriter
        writer = NeatKGWriter(neat_filler=fitted_filler, kg_store=None)
        triple = writer.write(contexts[0])
        assert isinstance(triple, dict)

    def test_wrong_schema_raises(self, contexts, targets):
        from src.interfaces.neat_slot_filler import NeatSlotFiller, NeatKGWriter
        sf = NeatSlotFiller(slot_names=CUSTOM_SCHEMA, context_dim=CONTEXT_DIM,
                             pop_size=POP, generation_limit=GEN, seed=2)
        sf.fit(contexts, targets)
        with pytest.raises(AssertionError):
            NeatKGWriter(neat_filler=sf)
