"""
pytest — NeatGate / NeatAugmentedReward (B型 I/F)
"""

import numpy as np
import pytest

CONTEXT_DIM = 4
NUM_SKILLS = 3
N = 20
POP = 10
GEN = 5
SEED = 0


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(SEED)


@pytest.fixture(scope="module")
def contexts(rng):
    return rng.standard_normal((N, CONTEXT_DIM)).astype(np.float32)


@pytest.fixture(scope="module")
def targets(rng):
    # 0/1 二値マスク
    return (rng.random((N, NUM_SKILLS)) > 0.5).astype(np.float32)


@pytest.fixture(scope="module")
def fitted_gate(contexts, targets):
    from src.interfaces.neat_gate import NeatGate
    gate = NeatGate(
        context_dim=CONTEXT_DIM,
        num_skills=NUM_SKILLS,
        pop_size=POP,
        species_size=3,
        max_nodes=30,
        max_conns=50,
        generation_limit=GEN,
        seed=SEED,
    )
    gate.fit(contexts, targets)
    return gate


# ─── NeatGate テスト ───

class TestNeatGate:

    def test_import(self):
        from src.interfaces.neat_gate import NeatGate
        assert NeatGate is not None

    def test_fit_returns_self(self, contexts, targets):
        from src.interfaces.neat_gate import NeatGate
        gate = NeatGate(context_dim=CONTEXT_DIM, num_skills=NUM_SKILLS,
                        pop_size=POP, generation_limit=GEN, seed=1)
        assert gate.fit(contexts, targets) is gate

    def test_fit_sets_state(self, fitted_gate):
        assert fitted_gate._state is not None
        assert fitted_gate._best_params is not None

    def test_activate_single_shape(self, fitted_gate, contexts):
        mask = fitted_gate.activate(contexts[0])
        assert mask.shape == (NUM_SKILLS,)
        assert mask.dtype == bool

    def test_activate_batch_shape(self, fitted_gate, contexts):
        masks = fitted_gate.activate(contexts)
        assert masks.shape == (N, NUM_SKILLS)
        assert masks.dtype == bool

    def test_logits_shape(self, fitted_gate, contexts):
        logits = fitted_gate.logits(contexts[0])
        assert logits.shape == (NUM_SKILLS,)
        assert np.isfinite(logits).all()

    def test_probs_range(self, fitted_gate, contexts):
        probs = fitted_gate.probs(contexts[0])
        assert probs.shape == (NUM_SKILLS,)
        assert np.all((probs >= 0) & (probs <= 1))

    def test_accuracy_range(self, fitted_gate, contexts, targets):
        acc = fitted_gate.accuracy(contexts, targets)
        assert 0.0 <= acc <= 1.0

    def test_bcs_descriptor_shape(self, fitted_gate, contexts):
        desc = fitted_gate.bcs_descriptor(contexts)
        assert desc.shape == (2,)
        assert np.all((desc >= 0) & (desc <= 1))

    def test_context_dim_mismatch_raises(self, contexts, targets):
        from src.interfaces.neat_gate import NeatGate
        gate = NeatGate(context_dim=CONTEXT_DIM + 1, num_skills=NUM_SKILLS,
                        pop_size=POP, generation_limit=GEN)
        with pytest.raises(AssertionError):
            gate.fit(contexts, targets)

    def test_activate_before_fit_raises(self, contexts):
        from src.interfaces.neat_gate import NeatGate
        gate = NeatGate(context_dim=CONTEXT_DIM, num_skills=NUM_SKILLS)
        with pytest.raises(AssertionError):
            gate.activate(contexts[0])


# ─── NeatAugmentedReward テスト ───

class TestNeatAugmentedReward:

    def test_import(self):
        from src.interfaces.neat_gate import NeatAugmentedReward
        assert NeatAugmentedReward is not None

    def test_call_returns_float(self, fitted_gate, contexts):
        from src.interfaces.neat_gate import NeatAugmentedReward
        base_reward = lambda ctx, act: 0.5
        aug = NeatAugmentedReward(neat_gate=fitted_gate, base_reward=base_reward)
        result = aug(contexts[0], np.zeros(CONTEXT_DIM, dtype=np.float32))
        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_reward_in_range(self, fitted_gate, contexts):
        from src.interfaces.neat_gate import NeatAugmentedReward
        # base_reward が [0,1] の場合、augmented も概ね有界
        base_reward = lambda ctx, act: float(np.clip(np.mean(ctx), 0, 1))
        aug = NeatAugmentedReward(neat_gate=fitted_gate, base_reward=base_reward,
                                   gate_weight=0.3)
        rewards = [aug(ctx, np.zeros(CONTEXT_DIM)) for ctx in contexts[:5]]
        assert all(np.isfinite(r) for r in rewards)
