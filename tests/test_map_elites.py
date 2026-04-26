"""
pytest — TRIDENTArchive / SkillRepertoire / EvolutionLoop
"""

import numpy as np
import pytest

SEED = 0


@pytest.fixture
def rng():
    return np.random.default_rng(SEED)


def _make_descriptor(rng):
    return rng.random(2).astype(np.float32)


# ─── SkillRepertoire テスト ───

class TestSkillRepertoire:

    def test_import(self):
        from src.map_elites_archive import SkillRepertoire
        assert SkillRepertoire is not None

    def test_initial_filled_cells(self):
        from src.map_elites_archive import SkillRepertoire
        repo = SkillRepertoire("indexer", grid_size=4)
        assert repo.filled_cells == 0

    def test_initial_coverage(self):
        from src.map_elites_archive import SkillRepertoire
        repo = SkillRepertoire("gate", grid_size=4)
        assert repo.coverage == 0.0

    def test_add_returns_bool(self, rng):
        from src.map_elites_archive import SkillRepertoire, SkillRecord
        repo = SkillRepertoire("indexer", grid_size=4)
        rec = SkillRecord(
            skill_type="indexer",
            skill="dummy",
            fitness=0.5,
            descriptor=_make_descriptor(rng),
        )
        result = repo.add(rec)
        assert isinstance(result, bool)

    def test_add_increases_filled(self, rng):
        from src.map_elites_archive import SkillRepertoire, SkillRecord
        repo = SkillRepertoire("indexer", grid_size=4)
        rec = SkillRecord("indexer", "s1", 0.5, _make_descriptor(rng))
        repo.add(rec)
        assert repo.filled_cells >= 1

    def test_better_fitness_replaces(self, rng):
        from src.map_elites_archive import SkillRepertoire, SkillRecord
        repo = SkillRepertoire("gate", grid_size=4)
        desc = np.array([0.5, 0.5], dtype=np.float32)
        rec1 = SkillRecord("gate", "s1", 0.3, desc.copy())
        rec2 = SkillRecord("gate", "s2", 0.9, desc.copy())
        repo.add(rec1)
        adopted = repo.add(rec2)
        assert adopted  # 高 fitness は採用される

    def test_worse_fitness_not_replaced(self, rng):
        from src.map_elites_archive import SkillRepertoire, SkillRecord
        repo = SkillRepertoire("gate", grid_size=4)
        desc = np.array([0.5, 0.5], dtype=np.float32)
        rec1 = SkillRecord("gate", "s1", 0.9, desc.copy())
        rec2 = SkillRecord("gate", "s2", 0.1, desc.copy())
        repo.add(rec1)
        adopted = repo.add(rec2)
        assert not adopted  # 低 fitness は採用されない

    def test_get_returns_record(self, rng):
        from src.map_elites_archive import SkillRepertoire, SkillRecord
        repo = SkillRepertoire("indexer", grid_size=4)
        desc = np.array([0.25, 0.25], dtype=np.float32)
        rec = SkillRecord("indexer", "myskill", 0.7, desc)
        repo.add(rec)
        result = repo.get(desc)
        assert result is not None

    def test_best_skill(self, rng):
        from src.map_elites_archive import SkillRepertoire, SkillRecord
        repo = SkillRepertoire("slot_filler", grid_size=4)
        for i in range(5):
            desc = rng.random(2).astype(np.float32)
            repo.add(SkillRecord("slot_filler", f"s{i}", float(i) * 0.1, desc))
        best = repo.best_skill()
        assert best is not None
        assert best.fitness == pytest.approx(0.4, abs=1e-5)

    def test_summary_keys(self, rng):
        from src.map_elites_archive import SkillRepertoire, SkillRecord
        repo = SkillRepertoire("indexer", grid_size=4)
        repo.add(SkillRecord("indexer", "s", 0.5, rng.random(2).astype(np.float32)))
        s = repo.summary()
        for key in ("grid_size", "num_cells", "filled_cells", "coverage", "best_fitness", "qd_score"):
            assert key in s


# ─── TRIDENTArchive テスト ───

class TestTRIDENTArchive:

    def test_import(self):
        from src.map_elites_archive import TRIDENTArchive
        assert TRIDENTArchive is not None

    def test_initial_total_skills(self):
        from src.map_elites_archive import TRIDENTArchive
        arch = TRIDENTArchive(grid_sizes={"indexer": 4, "gate": 4, "slot_filler": 4})
        assert arch.total_skills == 0

    def test_add_indexer(self, rng):
        from src.map_elites_archive import TRIDENTArchive
        arch = TRIDENTArchive(grid_sizes={"indexer": 4, "gate": 4, "slot_filler": 4})
        arch.add_indexer("skill_obj", 0.5, rng.random(2).astype(np.float32))
        assert arch.total_skills >= 1

    def test_add_gate(self, rng):
        from src.map_elites_archive import TRIDENTArchive
        arch = TRIDENTArchive(grid_sizes={"indexer": 4, "gate": 4, "slot_filler": 4})
        arch.add_gate("gate_obj", 0.6, rng.random(2).astype(np.float32))
        assert arch.total_skills >= 1

    def test_add_slot_filler(self, rng):
        from src.map_elites_archive import TRIDENTArchive
        arch = TRIDENTArchive(grid_sizes={"indexer": 4, "gate": 4, "slot_filler": 4})
        arch.add_slot_filler("sf_obj", 0.7, rng.random(2).astype(np.float32))
        assert arch.total_skills >= 1

    def test_best_indexer(self, rng):
        from src.map_elites_archive import TRIDENTArchive
        arch = TRIDENTArchive(grid_sizes={"indexer": 4, "gate": 4, "slot_filler": 4})
        arch.add_indexer("s1", 0.3, np.array([0.1, 0.1], dtype=np.float32))
        arch.add_indexer("s2", 0.9, np.array([0.9, 0.9], dtype=np.float32))
        best = arch.best_indexer()
        assert best is not None
        assert best.fitness == pytest.approx(0.9, abs=1e-5)

    def test_summary_has_all_types(self):
        from src.map_elites_archive import TRIDENTArchive
        arch = TRIDENTArchive(grid_sizes={"indexer": 4, "gate": 4, "slot_filler": 4})
        s = arch.summary()
        assert "indexer" in s and "gate" in s and "slot_filler" in s


# ─── EvolutionLoop テスト ───

class TestEvolutionLoop:

    def test_import(self):
        from src.map_elites_archive import EvolutionLoop
        assert EvolutionLoop is not None

    def test_run_returns_archive(self, rng):
        from src.map_elites_archive import TRIDENTArchive, EvolutionLoop

        def factory(stype, r):
            desc = r.random(2).astype(np.float32)
            fitness = float(r.random())
            return "dummy_skill", fitness, desc

        arch = TRIDENTArchive(grid_sizes={"indexer": 4, "gate": 4, "slot_filler": 4})
        loop = EvolutionLoop(archive=arch, skill_factory=factory,
                              max_iterations=9, seed=SEED)
        result = loop.run()
        assert result is arch

    def test_run_populates_archive(self, rng):
        from src.map_elites_archive import TRIDENTArchive, EvolutionLoop

        def factory(stype, r):
            desc = r.random(2).astype(np.float32)
            return "s", float(r.random()), desc

        arch = TRIDENTArchive(grid_sizes={"indexer": 4, "gate": 4, "slot_filler": 4})
        loop = EvolutionLoop(archive=arch, skill_factory=factory,
                              max_iterations=12, seed=SEED)
        loop.run()
        assert arch.total_skills >= 1

    def test_history_length(self, rng):
        from src.map_elites_archive import TRIDENTArchive, EvolutionLoop

        n_iter = 6

        def factory(stype, r):
            return "s", float(r.random()), r.random(2).astype(np.float32)

        arch = TRIDENTArchive(grid_sizes={"indexer": 4, "gate": 4, "slot_filler": 4})
        loop = EvolutionLoop(archive=arch, skill_factory=factory,
                              max_iterations=n_iter, seed=SEED)
        loop.run()
        assert len(loop.history) == n_iter
