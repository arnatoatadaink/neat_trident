"""pytest — src.pytorch_proto.map_elites.GridArchive (QDax-free MAP-Elites feasibility prototype)"""

import numpy as np
import pytest
import torch

from src.pytorch_proto.map_elites import GridArchive, make_grid_centroids

SEED = 0


# Reimplemented in plain numpy rather than imported from src.map_elites_archive: importing
# torch then triggering JAX PJRT backend init in the same process hung indefinitely here
# (see docs/pytorch_map_elites_replacement_spike_20260714.md).
def _reference_grid_centroids(grid_size: int) -> np.ndarray:
    coords = np.linspace(0.0, 1.0, grid_size, dtype=np.float32)
    xx, yy = np.meshgrid(coords, coords)
    return np.stack([xx.ravel(), yy.ravel()], axis=1)


def _reference_resolve_cell(centroids: np.ndarray, descriptor: np.ndarray) -> int:
    dists = np.sum((centroids - descriptor) ** 2, axis=1)
    return int(np.argmin(dists))


@pytest.fixture
def rng():
    return np.random.default_rng(SEED)


def _descriptor(rng) -> torch.Tensor:
    return torch.from_numpy(rng.random(2).astype(np.float32))


class TestGridArchive:

    def test_initial_filled_cells(self):
        archive = GridArchive(grid_size=4)
        assert archive.filled_cells == 0

    def test_initial_coverage(self):
        archive = GridArchive(grid_size=4)
        assert archive.coverage == 0.0

    def test_add_increases_filled(self, rng):
        archive = GridArchive(grid_size=4)
        adopted, _ = archive.add(_descriptor(rng), 0.5)
        assert adopted
        assert archive.filled_cells == 1

    def test_better_fitness_replaces(self):
        archive = GridArchive(grid_size=4)
        desc = torch.tensor([0.5, 0.5])
        archive.add(desc, 0.3)
        adopted, _ = archive.add(desc, 0.9)
        assert adopted
        assert archive.best_fitness == pytest.approx(0.9, abs=1e-5)

    def test_worse_fitness_not_replaced(self):
        archive = GridArchive(grid_size=4)
        desc = torch.tensor([0.5, 0.5])
        archive.add(desc, 0.9)
        adopted, _ = archive.add(desc, 0.1)
        assert not adopted
        assert archive.best_fitness == pytest.approx(0.9, abs=1e-5)

    def test_qd_score_ignores_empty_cells(self):
        archive = GridArchive(grid_size=4)
        archive.add(torch.tensor([0.1, 0.1]), 0.4)
        archive.add(torch.tensor([0.9, 0.9]), 0.6)
        assert archive.qd_score == pytest.approx(1.0, abs=1e-5)


class TestCellAssignmentEquivalence:
    """The spike's core question: does a torch-only grid agree with the existing QDax-backed
    archive's formulas on cell assignment for the same descriptors? Checked against an
    independent numpy reimplementation of those formulas (see module docstring for why the
    live jax-backed module isn't imported here), not against QDax's internal bookkeeping."""

    def test_centroids_match_reference_formula(self):
        torch_centroids = make_grid_centroids(4, torch.device("cpu")).numpy()
        ref_centroids = _reference_grid_centroids(4)
        np.testing.assert_allclose(torch_centroids, ref_centroids, atol=1e-6)

    def test_resolve_cell_matches_reference_formula(self, rng):
        grid_size = 8
        archive = GridArchive(grid_size=grid_size)
        ref_centroids = _reference_grid_centroids(grid_size)

        for _ in range(20):
            desc = rng.random(2).astype(np.float32)
            torch_cell = archive.resolve_cell(torch.from_numpy(desc))
            ref_cell = _reference_resolve_cell(ref_centroids, desc)
            assert torch_cell == ref_cell


# ## Test Guarantee Gaps
#
# - Live QDax parity: not directly exercised. Equivalence is checked against an independent
#   numpy reimplementation of map_elites_archive.py's formulas, not by running the QDax-backed
#   module in-process (torch+JAX coexistence hung in this environment). accept-risk: the
#   formulas are ~10 lines each and were read side-by-side at write time; re-verify by running
#   test_map_elites.py::_resolve_cell values against this file's reference fn in a separate
#   process if QDax's internal cell math is ever suspected to diverge.
# - Descriptor values outside [0, 1]^2: not tested. accept-risk: SkillRecord/BCS descriptors
#   are produced by the existing normalized fitness functions; out-of-range inputs aren't a
#   realistic call path for either the JAX or torch archive.
# - GPU device placement (device="cuda"/"rocm"): not tested, CPU tensors only. add-test: once
#   an actual port target device is chosen, add a parametrized device fixture.
# - Concurrent/batched .add() calls: not tested — QDax's `.add()` is a batched, vectorized op;
#   this prototype's `.add()` is single-record only. add-test: needed before this could replace
#   EvolutionLoop's per-iteration usage 1:1 at scale, though EvolutionLoop itself already calls
#   add() one record at a time so this isn't a blocker for the spike's scope.
# - Tie-breaking when a descriptor is equidistant from two centroids: not tested. accept-risk:
#   torch.argmin and np.argmin both return the first minimal index for ties, so behavior should
#   match, but this specific edge case has no dedicated assertion.
# - N/A: concurrency/ordering (single-threaded, no shared mutable state across processes)
# - N/A: time/locale (no time- or locale-dependent logic in this module)
