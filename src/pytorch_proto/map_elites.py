"""Minimal grid MAP-Elites archive replicating the QDax `MapElitesRepertoire` surface actually
used by `src/map_elites_archive.py` (init_default/add/fitnesses/centroids, nearest-centroid cell
resolution, elitist replace-if-better) — feasibility prototype for pytorch-map-elites-replacement-spike."""

from __future__ import annotations

from dataclasses import dataclass

import torch


def make_grid_centroids(grid_size: int, device: torch.device) -> torch.Tensor:
    """[0, 1]^2 uniform grid centroids, shape (grid_size**2, 2) — mirrors make_grid_centroids in map_elites_archive.py."""
    coords = torch.linspace(0.0, 1.0, grid_size, device=device)
    xx, yy = torch.meshgrid(coords, coords, indexing="xy")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)


@dataclass
class GridArchive:
    """Elitist grid archive: at most one (genotype, fitness) per nearest-centroid cell."""

    grid_size: int
    device: torch.device = torch.device("cpu")

    def __post_init__(self) -> None:
        self.num_cells: int = self.grid_size**2
        self.centroids: torch.Tensor = make_grid_centroids(self.grid_size, self.device)  # (num_cells, 2)
        self.fitnesses: torch.Tensor = torch.full((self.num_cells,), float("-inf"), device=self.device)
        self.genotypes: torch.Tensor = torch.zeros((self.num_cells, 2), device=self.device)

    def resolve_cell(self, descriptor: torch.Tensor) -> int:
        """Nearest centroid by squared Euclidean distance — must match src.map_elites_archive._resolve_cell."""
        dists = torch.sum((self.centroids - descriptor) ** 2, dim=1)
        return int(torch.argmin(dists).item())

    def add(self, descriptor: torch.Tensor, fitness: float) -> tuple[bool, int]:
        """Replace the cell's occupant only if fitness improves on it; returns (adopted, cell_index)."""
        cell = self.resolve_cell(descriptor)
        if fitness > float(self.fitnesses[cell].item()):
            self.fitnesses[cell] = fitness
            self.genotypes[cell] = descriptor
            return True, cell
        return False, cell

    @property
    def filled_cells(self) -> int:
        return int(torch.sum(self.fitnesses > float("-inf")).item())

    @property
    def coverage(self) -> float:
        return self.filled_cells / self.num_cells

    @property
    def best_fitness(self) -> float | None:
        valid = self.fitnesses[self.fitnesses > float("-inf")]
        if valid.numel() == 0:
            return None
        return float(torch.max(valid).item())

    @property
    def qd_score(self) -> float:
        """QD score = sum(max(fitness_per_cell, 0)); empty cells (-inf) clamp to 0."""
        return float(torch.clamp(self.fitnesses, min=0.0).sum().item())
