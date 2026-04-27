"""
TRIDENT — 仮説E Phase 3: HyperbolicAssociationFn
ポアンカレ球ベースの3項スコア関数 (geoopt 必須)

score = 1/(1+d(h_q, h_c)) + ctx_weight*(1/(1+d(h_q, h_ctx)) + 1/(1+d(h_c, h_ctx)))
h_x   = expmap0(scale * x_euclidean)

ファイル: src/med_integration/hyperbolic_association.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _require_geoopt():
    try:
        import geoopt
        import torch
        return geoopt, torch
    except ImportError as e:
        raise ImportError(
            "geoopt が必要です: poetry run pip install geoopt"
        ) from e


class HyperbolicAssociationFn:
    """
    ポアンカレ球ベースの3項スコア関数。

    euclidean embedding を expmap0 でポアンカレ球に射影し、
    双曲距離ベースのスコアを計算する。

    scale パラメータで射影後のノルムを調整する (デフォルト 0.5 → ノルム ≈ 0.46)。
    """

    def __init__(
        self,
        c: float = 1.0,
        ctx_weight: float = 0.3,
        scale: float = 0.5,
    ):
        geoopt, torch = _require_geoopt()
        self._geoopt = geoopt
        self._torch = torch

        self.c = c
        self.ctx_weight = ctx_weight
        self.scale = scale
        self.manifold = geoopt.PoincareBall(c=c)

        self.arch_meta: dict = {
            "arch_type": "hyperbolic",
            "c": c,
            "ctx_weight": ctx_weight,
            "scale": scale,
            "generation": 0,
        }

    # ─── 内部ユーティリティ ────────────────────

    def _to_hyp(self, x: np.ndarray):
        """numpy (dim,) → ポアンカレ球上の torch Tensor (float64)"""
        t = self._torch.tensor(
            x.astype(np.float64) * self.scale,
            dtype=self._torch.float64,
        )
        return self.manifold.expmap0(t)

    def _hyp_dist(self, h1, h2) -> float:
        return float(self.manifold.dist(h1, h2).item())

    # ─── スコア計算 ────────────────────────────

    def score(
        self,
        query: np.ndarray,
        candidate: np.ndarray,
        context: np.ndarray | None = None,
    ) -> float:
        """
        Parameters
        ----------
        query     : (dim,)
        candidate : (dim,)
        context   : (dim,) or None

        Returns
        -------
        score : float  (高いほど関連度が高い)
        """
        h_q = self._to_hyp(query)
        h_c = self._to_hyp(candidate)

        s = 1.0 / (1.0 + self._hyp_dist(h_q, h_c))

        if context is not None:
            h_ctx = self._to_hyp(context)
            s += self.ctx_weight * (
                1.0 / (1.0 + self._hyp_dist(h_q, h_ctx))
                + 1.0 / (1.0 + self._hyp_dist(h_c, h_ctx))
            )

        return s

    def score_batch(
        self,
        query: np.ndarray,
        candidates: np.ndarray,
        context: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        query      : (dim,)
        candidates : (n, dim)
        context    : (dim,) or None

        Returns
        -------
        scores : (n,) float64
        """
        return np.array(
            [self.score(query, c, context) for c in candidates],
            dtype=np.float64,
        )

    # ─── 永続化 ───────────────────────────────

    def to_dict(self) -> dict:
        return {
            "arch_type":  "hyperbolic",
            "c":          self.c,
            "ctx_weight": self.ctx_weight,
            "scale":      self.scale,
            "generation": self.arch_meta["generation"],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "HyperbolicAssociationFn":
        fn = cls(
            c=d["c"],
            ctx_weight=d["ctx_weight"],
            scale=d.get("scale", 0.5),
        )
        fn.arch_meta["generation"] = d.get("generation", 0)
        return fn

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
        )

    @classmethod
    def load(cls, path: str | Path) -> "HyperbolicAssociationFn":
        return cls.from_dict(json.loads(Path(path).read_text()))
