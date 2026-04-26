"""
TRIDENT — 仮説E: 文脈依存連想検索
AssociationFn + ContextSensitiveSearch

設計思想:
  score = w0*cos(q,c) + w1*cos(q,ctx) + w2*cos(c,ctx) + w3*cos(q-ctx,c)
  将来は TRIDENT が NEAT でアーキテクチャを差し替えられるプラグイン構造

ファイル: src/med_integration/context_search.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np


# ──────────────────────────────────────────────
# Phase 4: TRIDENT プラグイン Protocol
# ──────────────────────────────────────────────

class AssociationFnProtocol(Protocol):
    """
    TRIDENT が差し替え可能な association_fn の Protocol。
    MLP固定版 (AssociationFn) と NEAT進化版 (NEATAssociationFn) が共通で満たす。
    """
    def score(
        self,
        query:     np.ndarray,
        candidate: np.ndarray,
        context:   np.ndarray | None,
    ) -> float: ...

    def score_batch(
        self,
        query:     np.ndarray,
        candidates: np.ndarray,
        context:   np.ndarray | None,
    ) -> np.ndarray: ...

    def to_dict(self) -> dict: ...

    @classmethod
    def from_dict(cls, d: dict) -> "AssociationFnProtocol": ...


# ──────────────────────────────────────────────
# SearchResult
# ──────────────────────────────────────────────

@dataclass
class SearchResult:
    index:       int
    text:        str
    base_score:  float          # FAISS / コサイン類似度
    assoc_score: float          # association_fn の出力
    final_score: float          # alpha * base + (1-alpha) * assoc
    embedding:   np.ndarray     # 候補の埋め込みベクトル


# ──────────────────────────────────────────────
# AssociationFn
# ──────────────────────────────────────────────

class AssociationFn:
    """
    3項スコア関数: score = f(query, candidate, context)

    現状: MLP固定アーキテクチャ (numpy, torch不要)
    将来: TRIDENT が NEAT でアーキテクチャを差し替え可能

    重み w = [w0, w1, w2, w3] の意味:
      w0: cos(query, candidate)          — 直接類似度
      w1: cos(query, context)            — クエリ・文脈類似度
      w2: cos(candidate, context)        — 候補・文脈類似度
      w3: cos(query - context, candidate) — 文脈差分ベクトルとの類似度
    """

    def __init__(self, weights: list[float] | None = None):
        if weights is not None:
            w = np.array(weights, dtype=np.float64)
        else:
            w = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64)
        # 正規化して合計1にする
        self.weights: np.ndarray = w / (w.sum() + 1e-12)

        # TRIDENT 進化用メタデータ
        self.arch_meta: dict[str, Any] = {
            "arch_type": "mlp",
            "n_weights": 4,
            "fitness_history": [],
            "generation": 0,
        }

    # ─── スコア計算 ────────────────────────────

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-12 or nb < 1e-12:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

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
        """
        if context is None:
            # 文脈なし: w0 のみ有効
            return self._cosine(query, candidate)

        f0 = self._cosine(query, candidate)
        f1 = self._cosine(query, context)
        f2 = self._cosine(candidate, context)
        diff = query - context
        f3 = self._cosine(diff, candidate)

        return float(
            self.weights[0] * f0
            + self.weights[1] * f1
            + self.weights[2] * f2
            + self.weights[3] * f3
        )

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
        scores : (n,)
        """
        return np.array(
            [self.score(query, c, context) for c in candidates],
            dtype=np.float64,
        )

    # ─── 重み学習 ─────────────────────────────

    def fit(
        self,
        feedback_pairs: list[dict],
        lr: float = 0.1,
    ) -> None:
        """
        フィードバックペアから重みを更新する (勾配なし簡易版)。

        feedback_pairs の各要素:
          {
            "query":     np.ndarray (dim,)
            "candidate": np.ndarray (dim,)
            "context":   np.ndarray (dim,) or None
            "label":     float  (1.0=関連, 0.0=非関連)
          }
        """
        if not feedback_pairs:
            return

        grad = np.zeros(4, dtype=np.float64)
        for pair in feedback_pairs:
            q   = pair["query"]
            c   = pair["candidate"]
            ctx = pair.get("context")
            lbl = float(pair["label"])

            if ctx is None:
                continue

            feats = np.array([
                self._cosine(q, c),
                self._cosine(q, ctx),
                self._cosine(c, ctx),
                self._cosine(q - ctx, c),
            ])
            pred  = float(self.weights @ feats)
            error = lbl - pred
            grad += error * feats

        self.weights += lr * grad / len(feedback_pairs)
        # 非負 & 正規化
        self.weights = np.clip(self.weights, 0, None)
        s = self.weights.sum()
        if s > 1e-12:
            self.weights /= s

        self.arch_meta["generation"] += 1
        self.arch_meta["fitness_history"].append(float(np.mean(
            [(p["label"] - self.score(p["query"], p["candidate"], p.get("context"))) ** 2
             for p in feedback_pairs]
        )))

    # ─── 永続化 ───────────────────────────────

    def to_dict(self) -> dict:
        return {
            "arch_type":      self.arch_meta["arch_type"],
            "weights":        self.weights.tolist(),
            "fitness_history": self.arch_meta["fitness_history"],
            "generation":     self.arch_meta["generation"],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AssociationFn":
        fn = cls(weights=d["weights"])
        fn.arch_meta["fitness_history"] = d.get("fitness_history", [])
        fn.arch_meta["generation"]      = d.get("generation", 0)
        return fn

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, path: str | Path) -> "AssociationFn":
        return cls.from_dict(json.loads(Path(path).read_text()))


# ──────────────────────────────────────────────
# ContextSensitiveSearch
# ──────────────────────────────────────────────

class ContextSensitiveSearch:
    """
    文脈依存検索: FAISS (or numpy fallback) + AssociationFn rerankingの組み合わせ。

    build_index(embeddings, texts)
    search(query_emb, context_emb, k, alpha) -> list[SearchResult]
    """

    def __init__(
        self,
        association_fn: AssociationFn | None = None,
        normalize: bool = True,
    ):
        self.assoc_fn  = association_fn or AssociationFn()
        self.normalize = normalize

        self._embeddings: np.ndarray | None = None  # (n, dim)
        self._texts:      list[str]          = []
        self._faiss_index = None

    # ─── インデックス構築 ──────────────────────

    def build_index(self, embeddings: np.ndarray, texts: list[str]) -> None:
        """
        Parameters
        ----------
        embeddings : (n, dim) float32
        texts      : list[str] length n
        """
        emb = np.array(embeddings, dtype=np.float32)
        if self.normalize:
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            emb = emb / np.where(norms < 1e-12, 1.0, norms)

        self._embeddings = emb
        self._texts      = list(texts)

        # FAISS インデックス構築 (faiss 未インストール時は numpy fallback)
        try:
            import faiss
            dim = emb.shape[1]
            index = faiss.IndexFlatIP(dim)  # 内積 (正規化済みなのでコサイン)
            index.add(emb)
            self._faiss_index = index
        except ImportError:
            self._faiss_index = None

    # ─── 検索 ──────────────────────────────────

    def search(
        self,
        query_emb:   np.ndarray,
        context_emb: np.ndarray | None = None,
        k:           int = 5,
        alpha:       float = 0.5,
        prefetch_k:  int | None = None,
    ) -> list[SearchResult]:
        """
        Parameters
        ----------
        query_emb   : (dim,)
        context_emb : (dim,) or None
        k           : 返す件数
        alpha       : base_score の重み (1-alpha が assoc_score の重み)
        prefetch_k  : FAISS で先取りする件数 (デフォルト: k*3)

        Returns
        -------
        list[SearchResult] length <= k, final_score 降順
        """
        if self._embeddings is None:
            raise RuntimeError("build_index() を先に呼んでください")

        if prefetch_k is None:
            prefetch_k = min(k * 3, len(self._texts))

        q = np.array(query_emb, dtype=np.float32)
        if self.normalize:
            nq = np.linalg.norm(q)
            if nq > 1e-12:
                q = q / nq

        # ─ FAISS / numpy で候補取得 ────────────
        base_scores, indices = self._retrieve(q, prefetch_k)

        # ─ AssociationFn でリランク ────────────
        ctx = None
        if context_emb is not None:
            ctx = np.array(context_emb, dtype=np.float64)
            if self.normalize:
                nc = np.linalg.norm(ctx)
                if nc > 1e-12:
                    ctx = ctx / nc

        results: list[SearchResult] = []
        for idx, base in zip(indices, base_scores):
            cand = self._embeddings[idx].astype(np.float64)
            assoc = self.assoc_fn.score(q.astype(np.float64), cand, ctx)
            final = alpha * float(base) + (1.0 - alpha) * assoc
            results.append(SearchResult(
                index=int(idx),
                text=self._texts[idx],
                base_score=float(base),
                assoc_score=assoc,
                final_score=final,
                embedding=self._embeddings[idx].copy(),
            ))

        results.sort(key=lambda r: r.final_score, reverse=True)
        return results[:k]

    def _retrieve(
        self, query: np.ndarray, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """(scores, indices) を返す。FAISS 優先、なければ numpy。"""
        n = len(self._texts)
        k = min(k, n)

        if self._faiss_index is not None:
            scores, indices = self._faiss_index.search(query.reshape(1, -1), k)
            return scores[0], indices[0]
        else:
            # numpy ブルートフォース (正規化済みコサイン = 内積)
            sims = self._embeddings.astype(np.float32) @ query
            top_idx = np.argsort(sims)[::-1][:k]
            return sims[top_idx], top_idx

    # ─── Phase 4: プラグイン差し替え ─────────────

    def swap_association_fn(self, fn: "AssociationFnProtocol") -> None:
        """
        TRIDENT が NEAT 進化で得た association_fn を差し替える。
        インデックス再構築は不要 (検索時にのみ使われる)。
        """
        self.assoc_fn = fn  # type: ignore[assignment]

    # ─── プロパティ ───────────────────────────

    @property
    def ntotal(self) -> int:
        return len(self._texts)

    @property
    def dimension(self) -> int:
        if self._embeddings is None:
            return 0
        return self._embeddings.shape[1]
