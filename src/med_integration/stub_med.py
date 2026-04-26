"""
TRIDENT — MED スタブ実装
MEDIndexerProtocol / MEDSkillStoreProtocol のインメモリスタブ。

MED実装準拠 (MED/src/memory/faiss_index.py DomainIndex):
  search(query: (dim,), k) → list[tuple[doc_id: str, score: float]]
  add(doc_ids: list[str], embeddings: (n, dim)) → None

差し替え方法:
  StubMEDIndexer  → MED の DomainIndex に置き換えるだけで動作
  StubMEDSkillStore → MED のスキルストアに置き換える
"""

from __future__ import annotations

import numpy as np

from .interfaces import MEDIndexerProtocol, MEDSkillStoreProtocol


class StubMEDIndexer:
    """
    MEDIndexerProtocol のインメモリスタブ。

    MED の DomainIndex と同一シグネチャ:
      search(query: (dim,), k) → list[tuple[str, float]]
      add(doc_ids: list[str], embeddings: (n, dim)) → None

    Parameters
    ----------
    dimension : インデックスのベクトル次元数
    normalize : add() 時に L2 正規化するか (True = コサイン類似度用)
    """

    def __init__(self, dimension: int, normalize: bool = True):
        self._dimension = dimension
        self._normalize = normalize
        self._vectors: list[np.ndarray] = []
        self._doc_ids: list[str] = []

    # ─── MEDIndexerProtocol 実装 ───

    def search(self, query: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
        """
        numpy ブルートフォースでコサイン類似度 k 近傍を返す。

        MED 準拠: DomainIndex.search() と同一シグネチャ・戻値形式。

        Parameters
        ----------
        query : (D,) L2正規化済みクエリベクトル
        k     : 近傍数

        Returns
        -------
        list[tuple[doc_id: str, score: float]]  降順ソート済み
        """
        q = np.asarray(query, dtype=np.float32).ravel()
        assert q.shape == (self._dimension,), (
            f"query dim mismatch: expected {self._dimension}, got {q.shape}"
        )

        if not self._vectors:
            return []

        corpus = np.stack(self._vectors)         # (N, D)
        q_n = q / (np.linalg.norm(q) + 1e-8)
        scores = corpus @ q_n                    # (N,)

        k_eff = min(k, len(self._vectors))
        top_k_idx = np.argsort(-scores)[:k_eff]

        return [(self._doc_ids[i], float(scores[i])) for i in top_k_idx]

    def add(self, doc_ids: list[str], embeddings: np.ndarray) -> None:
        """
        ベクトル群をインデックスに追加する。

        MED 準拠: DomainIndex.add(doc_ids, embeddings) と同一シグネチャ。

        Parameters
        ----------
        doc_ids    : ドキュメントID リスト (長さ N)
        embeddings : (N, D) または (D,) ベクトル群
        """
        vecs = np.asarray(embeddings, dtype=np.float32)
        if vecs.ndim == 1:
            vecs = vecs[None]
        assert vecs.shape[1] == self._dimension, (
            f"embeddings dim mismatch: expected {self._dimension}, got {vecs.shape[1]}"
        )
        assert len(doc_ids) == len(vecs), (
            f"doc_ids length {len(doc_ids)} != embeddings rows {len(vecs)}"
        )

        if self._normalize:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
            vecs = vecs / norms

        for doc_id, v in zip(doc_ids, vecs):
            self._doc_ids.append(doc_id)
            self._vectors.append(v.copy())

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def ntotal(self) -> int:
        return len(self._vectors)

    def reset(self) -> None:
        """インデックスをリセットする。"""
        self._vectors.clear()
        self._doc_ids.clear()


class StubMEDSkillStore:
    """
    MEDSkillStoreProtocol のインメモリスタブ。

    Parameters
    ----------
    validate_keys : 必須キーの検証を行うか
    """

    REQUIRED_KEYS = {"type", "fitness", "descriptor"}

    def __init__(self, validate_keys: bool = True):
        self._validate = validate_keys
        self._store: dict[int, dict] = {}
        self._next_id: int = 0

    def store_skill(self, skill: dict) -> int:
        if self._validate:
            missing = self.REQUIRED_KEYS - set(skill.keys())
            assert not missing, f"スキル dict に必須キーがありません: {missing}"

        skill_id = self._next_id
        self._store[skill_id] = dict(skill)
        self._store[skill_id]["_id"] = skill_id
        self._next_id += 1
        return skill_id

    def get_skill(self, skill_id: int) -> dict:
        return self._store.get(skill_id, {})

    def list_skills(self, skill_type: str | None = None) -> list[dict]:
        skills = list(self._store.values())
        if skill_type is not None:
            skills = [s for s in skills if s.get("type") == skill_type]
        return skills

    @property
    def size(self) -> int:
        return len(self._store)

    def best_skill(self, skill_type: str | None = None) -> dict | None:
        candidates = self.list_skills(skill_type)
        if not candidates:
            return None
        return max(candidates, key=lambda s: s.get("fitness", -float("inf")))

    def summary(self) -> dict:
        by_type: dict[str, int] = {}
        for s in self._store.values():
            t = s.get("type", "unknown")
            by_type[t] = by_type.get(t, 0) + 1
        return {"total": self.size, "by_type": by_type}
