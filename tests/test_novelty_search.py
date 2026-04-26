"""
pytest — NoveltyArchive / NoveltyFitness / knn_novelty_scores
"""

import numpy as np
import jax.numpy as jnp
import pytest

BEHAVIOR_DIM = 2
SEED = 0


@pytest.fixture
def rng():
    return np.random.default_rng(SEED)


@pytest.fixture
def archive():
    from src.novelty_search import NoveltyArchive
    return NoveltyArchive(
        behavior_dim=BEHAVIOR_DIM,
        max_size=20,
        add_prob=0.0,
        novelty_threshold=0.05,
        k_neighbors=3,
        seed=SEED,
    )


# ─── knn_novelty_scores テスト ───

class TestKnnNoveltyScores:

    def test_import(self):
        from src.novelty_search import knn_novelty_scores
        assert knn_novelty_scores is not None

    def test_output_shape(self, rng):
        from src.novelty_search import knn_novelty_scores
        cands = jnp.array(rng.random((5, BEHAVIOR_DIM), dtype=np.float32))
        arch  = jnp.array(rng.random((10, BEHAVIOR_DIM), dtype=np.float32))
        scores = knn_novelty_scores(cands, arch, k=3)
        assert scores.shape == (5,)

    def test_scores_non_negative(self, rng):
        from src.novelty_search import knn_novelty_scores
        cands = jnp.array(rng.random((4, BEHAVIOR_DIM), dtype=np.float32))
        arch  = jnp.array(rng.random((8, BEHAVIOR_DIM), dtype=np.float32))
        scores = knn_novelty_scores(cands, arch, k=2)
        assert np.all(np.array(scores) >= 0)

    def test_identical_candidate_low_score(self):
        from src.novelty_search import knn_novelty_scores
        # 候補がアーカイブと完全に同じ → 距離 ≈ 0
        pts = jnp.ones((3, BEHAVIOR_DIM), dtype=jnp.float32)
        scores = knn_novelty_scores(pts[:1], pts, k=2)
        assert float(scores[0]) < 0.01

    def test_distant_candidate_high_score(self):
        from src.novelty_search import knn_novelty_scores
        arch  = jnp.zeros((5, BEHAVIOR_DIM), dtype=jnp.float32)
        cand  = jnp.ones((1, BEHAVIOR_DIM), dtype=jnp.float32) * 100.0
        scores = knn_novelty_scores(cand, arch, k=3)
        assert float(scores[0]) > 1.0


# ─── NoveltyArchive テスト ───

class TestNoveltyArchive:

    def test_import(self):
        from src.novelty_search import NoveltyArchive
        assert NoveltyArchive is not None

    def test_initial_size(self, archive):
        assert archive.size == 0

    def test_behaviors_array_empty(self, archive):
        assert archive.behaviors_array is None

    def test_compute_novelty_empty_archive(self, archive, rng):
        bv = rng.random(BEHAVIOR_DIM).astype(np.float32)
        score = archive.compute_novelty(bv)
        assert score == 1.0  # アーカイブ空 → 最大新規性

    def test_try_add_novel_behavior(self, rng):
        from src.novelty_search import NoveltyArchive
        arch = NoveltyArchive(behavior_dim=BEHAVIOR_DIM, novelty_threshold=0.0,
                               add_prob=0.0, k_neighbors=1, seed=0)
        bv = rng.random(BEHAVIOR_DIM).astype(np.float32)
        added, score = arch.try_add(bv, skill_type="indexer")
        assert added
        assert arch.size == 1

    def test_try_add_below_threshold(self, rng):
        from src.novelty_search import NoveltyArchive
        arch = NoveltyArchive(behavior_dim=BEHAVIOR_DIM, novelty_threshold=999.0,
                               add_prob=0.0, k_neighbors=1, seed=0)
        # 1つ追加してアーカイブを初期化
        bv0 = np.zeros(BEHAVIOR_DIM, dtype=np.float32)
        arch.try_add(bv0, skill_type="indexer")
        # 同じ点 → 新規性 ≈ 0 < 999 → 追加されない
        added, _ = arch.try_add(bv0.copy(), skill_type="indexer")
        assert not added

    def test_compute_novelty_batch(self, rng):
        from src.novelty_search import NoveltyArchive
        arch = NoveltyArchive(behavior_dim=BEHAVIOR_DIM, novelty_threshold=0.0,
                               add_prob=1.0, k_neighbors=2, seed=0)
        for _ in range(5):
            arch.try_add(rng.random(BEHAVIOR_DIM).astype(np.float32), "indexer")
        batch = rng.random((4, BEHAVIOR_DIM)).astype(np.float32)
        scores = arch.compute_novelty_batch(batch)
        assert scores.shape == (4,)
        assert np.all(scores >= 0)

    def test_max_size_fifo(self, rng):
        from src.novelty_search import NoveltyArchive
        arch = NoveltyArchive(behavior_dim=BEHAVIOR_DIM, max_size=3,
                               novelty_threshold=0.0, add_prob=1.0,
                               k_neighbors=1, seed=0)
        for _ in range(5):
            arch.try_add(rng.random(BEHAVIOR_DIM).astype(np.float32), "indexer")
        assert arch.size <= 3

    def test_behaviors_array_shape(self, rng):
        from src.novelty_search import NoveltyArchive
        arch = NoveltyArchive(behavior_dim=BEHAVIOR_DIM, novelty_threshold=0.0,
                               add_prob=1.0, k_neighbors=1, seed=0)
        for _ in range(4):
            arch.try_add(rng.random(BEHAVIOR_DIM).astype(np.float32), "indexer")
        arr = arch.behaviors_array
        assert arr is not None
        assert arr.shape[1] == BEHAVIOR_DIM

    def test_most_novel(self, rng):
        from src.novelty_search import NoveltyArchive
        arch = NoveltyArchive(behavior_dim=BEHAVIOR_DIM, novelty_threshold=0.0,
                               add_prob=1.0, k_neighbors=1, seed=0)
        for _ in range(6):
            arch.try_add(rng.random(BEHAVIOR_DIM).astype(np.float32), "indexer")
        top = arch.most_novel(3)
        assert len(top) <= 3
        # 新規性スコアが降順
        scores = [r.novelty_score for r in top]
        assert scores == sorted(scores, reverse=True)

    def test_summary_keys(self, rng):
        from src.novelty_search import NoveltyArchive
        arch = NoveltyArchive(behavior_dim=BEHAVIOR_DIM, novelty_threshold=0.0,
                               add_prob=1.0, k_neighbors=1, seed=0)
        arch.try_add(rng.random(BEHAVIOR_DIM).astype(np.float32), "indexer")
        s = arch.summary()
        assert "size" in s and "mean_novelty" in s and "max_novelty" in s


# ─── NoveltyFitness テスト ───

class TestNoveltyFitness:

    def test_import(self):
        from src.novelty_search import NoveltyFitness
        assert NoveltyFitness is not None

    def test_combined_fitness_range(self, rng):
        from src.novelty_search import NoveltyArchive, NoveltyFitness
        arch = NoveltyArchive(behavior_dim=BEHAVIOR_DIM, novelty_threshold=0.0,
                               add_prob=1.0, k_neighbors=1, seed=0)
        for _ in range(3):
            arch.try_add(rng.random(BEHAVIOR_DIM).astype(np.float32), "indexer")
        nf = NoveltyFitness(ns_archive=arch, alpha=0.5)
        bv = rng.random(BEHAVIOR_DIM).astype(np.float32)
        score = nf(bv, task_fitness=-0.1)
        assert np.isfinite(score)

    def test_alpha_zero_uses_task_fitness(self, rng):
        from src.novelty_search import NoveltyArchive, NoveltyFitness
        arch = NoveltyArchive(behavior_dim=BEHAVIOR_DIM, novelty_threshold=0.0,
                               add_prob=1.0, k_neighbors=1, seed=0)
        arch.try_add(rng.random(BEHAVIOR_DIM).astype(np.float32), "indexer")
        nf = NoveltyFitness(ns_archive=arch, alpha=0.0)
        bv = rng.random(BEHAVIOR_DIM).astype(np.float32)
        # alpha=0 → novelty の影響なし → task_fitness 由来の値
        score = nf(bv, task_fitness=0.0)  # norm_task = 1/(1+0) = 1.0
        assert abs(score - 1.0) < 1e-5
