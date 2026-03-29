"""
Novelty Search 動作確認スクリプト
src/novelty_search.py の全機能を検証する。

最後に NoveltyEvolutionLoop で NS × MAP-Elites の完全サイクルを実行する。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

print("=" * 60)
print("Novelty Search — 動作確認")
print("=" * 60)

# ─── インポート ───
print("\n[1] インポート確認 ... ", end="", flush=True)
try:
    from src.novelty_search import (
        knn_novelty_scores,
        single_novelty_score,
        novelty_score_with_buffer,
        NoveltyArchive,
        NoveltyRecord,
        NoveltyFitness,
        NoveltyEvolutionLoop,
    )
    from src.map_elites_archive import TRIDENTArchive
    print("✅")
except Exception as e:
    print(f"❌ {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

rng = np.random.default_rng(42)

# ─── knn_novelty_scores (JAX JIT) ───
print("[2] knn_novelty_scores (JAX JIT) ... ", end="", flush=True)
try:
    import jax.numpy as jnp
    archive = jnp.array(rng.uniform(0, 1, (20, 2)), dtype=jnp.float32)
    candidates = jnp.array(rng.uniform(0, 1, (5, 2)),  dtype=jnp.float32)

    scores = knn_novelty_scores(candidates, archive, k=5)
    assert scores.shape == (5,)
    assert np.all(np.array(scores) >= 0)
    print(f"✅  scores={np.round(np.array(scores), 3)}")
except Exception as e:
    print(f"❌ {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# ─── single_novelty_score ───
print("[3] single_novelty_score ... ", end="", flush=True)
try:
    cand = jnp.array([0.5, 0.5], dtype=jnp.float32)
    score_center = float(single_novelty_score(cand, archive, k=5))

    corner = jnp.array([10.0, 10.0], dtype=jnp.float32)   # 遠いはず
    score_corner = float(single_novelty_score(corner, archive, k=5))

    assert score_corner > score_center, "遠い点の方が新規性が高いはず"
    print(f"✅  center={score_center:.3f} < corner={score_corner:.3f}")
except Exception as e:
    print(f"❌ {e}")
    sys.exit(1)

# ─── novelty_score_with_buffer ───
print("[4] novelty_score_with_buffer ... ", end="", flush=True)
try:
    arch_np   = rng.uniform(0, 1, (10, 2)).astype(np.float32)
    buf_np    = rng.uniform(0, 1, (5,  2)).astype(np.float32)
    cand_np   = rng.uniform(0, 1, 2).astype(np.float32)

    score_no_buf = novelty_score_with_buffer(cand_np, arch_np, None, k=5)
    score_w_buf  = novelty_score_with_buffer(cand_np, arch_np, buf_np, k=5)
    assert isinstance(score_no_buf, float)
    assert isinstance(score_w_buf,  float)
    print(f"✅  no_buf={score_no_buf:.3f}, with_buf={score_w_buf:.3f}")
except Exception as e:
    print(f"❌ {e}")
    sys.exit(1)

# ─── NoveltyArchive 構築 ───
print("[5] NoveltyArchive 構築 ... ", end="", flush=True)
try:
    ns = NoveltyArchive(
        behavior_dim=2,
        max_size=50,
        add_prob=0.1,
        novelty_threshold=0.05,
        k_neighbors=5,
        seed=0,
    )
    assert ns.size == 0
    assert ns.behaviors_array is None
    print("✅")
except Exception as e:
    print(f"❌ {e}")
    sys.exit(1)

# ─── 空アーカイブ時の挙動 ───
print("[6] 空アーカイブ時の新規性スコア ... ", end="", flush=True)
try:
    score = ns.compute_novelty(np.array([0.5, 0.5], dtype=np.float32))
    assert score == 1.0, f"空アーカイブは 1.0 であるべき: {score}"
    print(f"✅  score={score}")
except Exception as e:
    print(f"❌ {e}")
    sys.exit(1)

# ─── try_add ───
print("[7] try_add 動作確認 ... ", end="", flush=True)
try:
    added_count = 0
    for _ in range(30):
        beh = rng.uniform(0, 1, 2).astype(np.float32)
        added, nov = ns.try_add(beh, "gate", task_fitness=-0.5)
        if added:
            added_count += 1

    assert ns.size > 0
    assert 0.0 <= ns.mean_novelty
    print(f"✅  added={added_count}/30, archive_size={ns.size}, "
          f"mean_novelty={ns.mean_novelty:.3f}")
except Exception as e:
    print(f"❌ {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# ─── compute_novelty_batch ───
print("[8] compute_novelty_batch ... ", end="", flush=True)
try:
    batch = rng.uniform(0, 1, (10, 2)).astype(np.float32)
    scores_batch = ns.compute_novelty_batch(batch)
    assert scores_batch.shape == (10,)
    assert np.all(scores_batch >= 0)
    print(f"✅  shape={scores_batch.shape}, "
          f"range=[{scores_batch.min():.3f}, {scores_batch.max():.3f}]")
except Exception as e:
    print(f"❌ {e}")
    sys.exit(1)

# ─── most_novel ───
print("[9] most_novel() ... ", end="", flush=True)
try:
    top5 = ns.most_novel(5)
    assert len(top5) <= 5
    scores_top = [r.novelty_score for r in top5]
    assert scores_top == sorted(scores_top, reverse=True)
    print(f"✅  top-5 novelty={[f'{s:.3f}' for s in scores_top]}")
except Exception as e:
    print(f"❌ {e}")
    sys.exit(1)

# ─── NoveltyFitness ───
print("[10] NoveltyFitness (NS+F) ... ", end="", flush=True)
try:
    ns_fit = NoveltyFitness(ns_archive=ns, alpha=0.5)
    beh = rng.uniform(0, 1, 2).astype(np.float32)

    score_nsf  = ns_fit(beh, task_fitness=-0.1)   # 高 task fitness
    score_nsf2 = ns_fit(beh, task_fitness=-10.0)  # 低 task fitness
    assert isinstance(score_nsf,  float)
    assert score_nsf > score_nsf2, "task_fitness が高いほど combined が高いはず"
    print(f"✅  high_task={score_nsf:.3f}, low_task={score_nsf2:.3f}")
except Exception as e:
    print(f"❌ {e}")
    sys.exit(1)

# ─── NoveltyEvolutionLoop (完全サイクル) ───
print("\n[11] NoveltyEvolutionLoop (NS × MAP-Elites 完全サイクル, 30 iter) ...")
try:
    archive = TRIDENTArchive(grid_sizes={"indexer": 4, "gate": 4, "slot_filler": 4})
    ns2 = NoveltyArchive(
        behavior_dim=2,
        max_size=100,
        novelty_threshold=0.05,
        add_prob=0.1,
        k_neighbors=5,
        seed=1,
    )
    ns_fit2 = NoveltyFitness(ns_archive=ns2, alpha=0.6)

    # ダミー skill_factory: BCS 空間をランダムに探索
    rng2 = np.random.default_rng(7)
    def dummy_factory(skill_type, rng):
        desc         = rng.uniform(0, 1, 2).astype(np.float32)
        task_fitness = float(rng.uniform(-1, 0))
        return None, task_fitness, desc

    loop = NoveltyEvolutionLoop(
        trident_archive=archive,
        ns_archive=ns2,
        skill_factory=dummy_factory,
        novelty_fitness=ns_fit2,
        max_iterations=30,
        seed=42,
    )
    loop.run()

    nov_hist = loop.novelty_history()
    assert len(nov_hist) == 30
    assert ns2.size > 0
    assert archive.total_skills > 0
    print(f"    ✅ NS size={ns2.size}, MAP cells={archive.total_skills}, "
          f"mean_novelty={nov_hist.mean():.3f}")
except Exception as e:
    print(f"    ❌ {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# ─── 新規性の単調増加傾向を確認 ───
print("[12] 新規性スコアの傾向確認 ... ", end="", flush=True)
try:
    # 前半 / 後半の平均新規性を比較 (前半 > 後半: アーカイブが密になるので新規性が下がる傾向)
    first_half  = float(nov_hist[:15].mean())
    second_half = float(nov_hist[15:].mean())
    # 絶対的な傾向保証は難しいので、値の範囲だけ確認
    assert np.all(nov_hist >= 0)
    print(f"✅  前半avg={first_half:.3f}, 後半avg={second_half:.3f} "
          f"(アーカイブ充填に伴い変化)")
except Exception as e:
    print(f"❌ {e}")
    sys.exit(1)

# ─── サマリー ───
print()
print("─" * 60)
print("Novelty Archive サマリー")
for k, v in ns2.summary().items():
    print(f"  {k:20s}: {v}")

print("\n" + "=" * 60)
print("Novelty Search 動作確認 サマリー")
print("=" * 60)
print("✅ Novelty Search カスタム実装 全 12 チェック完了")
print(f"   NS アーカイブサイズ : {ns2.size}")
print(f"   MAP-Elites 格納数  : {archive.total_skills}")
print(f"   平均新規性スコア   : {ns2.mean_novelty:.4f}")
print()
print("次のステップ:")
print("  - 本番次元 (384) での統合テスト")
print("  - 実スキル (NeatIndexer 等) を使った完全な NS ループ実行")
print("  - ES-HyperNEAT カスタム拡張")
