"""
Phase 4 動作確認スクリプト
TRIDENTArchive (QDax MAP-Elites 統合) の動作を検証する。

実際に A/B/C 型スキルを進化させ、アーカイブに追加して統計を確認する。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

print("=" * 60)
print("Phase 4 — TRIDENTArchive (MAP-Elites 統合) 動作確認")
print("=" * 60)

# ─── インポート確認 ───
print("\n[1] インポート確認 ... ", end="", flush=True)
try:
    from src.map_elites_archive import (
        TRIDENTArchive,
        SkillRepertoire,
        SkillRecord,
        EvolutionLoop,
        make_grid_centroids,
    )
    from src.interfaces.neat_indexer  import NeatIndexer
    from src.interfaces.neat_gate     import NeatGate
    from src.interfaces.neat_slot_filler import NeatSlotFiller, KG_SCHEMA
    print("✅")
except Exception as e:
    print(f"❌ {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# ─── グリッドセントロイド ───
print("[2] グリッドセントロイド生成 ... ", end="", flush=True)
try:
    c4  = make_grid_centroids(4)
    c16 = make_grid_centroids(16)
    assert c4.shape  == (16, 2)
    assert c16.shape == (256, 2)
    print(f"✅  4×4={c4.shape}, 16×16={c16.shape}")
except Exception as e:
    print(f"❌ {e}")
    sys.exit(1)

# ─── SkillRepertoire ───
print("[3] SkillRepertoire 構築・追加 ... ", end="", flush=True)
try:
    repo = SkillRepertoire("gate", grid_size=4)
    assert repo.num_cells == 16
    assert repo.filled_cells == 0

    # ダミースキルを3個追加
    rng = np.random.default_rng(0)
    adopted = 0
    for _ in range(5):
        desc    = rng.uniform(0, 1, size=2).astype(np.float32)
        fitness = float(rng.uniform(-1, 0))
        rec = SkillRecord(
            skill_type="gate",
            skill=None,
            fitness=fitness,
            descriptor=desc,
        )
        if repo.add(rec):
            adopted += 1

    assert repo.filled_cells > 0
    assert repo.coverage > 0.0
    print(f"✅  filled={repo.filled_cells}/16, adopted={adopted}/5, "
          f"coverage={repo.coverage:.1%}")
except Exception as e:
    print(f"❌ {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# ─── TRIDENTArchive 構築 ───
print("[4] TRIDENTArchive 構築 ... ", end="", flush=True)
try:
    archive = TRIDENTArchive(grid_sizes={"indexer": 4, "gate": 4, "slot_filler": 4})
    assert len(archive.repertoires) == 3
    assert archive.total_skills == 0
    print("✅")
except Exception as e:
    print(f"❌ {e}")
    sys.exit(1)

# ─── 実スキルを進化させてアーカイブに追加 ───
print("[5] A型 NeatIndexer を進化 → アーカイブ追加 ...")

rng = np.random.default_rng(3)
DIM      = 8
CORPUS_N = 20
QUERY_N  = 5

corpus  = rng.standard_normal((CORPUS_N, DIM)).astype(np.float32)
queries = rng.standard_normal((QUERY_N,  DIM)).astype(np.float32)
corpus_n  = corpus  / (np.linalg.norm(corpus,  axis=1, keepdims=True) + 1e-8)
queries_n = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-8)
true_scores = queries_n @ corpus_n.T
true_nbrs   = np.argsort(-true_scores, axis=1)[:, :3]

try:
    indexer = NeatIndexer(input_dim=DIM, pop_size=15, species_size=3,
                          generation_limit=3, k=3, seed=10)
    indexer.fit(corpus=corpus, queries=queries)

    desc    = indexer.bcs_descriptor(queries, true_neighbors=true_nbrs)
    fitness = -0.08  # fit 後の best fitness の近似値

    adopted = archive.add_indexer(indexer, fitness, desc,
                                  metadata={"dim": DIM, "corpus_n": CORPUS_N})
    print(f"    ✅ adopted={adopted}, desc={np.round(desc, 3)}, "
          f"filled={archive.repertoires['indexer'].filled_cells}")
except Exception as e:
    print(f"    ❌ {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

print("[6] B型 NeatGate を進化 → アーカイブ追加 ...")

CTX_DIM   = 8
N_SKILLS  = 4
N_SAMPLES = 30
ctx  = rng.standard_normal((N_SAMPLES, CTX_DIM)).astype(np.float32)
tgt  = (ctx[:, :N_SKILLS] > 0).astype(np.float32)

try:
    gate = NeatGate(context_dim=CTX_DIM, num_skills=N_SKILLS,
                    pop_size=15, species_size=3, generation_limit=3, seed=20)
    gate.fit(contexts=ctx, targets=tgt)

    desc    = gate.bcs_descriptor(ctx)
    fitness = -0.6

    adopted = archive.add_gate(gate, fitness, desc,
                               metadata={"ctx_dim": CTX_DIM, "skills": N_SKILLS})
    print(f"    ✅ adopted={adopted}, desc={np.round(desc, 3)}, "
          f"filled={archive.repertoires['gate'].filled_cells}")
except Exception as e:
    print(f"    ❌ {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

print("[7] C型 NeatSlotFiller を進化 → アーカイブ追加 ...")

slot_ctx = rng.standard_normal((N_SAMPLES, CTX_DIM)).astype(np.float32)
slot_tgt = np.tanh(slot_ctx[:, :3]).astype(np.float32)

try:
    filler = NeatSlotFiller(slot_names=KG_SCHEMA, context_dim=CTX_DIM,
                            pop_size=15, species_size=3, generation_limit=3, seed=30)
    filler.fit(contexts=slot_ctx, targets=slot_tgt)

    desc    = filler.bcs_descriptor(slot_ctx, targets=slot_tgt)
    fitness = -0.5

    adopted = archive.add_slot_filler(filler, fitness, desc,
                                      metadata={"schema": KG_SCHEMA})
    print(f"    ✅ adopted={adopted}, desc={np.round(desc, 3)}, "
          f"filled={archive.repertoires['slot_filler'].filled_cells}")
except Exception as e:
    print(f"    ❌ {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# ─── best_* 取得 ───
print("[8] best_skill 取得 ... ", end="", flush=True)
try:
    best_idx = archive.best_indexer()
    best_gat = archive.best_gate()
    best_slt = archive.best_slot_filler()
    assert best_idx is not None
    assert best_gat is not None
    assert best_slt is not None
    print(f"✅  indexer.fitness={best_idx.fitness:.3f}, "
          f"gate.fitness={best_gat.fitness:.3f}, "
          f"filler.fitness={best_slt.fitness:.3f}")
except Exception as e:
    print(f"❌ {e}")
    sys.exit(1)

# ─── descriptor で検索 ───
print("[9] descriptor 検索 ... ", end="", flush=True)
try:
    query_desc = np.array([0.5, 0.5], dtype=np.float32)
    found = archive.get_gate(query_desc)
    # 小さいグリッド (4×4) なので必ず何かのセルに近い
    print(f"✅  found={'あり' if found else 'なし (セルが空)'}")
except Exception as e:
    print(f"❌ {e}")
    sys.exit(1)

# ─── EvolutionLoop (ダミースキルで多様性テスト) ───
print("[10] EvolutionLoop (ダミー 12 イテレーション) ...")

try:
    archive2 = TRIDENTArchive(grid_sizes={"indexer": 4, "gate": 4, "slot_filler": 4})
    rng2 = np.random.default_rng(99)

    def dummy_factory(skill_type, rng):
        desc    = rng.uniform(0, 1, size=2).astype(np.float32)
        fitness = float(rng.uniform(-1, 0))
        return None, fitness, desc

    loop = EvolutionLoop(archive2, dummy_factory, max_iterations=12, seed=42)
    loop.run()

    total = archive2.total_skills
    assert total > 0
    print(f"    ✅ 総スキル数={total}, 履歴={len(loop.history)} イテレーション")
except Exception as e:
    print(f"    ❌ {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# ─── サマリー表示 ───
print()
archive.print_summary()

# ─── 最終サマリー ───
print("=" * 60)
print("Phase 4 動作確認 サマリー")
print("=" * 60)
print("✅ TRIDENTArchive (MAP-Elites 統合) 動作確認完了")
print(f"   実スキル格納数 : {archive.total_skills}")
print(f"   EvolutionLoop  : {len(loop.history)} iter, {archive2.total_skills} スキル蓄積")
print()
print("次のステップ:")
print("  - Novelty Search カスタム実装 (src/novelty_search.py)")
print("  - 本番次元 (384) での統合テスト")
print("  - Phase 4 統合: NS → アーカイブ更新ループ")
