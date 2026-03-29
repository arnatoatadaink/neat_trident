"""
本番次元 (384) 統合テスト
─────────────────────────────────────────────────────────────
FAISS 同等の 384 次元埋め込み空間で、TRIDENT の全モジュールを結合する。

次元圧縮戦略 (ES-HyperNEAT 的アプローチ):
  384 次元入力
    ↓ ランダム射影 (固定 JAX 行列)  384 → PROJ_DIM
  PROJ_DIM 次元で NEAT 動作
    ↓ (A型のみ) 逆射影 PROJ_DIM → 384
  384 次元出力 (コサイン類似度スコア)

射影行列はランダム直交行列 (RP: Random Projection) で固定。
学習対象は NEAT トポロジ重みのみ。

テスト構成:
  corpus   : 100 サンプル × 384 次元 (合成 MiniLM 相当)
  queries  :  20 サンプル × 384 次元
  PROJ_DIM : 32  (計算可能な最小次元)
  世代数   :  5  (統合テスト用; 本番は 100+)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import jax.numpy as jnp
import time

print("=" * 65)
print("本番次元 (384) 統合テスト")
print("=" * 65)

# ─── 定数 ───
DIM      = 384    # FAISS / MiniLM 次元
PROJ_DIM = 32     # NEAT 動作次元
CORPUS_N = 100
QUERY_N  = 20
K        = 5
N_SKILLS = 8      # B型スキル数
SLOT_SCH = ("node", "relation", "weight")  # C型スキーマ
GEN      = 5      # 世代数 (統合テスト用)
POP      = 30     # 集団サイズ

# ─── インポート ───
print("\n[1] モジュールインポート ... ", end="", flush=True)
try:
    from src.interfaces.neat_indexer    import NeatIndexer, HybridIndexer
    from src.interfaces.neat_gate       import NeatGate, NeatAugmentedReward
    from src.interfaces.neat_slot_filler import NeatSlotFiller, NeatKGWriter, KG_SCHEMA
    from src.map_elites_archive         import TRIDENTArchive, EvolutionLoop
    from src.novelty_search             import (
        NoveltyArchive, NoveltyFitness, NoveltyEvolutionLoop
    )
    print("✅")
except Exception as e:
    print(f"❌ {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# ─── 合成 384 次元データ生成 ───
print("[2] 合成 384 次元データ生成 ... ", end="", flush=True)
try:
    rng = np.random.default_rng(0)
    corpus_384  = rng.standard_normal((CORPUS_N, DIM)).astype(np.float32)
    queries_384 = rng.standard_normal((QUERY_N,  DIM)).astype(np.float32)

    # 正規化
    corpus_384  /= np.linalg.norm(corpus_384,  axis=1, keepdims=True) + 1e-8
    queries_384 /= np.linalg.norm(queries_384, axis=1, keepdims=True) + 1e-8

    # ground truth
    true_scores = queries_384 @ corpus_384.T          # (Q, C)
    true_nbrs   = np.argsort(-true_scores, axis=1)[:, :K]  # (Q, K)

    print(f"✅  corpus={corpus_384.shape}, queries={queries_384.shape}")
except Exception as e:
    print(f"❌ {e}")
    sys.exit(1)

# ─── ランダム射影行列 384 → 32 ───
print("[3] ランダム射影行列 (384→32) 生成 ... ", end="", flush=True)
try:
    # 直交ランダム射影 (Johnson-Lindenstrauss)
    rng_proj = np.random.default_rng(42)
    R_raw    = rng_proj.standard_normal((DIM, PROJ_DIM)).astype(np.float32)
    # 列を正規化 (近似直交)
    R = R_raw / (np.linalg.norm(R_raw, axis=0, keepdims=True) + 1e-8)

    def project(x: np.ndarray) -> np.ndarray:
        """384 次元 → 32 次元射影"""
        return x @ R   # (..., PROJ_DIM)

    corpus_32  = project(corpus_384)   # (100, 32)
    queries_32 = project(queries_384)  # ( 20, 32)

    # 射影後の正規化
    corpus_32  /= np.linalg.norm(corpus_32,  axis=1, keepdims=True) + 1e-8
    queries_32 /= np.linalg.norm(queries_32, axis=1, keepdims=True) + 1e-8

    print(f"✅  射影行列 {R.shape}, corpus_32={corpus_32.shape}")
except Exception as e:
    print(f"❌ {e}")
    sys.exit(1)

# ─── A型: NeatIndexer (32 次元で動作, 384次元で評価) ───
print(f"\n[4] A型 NeatIndexer 384→32 射影版 ({GEN}世代) ...")
try:
    t0 = time.time()
    indexer = NeatIndexer(
        input_dim=PROJ_DIM,
        pop_size=POP,
        species_size=5,
        generation_limit=GEN,
        k=K,
        seed=1,
    )
    indexer.fit(corpus=corpus_32, queries=queries_32)

    # 384 次元クエリを射影してから検索
    q384 = queries_384[0]                    # 1本目クエリ (384)
    q32  = project(q384[None])[0]            # → 32次元
    q32  /= np.linalg.norm(q32) + 1e-8

    idx, scores = indexer.search(q32, k=K)
    desc_a = indexer.bcs_descriptor(queries_32, true_neighbors=true_nbrs)

    elapsed = time.time() - t0
    print(f"    ✅ {elapsed:.1f}s  top-{K} indices={idx}, desc={np.round(desc_a, 3)}")
except Exception as e:
    print(f"    ❌ {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# ─── B型: NeatGate (32次元文脈) ───
print(f"[5] B型 NeatGate 384→32 射影版 ({GEN}世代) ...")
try:
    t0 = time.time()
    # 文脈: 射影済み 32次元クエリ
    # ターゲット: クエリの前半 8次元の符号でスキル選択
    gate_ctx = queries_32                     # (20, 32)
    gate_tgt = (queries_32[:, :N_SKILLS] > 0).astype(np.float32)  # (20, 8)

    gate = NeatGate(
        context_dim=PROJ_DIM,
        num_skills=N_SKILLS,
        pop_size=POP,
        species_size=5,
        generation_limit=GEN,
        seed=2,
    )
    gate.fit(contexts=gate_ctx, targets=gate_tgt)

    mask = gate.activate(queries_32[0])
    acc  = gate.accuracy(gate_ctx, gate_tgt)
    desc_b = gate.bcs_descriptor(gate_ctx)

    elapsed = time.time() - t0
    print(f"    ✅ {elapsed:.1f}s  mask={mask}, acc={acc:.3f}, desc={np.round(desc_b, 3)}")
except Exception as e:
    print(f"    ❌ {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# ─── C型: NeatSlotFiller (32次元文脈) ───
print(f"[6] C型 NeatSlotFiller 384→32 射影版 ({GEN}世代) ...")
try:
    t0 = time.time()
    slot_ctx = queries_32                    # (20, 32)
    slot_tgt = np.tanh(queries_32[:, :3])   # (20, 3) ← 最初の 3 成分

    filler = NeatSlotFiller(
        slot_names=KG_SCHEMA,
        context_dim=PROJ_DIM,
        pop_size=POP,
        species_size=5,
        generation_limit=GEN,
        seed=3,
    )
    filler.fit(contexts=slot_ctx, targets=slot_tgt)

    slots   = filler.fill(queries_32[0])
    fr      = filler.fill_rate(slot_ctx)
    desc_c  = filler.bcs_descriptor(slot_ctx, targets=slot_tgt)
    writer  = NeatKGWriter(neat_filler=filler)
    triple  = writer.generate_triple(queries_32[0])

    elapsed = time.time() - t0
    print(f"    ✅ {elapsed:.1f}s  slots={slots}")
    print(f"         triple={triple}, fill_rate={fr:.3f}, desc={np.round(desc_c, 3)}")
except Exception as e:
    print(f"    ❌ {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# ─── TRIDENTArchive に格納 ───
print("\n[7] TRIDENTArchive に A/B/C 型スキルを格納 ... ", end="", flush=True)
try:
    archive = TRIDENTArchive()   # 本番サイズ: 64×64 / 16×16 / 16×16

    ok_a = archive.add_indexer(
        indexer,
        fitness=float(-0.1),
        descriptor=desc_a,
        metadata={"dim": DIM, "proj_dim": PROJ_DIM, "corpus_n": CORPUS_N},
    )
    ok_b = archive.add_gate(
        gate,
        fitness=float(-0.6),
        descriptor=desc_b,
        metadata={"context_dim": DIM, "num_skills": N_SKILLS},
    )
    ok_c = archive.add_slot_filler(
        filler,
        fitness=float(-0.5),
        descriptor=desc_c,
        metadata={"schema": KG_SCHEMA},
    )
    assert all([ok_a, ok_b, ok_c]), f"格納失敗: A={ok_a}, B={ok_b}, C={ok_c}"
    print(f"✅  A={ok_a}, B={ok_b}, C={ok_c}, total={archive.total_skills}")
except Exception as e:
    print(f"❌ {e}")
    sys.exit(1)

# ─── NoveltyEvolutionLoop (射影版スキルで NS サイクル) ───
print("[8] NoveltyEvolutionLoop (射影版 NS × MAP-Elites, 15 iter) ...")
try:
    ns_archive = NoveltyArchive(
        behavior_dim=2,
        max_size=100,
        novelty_threshold=0.05,
        add_prob=0.15,
        k_neighbors=5,
        seed=10,
    )
    ns_fitness = NoveltyFitness(ns_archive=ns_archive, alpha=0.5)

    rng2 = np.random.default_rng(99)

    def projected_factory(skill_type, rng):
        """射影版の小規模スキルを素早く生成するファクトリ。"""
        subset_idx = rng.choice(CORPUS_N, size=20, replace=False)
        sub_corpus  = corpus_32[subset_idx]
        sub_queries = queries_32[:5]

        if skill_type == "indexer":
            sk = NeatIndexer(input_dim=PROJ_DIM, pop_size=10, species_size=3,
                             generation_limit=3, k=3,
                             seed=int(rng.integers(0, 9999)))
            sk.fit(corpus=sub_corpus, queries=sub_queries)
            desc = sk.bcs_descriptor(sub_queries)
            fit  = float(-rng.uniform(0.05, 0.3))

        elif skill_type == "gate":
            n_sk = 4
            ctx  = sub_queries
            tgt  = (sub_queries[:, :n_sk] > 0).astype(np.float32)
            sk   = NeatGate(context_dim=PROJ_DIM, num_skills=n_sk,
                            pop_size=10, species_size=3, generation_limit=3,
                            seed=int(rng.integers(0, 9999)))
            sk.fit(contexts=ctx, targets=tgt)
            desc = sk.bcs_descriptor(ctx)
            fit  = float(-rng.uniform(0.3, 1.0))

        else:
            ctx  = sub_queries
            tgt  = np.tanh(sub_queries[:, :3])
            sk   = NeatSlotFiller(slot_names=KG_SCHEMA, context_dim=PROJ_DIM,
                                  pop_size=10, species_size=3, generation_limit=3,
                                  seed=int(rng.integers(0, 9999)))
            sk.fit(contexts=ctx, targets=tgt)
            desc = sk.bcs_descriptor(ctx, targets=tgt)
            fit  = float(-rng.uniform(0.3, 0.8))

        return sk, fit, desc

    loop = NoveltyEvolutionLoop(
        trident_archive=archive,
        ns_archive=ns_archive,
        skill_factory=projected_factory,
        novelty_fitness=ns_fitness,
        max_iterations=15,
        seed=42,
    )
    loop.run()

    assert ns_archive.size > 0
    assert archive.total_skills > 0
    print(f"    ✅ NS size={ns_archive.size}, "
          f"MAP total={archive.total_skills}, "
          f"mean_novelty={ns_archive.mean_novelty:.3f}")
except Exception as e:
    print(f"    ❌ {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# ─── HybridIndexer (neat モード) で 384 次元クエリを検索 ───
print("[9] HybridIndexer で 384 次元クエリ検索 ... ", end="", flush=True)
try:
    best_rec = archive.best_indexer()
    assert best_rec is not None

    hybrid = HybridIndexer(neat_indexer=best_rec.skill, mode="neat")

    # 384 → 32 射影してから検索
    q_32 = project(queries_384[3:4])[0]
    q_32 /= np.linalg.norm(q_32) + 1e-8
    h_idx, h_scores = hybrid.search(q_32, k=K)

    assert len(h_idx) == K
    print(f"✅  top-{K}={h_idx}, scores={np.round(h_scores, 3)}")
except Exception as e:
    print(f"❌ {e}")
    sys.exit(1)

# ─── NeatAugmentedReward (GRPO 統合確認) ───
print("[10] NeatAugmentedReward (384→32 射影版) ... ", end="", flush=True)
try:
    best_gate_rec = archive.best_gate()
    assert best_gate_rec is not None

    def mock_grpo_reward(ctx32, action):
        return float(np.dot(ctx32[:4], action[:4]))

    aug_reward = NeatAugmentedReward(
        neat_gate=best_gate_rec.skill,
        base_reward=mock_grpo_reward,
        gate_weight=0.3,
    )
    ctx_32 = project(queries_384[0:1])[0]
    ctx_32 /= np.linalg.norm(ctx_32) + 1e-8
    action  = np.ones(PROJ_DIM, dtype=np.float32) / np.sqrt(PROJ_DIM)

    reward = aug_reward(ctx_32, action)
    assert isinstance(reward, float)
    print(f"✅  augmented_reward={reward:.4f}")
except Exception as e:
    print(f"❌ {e}")
    sys.exit(1)

# ─── アーカイブ統計 ───
print()
archive.print_summary()

# ─── 最終サマリー ───
print("=" * 65)
print("本番次元 (384) 統合テスト サマリー")
print("=" * 65)
print("✅ 全 10 項目通過")
print()
print(f"  入力次元         : {DIM}  (MiniLM 相当)")
print(f"  射影次元 (NEAT)  : {PROJ_DIM}")
print(f"  射影行列         : {R.shape}  固定ランダム直交")
print(f"  コーパス         : {CORPUS_N} サンプル")
print()
print("  モジュール別結果:")
print(f"    A型 NeatIndexer  : BCS {np.round(desc_a, 3)}")
print(f"    B型 NeatGate     : BCS {np.round(desc_b, 3)}, acc={acc:.3f}")
print(f"    C型 NeatSlotFiller: BCS {np.round(desc_c, 3)}, fill_rate={fr:.3f}")
print(f"    NS アーカイブ    : size={ns_archive.size}, mean_novelty={ns_archive.mean_novelty:.3f}")
print(f"    MAP-Elites 総数  : {archive.total_skills} スキル")
print()
print("次のステップ:")
print("  - ES-HyperNEAT カスタム拡張 (幾何的基板で射影行列を進化)")
print("  - NS × 実スキルの長期ループ (100+ 世代)")
print("  - faiss-cpu インストールで HybridIndexer hybrid モード検証")
