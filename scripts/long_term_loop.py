"""
TRIDENT — 長期進化ループ検証 (120 イテレーション)
═══════════════════════════════════════════════════════════════

NS × MAP-Elites × 実スキルの長期動作を確認する。

【速度最適化】
  - スキルタイプごとに transform を 1 回だけ計算 (事前キャッシュ)
  - 評価は JAX vmap でバッチ化 (Python ループ廃止)
  - 各イテレーションは ~50ms 目標

【検証指標】(10 イテレーションごと)
  - NS アーカイブサイズ   : 多様性の蓄積
  - MAP-Elites カバレッジ : BCS 空間探索度
  - QD スコア             : 質 × 多様性の複合指標
  - 平均新規性スコア      : 探索フロンティア推移
"""

from __future__ import annotations

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import jax
import jax.numpy as jnp
from jax import vmap

print("=" * 65)
print("TRIDENT 長期進化ループ (120 イテレーション) 検証")
print("=" * 65)

# ─── 設定 ───
TOTAL_ITER    = 120
GENERATIONS   = 3
POP_SIZE      = 15
SPECIES_SIZE  = 3
DIM           = 384
PROJ_DIM      = 16
CORPUS_N      = 80
QUERY_N       = 20
EVAL_QUERY    = 10
K             = 3
NS_ALPHA      = 0.6
LOG_INTERVAL  = 10
N_SKILLS_GATE = 10   # B型スキル数 = context_dim

# ─── モジュールインポート ───
print("\n[1] モジュールインポート ...", end=" ", flush=True)
try:
    from src.interfaces.neat_indexer     import NeatIndexer
    from src.interfaces.neat_gate        import NeatGate
    from src.interfaces.neat_slot_filler import NeatSlotFiller, GRPO_SCHEMA
    from src.map_elites_archive          import TRIDENTArchive
    from src.novelty_search              import NoveltyArchive, NoveltyFitness
    print("✅")
except Exception as e:
    print(f"❌ {e}"); import traceback; traceback.print_exc(); sys.exit(1)

# ─── 合成データ生成 ───
print("[2] 合成データ生成 ...", end=" ", flush=True)
try:
    master_rng = np.random.default_rng(42)

    corpus_384  = master_rng.standard_normal((CORPUS_N, DIM)).astype(np.float32)
    corpus_384 /= np.linalg.norm(corpus_384, axis=1, keepdims=True) + 1e-8

    query_pool   = master_rng.standard_normal((QUERY_N * 6, DIM)).astype(np.float32)
    query_pool  /= np.linalg.norm(query_pool, axis=1, keepdims=True) + 1e-8

    R = master_rng.standard_normal((DIM, PROJ_DIM)).astype(np.float32)
    R /= np.linalg.norm(R, axis=0, keepdims=True) + 1e-8

    def sketch(x):
        out = x @ R
        return out / (np.linalg.norm(out, axis=-1, keepdims=True) + 1e-8)

    corpus_p     = sketch(corpus_384)       # (C, PROJ_DIM)
    query_pool_p = sketch(query_pool)       # (Q*6, PROJ_DIM)

    # B型: context_dim = N_SKILLS_GATE = 10
    q_ctx = N_SKILLS_GATE
    gate_pool_ctxs = query_pool_p[:, :q_ctx]          # (pool, 10)
    gate_pool_lbls = (
        query_pool_p @ corpus_p[:N_SKILLS_GATE].T > 0.0
    ).astype(np.float32)                               # (pool, 10)

    # C型スロットターゲット
    n_slots = len(GRPO_SCHEMA)
    sf_target_pool = np.clip(
        master_rng.standard_normal((QUERY_N * 6, n_slots)).astype(np.float32),
        -1, 1,
    )

    # JAX 配列に変換
    corpus_j  = jnp.array(corpus_p)
    qpool_j   = jnp.array(query_pool_p)

    print(f"✅  corpus={corpus_p.shape}, pool={query_pool_p.shape}")
except Exception as e:
    print(f"❌ {e}"); sys.exit(1)

# ─── Phase A: スキル事前学習 ───
print("\n[3] スキル事前学習 (3 回コンパイル) ...")

t0 = time.time()

# A型
print("  [A型 NeatIndexer] ...", end=" ", flush=True)
base_indexer = NeatIndexer(
    input_dim=PROJ_DIM, k=K,
    pop_size=POP_SIZE, species_size=SPECIES_SIZE,
    generation_limit=GENERATIONS, seed=0,
)
base_indexer.fit(corpus_p, query_pool_p[:EVAL_QUERY])
print(f"✅ ({time.time()-t0:.1f}s)")

# B型
t1 = time.time()
print("  [B型 NeatGate]    ...", end=" ", flush=True)
base_gate = NeatGate(
    context_dim=q_ctx, num_skills=N_SKILLS_GATE,
    pop_size=POP_SIZE, species_size=SPECIES_SIZE,
    generation_limit=GENERATIONS, seed=1,
)
base_gate.fit(
    gate_pool_ctxs[:EVAL_QUERY],
    gate_pool_lbls[:EVAL_QUERY],
)
print(f"✅ ({time.time()-t1:.1f}s)")

# C型
t2 = time.time()
print("  [C型 NeatSlotFiller] ...", end=" ", flush=True)
base_filler = NeatSlotFiller(
    slot_names=GRPO_SCHEMA,
    context_dim=PROJ_DIM,
    pop_size=POP_SIZE, species_size=SPECIES_SIZE,
    generation_limit=GENERATIONS, seed=2,
)
base_filler.fit(query_pool_p[:EVAL_QUERY], sf_target_pool[:EVAL_QUERY])
print(f"✅ ({time.time()-t2:.1f}s)")

t_compile = time.time() - t0
print(f"  コンパイル合計: {t_compile:.1f}s")

# ─── Phase B: バッチ評価関数の構築 (transform は 1 回だけ) ───
print("\n[4] バッチ評価関数構築 ...", end=" ", flush=True)
try:
    algo_i = base_indexer._pipeline.algorithm
    algo_g = base_gate._pipeline.algorithm
    algo_s = base_filler._pipeline.algorithm

    st_i = base_indexer._state
    st_g = base_gate._state
    st_s = base_filler._state

    best_i = base_indexer._best_params
    best_g = base_gate._best_params
    best_s = base_filler._best_params

    # transform は 1 回だけ (全イテレーション共通)
    bt_i = algo_i.transform(st_i, best_i)   # A型 transformed params
    bt_g = algo_g.transform(st_g, best_g)   # B型
    bt_s = algo_s.transform(st_s, best_s)   # C型

    # vmap 化バッチ forward 関数 (JIT)
    @jax.jit
    def fwd_indexer_batch(queries: jnp.ndarray) -> jnp.ndarray:
        """(Q, D) → (Q, D)"""
        return vmap(lambda q: algo_i.forward(st_i, bt_i, q))(queries)

    @jax.jit
    def fwd_gate_batch(contexts: jnp.ndarray) -> jnp.ndarray:
        """(N, ctx_dim) → (N, n_skills)"""
        return vmap(lambda c: algo_g.forward(st_g, bt_g, c))(contexts)

    @jax.jit
    def fwd_filler_batch(contexts: jnp.ndarray) -> jnp.ndarray:
        """(N, PROJ_DIM) → (N, n_slots)"""
        return vmap(lambda c: algo_s.forward(st_s, bt_s, c))(contexts)

    # ウォームアップ (JIT コンパイル)
    _ = fwd_indexer_batch(qpool_j[:EVAL_QUERY])
    _ = fwd_gate_batch(jnp.array(gate_pool_ctxs[:EVAL_QUERY]))
    _ = fwd_filler_batch(qpool_j[:EVAL_QUERY])

    print("✅")
except Exception as e:
    print(f"❌ {e}"); import traceback; traceback.print_exc(); sys.exit(1)

# ─── 高速評価関数 ───

def eval_indexer_fast(seed: int):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(query_pool_p), EVAL_QUERY, replace=False)
    q_j = qpool_j[idx]                      # (Q, PROJ_DIM)

    outs = np.array(fwd_indexer_batch(q_j))  # (Q, PROJ_DIM)

    # Recall@k
    gt = np.argsort(-(q_j @ corpus_j.T), axis=1)[:, :K]    # (Q, k)
    pred_sims = jnp.array(outs) @ corpus_j.T
    top_k_pred = np.argsort(-np.array(pred_sims), axis=1)[:, :K]
    hit = sum(
        len(set(top_k_pred[qi].tolist()) & set(np.array(gt[qi]).tolist()))
        for qi in range(EVAL_QUERY)
    )
    recall_k = float(hit) / (EVAL_QUERY * K)

    # 正規化コサイン類似度
    norms = np.linalg.norm(outs, axis=1, keepdims=True) + 1e-8
    cos_sim = float(np.mean(np.sum(outs / norms * np.array(q_j), axis=1)))

    # BCS
    act_intensity = float(np.mean(np.abs(outs)))
    sig = 1.0 / (1.0 + np.exp(-act_intensity))
    bcs = np.array([
        float(np.clip(sig, 0.0, 1.0)),
        float(np.clip(recall_k, 0.0, 1.0)),
    ], dtype=np.float32)

    return base_indexer, cos_sim, bcs


def eval_gate_fast(seed: int):
    rng = np.random.default_rng(seed)
    pool_n = len(gate_pool_ctxs)
    idx = rng.choice(pool_n, EVAL_QUERY, replace=False)
    ctx_j = jnp.array(gate_pool_ctxs[idx])  # (N, q_ctx)
    lbl   = gate_pool_lbls[idx]              # (N, n_skills)

    logits = np.array(fwd_gate_batch(ctx_j))  # (N, n_skills)
    probs  = 1.0 / (1.0 + np.exp(-logits))

    preds = (probs > 0.5).astype(np.float32)
    acc   = float(np.mean(preds == lbl))
    fitness = acc - 1.0  # [0,1] → [-1, 0]

    # BCS
    firing_rate  = float(np.mean(preds))
    specificity  = float(np.mean(np.abs(probs - 0.5)) * 2.0)  # [0,1]
    bcs = np.array([
        float(np.clip(firing_rate, 0.0, 1.0)),
        float(np.clip(specificity, 0.0, 1.0)),
    ], dtype=np.float32)

    return base_gate, fitness, bcs


def eval_filler_fast(seed: int):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(query_pool_p), EVAL_QUERY, replace=False)
    ctx_j = qpool_j[idx]
    tgt   = sf_target_pool[idx]

    raw = np.array(fwd_filler_batch(ctx_j))     # (N, n_slots)
    preds = np.tanh(raw)
    fitness = -float(np.mean((preds - tgt) ** 2))

    fill_rate = float(np.mean(np.abs(preds) > base_filler.fill_threshold))
    accuracy  = float(np.clip(fitness + 1.0, 0.0, 1.0))
    bcs = np.array([
        float(np.clip(fill_rate, 0.0, 1.0)),
        accuracy,
    ], dtype=np.float32)

    return base_filler, fitness, bcs


_eval_funcs = {
    "indexer":     eval_indexer_fast,
    "gate":        eval_gate_fast,
    "slot_filler": eval_filler_fast,
}

# ─── アーカイブ・NS 初期化 ───
print("[5] アーカイブ・NS 初期化 ...", end=" ", flush=True)
trident = TRIDENTArchive(grid_sizes={"indexer": 16, "gate": 8, "slot_filler": 8})
ns_archive = NoveltyArchive(
    behavior_dim=2, max_size=300,
    novelty_threshold=0.05, k_neighbors=5, seed=0,
)
ns_fitness = NoveltyFitness(ns_archive, alpha=NS_ALPHA)
print("✅")

# ─── 長期進化ループ ───
print(f"\n[6] 長期進化ループ ({TOTAL_ITER} イテレーション)")
print(f"    評価データ変更 → vmap BCS 計算 → NS+MAP-Elites 更新")
print(f"    NS_alpha={NS_ALPHA}, k_nn={ns_archive.k_neighbors}")
print("-" * 65)

skill_types    = ["indexer", "gate", "slot_filler"]
history        = []
ns_added_total = 0
map_adopted_total = 0
t_loop_start   = time.time()

for i in range(TOTAL_ITER):
    stype = skill_types[i % len(skill_types)]
    seed  = i * 137 + 1001

    t_iter = time.time()
    try:
        skill, task_fit, bcs = _eval_funcs[stype](seed)
    except Exception as e:
        print(f"    [iter {i:3d}] ⚠ {stype}: {e}", flush=True)
        continue

    t_ms = (time.time() - t_iter) * 1000

    behaviors = ns_archive.behaviors_array
    combined  = ns_fitness(bcs, task_fit, extra_pool=behaviors)
    added, novelty = ns_archive.try_add(
        behavior=bcs, skill_type=stype, skill=skill,
        task_fitness=task_fit, metadata={"iter": i},
    )
    if added:   ns_added_total += 1

    if   stype == "indexer":
        adopted = trident.add_indexer(skill, task_fit, bcs, {"iter": i})
    elif stype == "gate":
        adopted = trident.add_gate(skill, task_fit, bcs, {"iter": i})
    else:
        adopted = trident.add_slot_filler(skill, task_fit, bcs, {"iter": i})
    if adopted: map_adopted_total += 1

    ri = trident.repertoires["indexer"]
    rg = trident.repertoires["gate"]
    rs = trident.repertoires["slot_filler"]

    history.append({
        "iter": i, "stype": stype, "task_fit": task_fit,
        "novelty": novelty, "combined": combined,
        "ns_size": ns_archive.size,
        "cov_i": ri.coverage, "cov_g": rg.coverage, "cov_s": rs.coverage,
        "qd_i": ri.qd_score, "qd_g": rg.qd_score, "qd_s": rs.qd_score,
        "t_ms": t_ms,
    })

    if (i + 1) % LOG_INTERVAL == 0:
        tot_cov = (ri.coverage + rg.coverage + rs.coverage) / 3
        tot_qd  = ri.qd_score + rg.qd_score + rs.qd_score
        elapsed = time.time() - t_loop_start
        print(
            f"  iter {i+1:3d}/{TOTAL_ITER}  "
            f"NS={ns_archive.size:3d}  MAP={trident.total_skills:3d}  "
            f"cov={tot_cov:.1%}  QD={tot_qd:.3f}  "
            f"nov={novelty:.3f}  t={t_ms:.0f}ms  [{elapsed:.1f}s]",
            flush=True,
        )

print("-" * 65)
t_loop = time.time() - t_loop_start

# ─── 最終レポート ───
print(f"\n[7] 最終レポート")
print("=" * 65)
n_done = max(len(history), 1)
t_ms_mean = float(np.mean([h["t_ms"] for h in history])) if history else 0
print(f"  コンパイル時間    : {t_compile:.1f}s")
print(f"  ループ時間        : {t_loop:.1f}s ({t_ms_mean:.0f}ms/iter)")
print(f"  総イテレーション  : {len(history)}/{TOTAL_ITER}")

print("\n  ── MAP-Elites ──")
for stype, rep in trident.repertoires.items():
    label = {"indexer":"A型 NeatIndexer","gate":"B型 NeatGate",
             "slot_filler":"C型 NeatSlotFiller"}[stype]
    print(f"  [{label}] {rep.filled_cells}/{rep.num_cells} ({rep.coverage:.1%})  "
          f"best={rep.best_fitness}  QD={rep.qd_score:.4f}")

print(f"\n  ── Novelty Search ──")
print(f"  size={ns_archive.size}  mean_nov={ns_archive.mean_novelty:.4f}  "
      f"max_nov={ns_archive.max_novelty:.4f}")
print(f"  NS 採用率: {ns_added_total}/{n_done} ({ns_added_total/n_done:.1%})  "
      f"MAP 採用率: {map_adopted_total}/{n_done} ({map_adopted_total/n_done:.1%})")

# ─── 推移テーブル ───
print(f"\n[8] 推移テーブル")
print(f"  {'iter':>5}  {'NS':>4}  {'MAP':>4}  "
      f"{'cov_A':>6}  {'cov_B':>6}  {'cov_C':>6}  "
      f"{'QD':>7}  {'nov':>6}  {'ms':>6}")
print("  " + "-" * 62)
for ci in range(LOG_INTERVAL - 1, len(history), LOG_INTERVAL):
    h = history[ci]
    qd = h["qd_i"] + h["qd_g"] + h["qd_s"]
    w  = history[max(0, ci - LOG_INTERVAL + 1): ci + 1]
    mn = float(np.mean([x["novelty"] for x in w]))
    mt = float(np.mean([x["t_ms"] for x in w]))
    print(f"  {h['iter']+1:5d}  {h['ns_size']:4d}  {trident.total_skills:4d}  "
          f"{h['cov_i']:6.1%}  {h['cov_g']:6.1%}  {h['cov_s']:6.1%}  "
          f"{qd:7.3f}  {mn:6.4f}  {mt:6.0f}")

# ─── 検証チェック ───
print(f"\n[9] 検証チェック")
checks, passed = 0, 0

def chk(ok, msg_ok, msg_fail):
    global checks, passed
    checks += 1
    if ok:
        passed += 1
        print(f"  ✅ {msg_ok}")
    else:
        print(f"  ⚠  {msg_fail}")

if len(history) >= 20:
    chk(history[-1]["ns_size"] > history[9]["ns_size"],
        f"NS 成長: {history[9]['ns_size']} → {history[-1]['ns_size']}",
        f"NS 停滞: {history[9]['ns_size']} → {history[-1]['ns_size']}")

if len(history) >= 10:
    fc = history[9]["cov_i"] + history[9]["cov_g"] + history[9]["cov_s"]
    lc = history[-1]["cov_i"] + history[-1]["cov_g"] + history[-1]["cov_s"]
    chk(lc > fc,
        f"MAP カバレッジ成長: {fc/3:.1%} → {lc/3:.1%}",
        f"MAP カバレッジ停滞: {fc/3:.1%} → {lc/3:.1%}")

fqd = history[-1]["qd_i"] + history[-1]["qd_g"] + history[-1]["qd_s"] if history else 0
chk(fqd > 0, f"QD スコア > 0: {fqd:.4f}", f"QD スコア ≤ 0: {fqd:.4f}")
chk(ns_archive.mean_novelty > 0.01,
    f"平均新規性 > 0.01: {ns_archive.mean_novelty:.4f}",
    f"平均新規性が低い: {ns_archive.mean_novelty:.4f}")
chk(t_ms_mean < 5000,
    f"iter 速度 {t_ms_mean:.0f}ms < 5000ms",
    f"iter 遅延 {t_ms_mean:.0f}ms")

print()
print("=" * 65)
msg = "✅ 長期進化ループ検証完了" if passed == checks else f"⚠  {passed}/{checks} チェック通過"
print(f"{msg} ({passed}/{checks})")
print("=" * 65)
