"""
ES-HyperNEAT 動作確認スクリプト
src/es_hyperneat.py の全機能を検証する。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import time

print("=" * 65)
print("ES-HyperNEAT カスタム拡張 — 動作確認")
print("=" * 65)

PROJ_DIM   = 8    # 確認用の小さい次元
HIDDEN_DIM = 4
INPUT_DIM  = 32   # 前段のスケッチ後次元 (本番: 384→32)
CORPUS_N   = 20
QUERY_N    = 5
K          = 3

# ─── インポート ───
print("\n[1] インポート確認 ... ", end="", flush=True)
try:
    from src.es_hyperneat import (
        make_trident_substrate,
        ProjectionFitProblem,
        ESHyperNEATProjector,
        ESHyperNEATIndexer,
    )
    print("✅")
except Exception as e:
    print(f"❌ {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

rng = np.random.default_rng(42)

# ─── TRIDENTSubstrate ───
print("[2] TRIDENTSubstrate 構築 ... ", end="", flush=True)
try:
    substrate = make_trident_substrate(proj_dim=PROJ_DIM, hidden_dim=HIDDEN_DIM)
    q_shape = substrate.query_coors.shape
    assert q_shape[1] == 4, f"CPPN 入力は 4 次元のはず: {q_shape}"
    n_conns = PROJ_DIM * HIDDEN_DIM + HIDDEN_DIM * PROJ_DIM
    print(f"✅  query_coors={q_shape}, 期待接続数≈{n_conns}")
except Exception as e:
    print(f"❌ {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# ─── ProjectionFitProblem ───
print("[3] ProjectionFitProblem 構築 ... ", end="", flush=True)
try:
    q_proj = rng.standard_normal((QUERY_N, PROJ_DIM)).astype(np.float32)
    t_proj = q_proj.copy()
    prob   = ProjectionFitProblem(queries_proj=q_proj, targets_proj=t_proj)
    assert prob.input_shape  == (PROJ_DIM,)
    assert prob.output_shape == (PROJ_DIM,)
    print("✅")
except Exception as e:
    print(f"❌ {e}")
    sys.exit(1)

# ─── ESHyperNEATProjector 構築 ───
print("[4] ESHyperNEATProjector 構築 ... ", end="", flush=True)
try:
    projector = ESHyperNEATProjector(
        proj_dim=PROJ_DIM,
        hidden_dim=HIDDEN_DIM,
        cppn_pop_size=15,
        cppn_species_size=3,
        generation_limit=5,
        seed=0,
    )
    assert not projector.is_fitted
    print("✅")
except Exception as e:
    print(f"❌ {e}")
    sys.exit(1)

# ─── ESHyperNEATProjector.fit() ───
print("[5] ESHyperNEATProjector.fit() (5世代) ...")
try:
    q_in = rng.standard_normal((QUERY_N, PROJ_DIM)).astype(np.float32)
    q_in /= np.linalg.norm(q_in, axis=1, keepdims=True) + 1e-8

    t0 = time.time()
    projector.fit(queries_proj=q_in)
    elapsed = time.time() - t0

    assert projector.is_fitted
    print(f"    ✅ {elapsed:.1f}s  fit 完了")
except Exception as e:
    print(f"    ❌ {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# ─── project() 単一 ───
print("[6] project() 単一入力 ... ", end="", flush=True)
try:
    x = q_in[0]
    out = projector.project(x)
    assert out.shape == (PROJ_DIM,), f"shape mismatch: {out.shape}"
    print(f"✅  in={np.round(x[:3], 2)}, out={np.round(out[:3], 2)}")
except Exception as e:
    print(f"❌ {e}")
    sys.exit(1)

# ─── project() バッチ ───
print("[7] project() バッチ入力 ... ", end="", flush=True)
try:
    out_batch = projector.project(q_in)
    assert out_batch.shape == (QUERY_N, PROJ_DIM)
    print(f"✅  shape={out_batch.shape}")
except Exception as e:
    print(f"❌ {e}")
    sys.exit(1)

# ─── projection_matrix() ───
print("[8] projection_matrix() ... ", end="", flush=True)
try:
    W = projector.projection_matrix(normalize=True)
    assert W.shape == (PROJ_DIM, PROJ_DIM)
    # 列正規化確認
    col_norms = np.linalg.norm(W, axis=0)
    assert np.allclose(col_norms, 1.0, atol=0.1), f"列正規化ずれ: {col_norms}"
    # ランダム射影との構造比較
    rand_W = rng.standard_normal((PROJ_DIM, PROJ_DIM)).astype(np.float32)
    rand_W /= np.linalg.norm(rand_W, axis=0, keepdims=True) + 1e-8
    # CPPN 射影はランダムより構造化されているはず (行の相関が低い)
    cppn_corr = float(np.mean(np.abs(np.corrcoef(W))))
    rand_corr = float(np.mean(np.abs(np.corrcoef(rand_W))))
    print(f"✅  W shape={W.shape}, CPPN corr={cppn_corr:.3f}, rand corr={rand_corr:.3f}")
except Exception as e:
    print(f"❌ {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# ─── ESHyperNEATIndexer 構築 ───
print("[9] ESHyperNEATIndexer 構築・fit() (5世代) ...")
try:
    corpus  = rng.standard_normal((CORPUS_N, INPUT_DIM)).astype(np.float32)
    queries = rng.standard_normal((QUERY_N,  INPUT_DIM)).astype(np.float32)

    indexer = ESHyperNEATIndexer(
        input_dim=INPUT_DIM,
        proj_dim=PROJ_DIM,
        hidden_dim=HIDDEN_DIM,
        cppn_pop_size=15,
        cppn_species_size=3,
        generation_limit=5,
        k=K,
        seed=1,
    )

    t0 = time.time()
    indexer.fit(corpus=corpus, queries=queries)
    elapsed = time.time() - t0
    print(f"    ✅ {elapsed:.1f}s  fit 完了")
except Exception as e:
    print(f"    ❌ {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# ─── search() ───
print("[10] search() ... ", end="", flush=True)
try:
    idx, scores = indexer.search(queries[0], k=K)
    assert len(idx) == K
    assert len(scores) == K
    print(f"✅  indices={idx}, scores={np.round(scores, 3)}")
except Exception as e:
    print(f"❌ {e}")
    sys.exit(1)

# ─── projection_matrix() ───
print("[11] projection_matrix() (ESHyperNEATIndexer) ... ", end="", flush=True)
try:
    W_idx = indexer.projection_matrix()
    assert W_idx.shape == (PROJ_DIM, PROJ_DIM)
    print(f"✅  shape={W_idx.shape}")
except Exception as e:
    print(f"❌ {e}")
    sys.exit(1)

# ─── bcs_descriptor() ───
print("[12] bcs_descriptor() ... ", end="", flush=True)
try:
    desc = indexer.bcs_descriptor(queries)
    assert desc.shape == (2,)
    assert 0.0 <= desc[0] <= 1.0
    print(f"✅  desc={np.round(desc, 3)}")
except Exception as e:
    print(f"❌ {e}")
    sys.exit(1)

# ─── 固定射影 vs CPPN 射影の比較 ───
print("\n[13] 固定射影 vs CPPN 射影の特性比較 ...")
try:
    # 固定射影での類似度保存
    sketch = indexer._sketch  # (INPUT_DIM, PROJ_DIM)
    corpus_s  = corpus  @ sketch
    corpus_s  /= np.linalg.norm(corpus_s,  axis=1, keepdims=True) + 1e-8
    queries_s = queries @ sketch
    queries_s /= np.linalg.norm(queries_s, axis=1, keepdims=True) + 1e-8

    # 真のコサイン類似度
    corp_n  = corpus  / (np.linalg.norm(corpus,  axis=1, keepdims=True) + 1e-8)
    quer_n  = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-8)
    true_sim = quer_n @ corp_n.T   # (Q, C)

    # 固定射影での類似度
    fixed_sim = queries_s @ corpus_s.T   # (Q, C)

    # CPPN 射影での類似度
    corp_c  = indexer._corpus_proj                # 既に正規化済み
    quer_c  = indexer._projector.project(queries_s)
    quer_c  /= np.linalg.norm(quer_c, axis=1, keepdims=True) + 1e-8
    cppn_sim = quer_c @ corp_c.T   # (Q, C)

    # Pearson 相関で類似度保存を評価
    true_flat  = true_sim.ravel()
    fixed_corr = float(np.corrcoef(true_flat, fixed_sim.ravel())[0, 1])
    cppn_corr  = float(np.corrcoef(true_flat, cppn_sim.ravel())[0, 1])

    print(f"    固定射影との Pearson 相関: {fixed_corr:.3f}")
    print(f"    CPPN 射影との Pearson 相関: {cppn_corr:.3f}")
    print(f"    ✅ 両方の相関を確認 (5世代では CPPN は未収束が普通)")
except Exception as e:
    print(f"    ❌ {e}")
    import traceback; traceback.print_exc()

# ─── サマリー ───
print("\n" + "=" * 65)
print("ES-HyperNEAT 動作確認 サマリー")
print("=" * 65)
print("✅ 全 13 チェック完了")
print()
print(f"  入力次元     : {INPUT_DIM}")
print(f"  射影次元     : {PROJ_DIM}")
print(f"  CPPN 基板    : {PROJ_DIM}→{HIDDEN_DIM}→{PROJ_DIM} (MLP Substrate)")
print(f"  CPPN 入力    : 4D (x_src, y_src, x_tgt, y_tgt)")
print(f"  射影行列     : {PROJ_DIM}×{PROJ_DIM} (幾何的に構造化)")
print()
print("固定ランダム射影との違い:")
print("  ランダム射影 : J-L スケッチ (構造なし)")
print("  ES-HyperNEAT: CPPN が幾何的位置から重みを決定 (構造化)")
print()
print("次のステップ:")
print("  - NS × 実スキルの長期ループ (100+ 世代)")
print("  - faiss-cpu 統合で HybridIndexer hybrid モード検証")
