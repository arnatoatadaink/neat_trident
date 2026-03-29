"""
Phase 1 動作確認スクリプト
NeatIndexer (A型 I/F) の基本動作を小規模データで検証する。

入力次元 = 16 (プロトタイプ; 本番は 384)
コーパス = 50 ランダムベクトル
クエリ   = 10 ランダムベクトル
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

print("=" * 60)
print("Phase 1 — NeatIndexer (A型 I/F) 動作確認")
print("=" * 60)

# ─── 合成データ作成 ───
rng = np.random.default_rng(0)
DIM      = 16    # プロトタイプ次元 (本番は 384)
CORPUS_N = 50
QUERY_N  = 10
K        = 3

corpus  = rng.standard_normal((CORPUS_N, DIM)).astype(np.float32)
queries = rng.standard_normal((QUERY_N, DIM)).astype(np.float32)

# ground truth: コサイン類似度で上位K件
corpus_n = corpus / (np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-8)
queries_n = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-8)
true_scores = queries_n @ corpus_n.T          # (Q, C)
true_neighbors = np.argsort(-true_scores, axis=1)[:, :K]  # (Q, K)

print(f"\nデータ:")
print(f"  corpus  : {corpus.shape}")
print(f"  queries : {queries.shape}")
print(f"  k       : {K}")

# ─── NeatIndexer インポート確認 ───
print("\n[1] NeatIndexer インポート確認 ... ", end="", flush=True)
try:
    from src.interfaces.neat_indexer import (
        NeatIndexer,
        HybridIndexer,
        VectorNeighborProblem,
        compute_bcs_descriptor,
    )
    print("✅")
except Exception as e:
    print(f"❌ {e}")
    sys.exit(1)

# ─── VectorNeighborProblem ───
print("[2] VectorNeighborProblem 構築 ... ", end="", flush=True)
try:
    problem = VectorNeighborProblem(queries=queries, corpus=corpus, k=K)
    assert problem.input_shape == (DIM,)
    assert problem.output_shape == (DIM,)
    print("✅")
except Exception as e:
    print(f"❌ {e}")
    sys.exit(1)

# ─── NeatIndexer 構築 ───
print("[3] NeatIndexer 構築 ... ", end="", flush=True)
try:
    indexer = NeatIndexer(
        input_dim=DIM,
        pop_size=20,
        species_size=3,
        max_nodes=30,
        max_conns=60,
        generation_limit=5,   # 確認用に短め
        k=K,
        seed=42,
    )
    print("✅")
except Exception as e:
    print(f"❌ {e}")
    sys.exit(1)

# ─── fit() 実行 ───
print("[4] NeatIndexer.fit() 進化 (5世代) ...")
try:
    indexer.fit(corpus=corpus, queries=queries)
    print("    ✅ fit 完了")
except Exception as e:
    print(f"    ❌ {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# ─── transform() ───
print("[5] transform() 動作確認 ... ", end="", flush=True)
try:
    out = indexer.transform(queries[0])
    assert out.shape == (DIM,), f"shape mismatch: {out.shape}"
    print(f"✅  output shape={out.shape}")
except Exception as e:
    print(f"❌ {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# ─── search() ───
print("[6] search() 動作確認 ... ", end="", flush=True)
try:
    idx, scores = indexer.search(queries[0], k=K)
    assert len(idx) == K
    print(f"✅  indices={idx}, scores={np.round(scores, 3)}")
except Exception as e:
    print(f"❌ {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# ─── BCS descriptor ───
print("[7] bcs_descriptor() 確認 ... ", end="", flush=True)
try:
    desc = indexer.bcs_descriptor(queries, true_neighbors=true_neighbors)
    assert desc.shape == (2,)
    assert 0.0 <= desc[0] <= 1.0 and 0.0 <= desc[1] <= 1.0, f"範囲外: {desc}"
    print(f"✅  descriptor={np.round(desc, 3)}")
except Exception as e:
    print(f"❌ {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# ─── HybridIndexer (neat モード) ───
print("[8] HybridIndexer(mode='neat') 確認 ... ", end="", flush=True)
try:
    hybrid = HybridIndexer(neat_indexer=indexer, mode="neat")
    h_idx, h_scores = hybrid.search(queries[0], k=K)
    assert len(h_idx) == K
    print(f"✅  indices={h_idx}")
except Exception as e:
    print(f"❌ {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# ─── サマリー ───
print("\n" + "=" * 60)
print("Phase 1 動作確認 サマリー")
print("=" * 60)
print("✅ NeatIndexer (A型 I/F) 基本動作確認完了")
print(f"   入力次元    : {DIM}  (本番: 384)")
print(f"   コーパス数  : {CORPUS_N}")
print(f"   BCS 記述子  : {np.round(desc, 3)}")
print()
print("次のステップ:")
print("  - Phase 2: B型 NeatGate 実装")
print("  - 本番次元 (384) での動作確認")
print("  - MAP-Elites アーカイブ (QDax) 統合")
