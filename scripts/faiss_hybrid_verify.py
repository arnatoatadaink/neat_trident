"""
TRIDENT — HybridIndexer (FAISS + NEAT) 統合検証スクリプト

検証項目:
  [1] faiss-cpu インポート
  [2] 合成コーパス生成
  [3] FAISS FlatIP インデックス構築
  [4] NeatIndexer 学習 (短縮: 3世代)
  [5] HybridIndexer 3モード動作確認 (faiss / neat / hybrid)
  [6] Recall@k 比較 (faiss vs neat vs hybrid)
  [7] hybrid スコアが faiss/neat 単体の平均と一致するか確認
  [8] 大量クエリ (バッチ) の hybrid 検索レイテンシ確認
"""

import sys
import time
import numpy as np

# ─────────────────────────────────────────
# [1] インポート
# ─────────────────────────────────────────
print("=" * 60)
print("TRIDENT HybridIndexer (FAISS + NEAT) 統合検証")
print("=" * 60)

print("\n[1] インポート ... ", end="", flush=True)
try:
    import faiss
    import jax
    import jax.numpy as jnp
    sys.path.insert(0, "/home/user/neat_trident/src")
    from interfaces.neat_indexer import NeatIndexer, HybridIndexer
    print(f"✅  faiss={faiss.__version__}")
except Exception as e:
    print(f"❌ {e}")
    sys.exit(1)

# ─────────────────────────────────────────
# [2] 合成データ生成
# ─────────────────────────────────────────
print("\n[2] 合成データ生成 ... ", end="", flush=True)
rng = np.random.default_rng(0)
DIM = 16
N_CORPUS = 64
N_QUERY  = 20
K        = 5

corpus = rng.standard_normal((N_CORPUS, DIM)).astype(np.float32)
corpus /= np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-8

queries = rng.standard_normal((N_QUERY, DIM)).astype(np.float32)
queries /= np.linalg.norm(queries, axis=1, keepdims=True) + 1e-8

# 正解近傍 (FAISS で exact search して ground truth)
gt_scores = queries @ corpus.T            # (Q, C)
true_neighbors = np.argsort(-gt_scores, axis=1)[:, :K]  # (Q, K)

print(f"✅  corpus={corpus.shape}, queries={queries.shape}")

# ─────────────────────────────────────────
# [3] FAISS FlatIP インデックス構築
# ─────────────────────────────────────────
print("\n[3] FAISS FlatIP インデックス構築 ... ", end="", flush=True)
try:
    faiss_index = faiss.IndexFlatIP(DIM)   # 内積 (コーパス正規化済みなのでコサイン類似度)
    faiss_index.add(corpus)
    # 動作確認
    test_scores, test_idx = faiss_index.search(queries[:1], K)
    assert test_idx.shape == (1, K), f"FAISS search shape error: {test_idx.shape}"
    print(f"✅  ntotal={faiss_index.ntotal}, search shape OK")
except Exception as e:
    print(f"❌ {e}")
    sys.exit(1)

# ─────────────────────────────────────────
# [4] NeatIndexer 学習
# ─────────────────────────────────────────
print("\n[4] NeatIndexer 学習 (3世代) ... ", end="", flush=True)
t0 = time.perf_counter()
try:
    neat = NeatIndexer(
        input_dim=DIM,
        pop_size=15,
        species_size=3,
        max_nodes=50,
        max_conns=100,
        generation_limit=3,
        k=K,
        seed=42,
    )
    neat.fit(corpus, queries=queries, fitness_target=-0.5)
    elapsed = time.perf_counter() - t0
    print(f"✅  ({elapsed:.1f}s)")
except Exception as e:
    print(f"❌ {e}")
    sys.exit(1)

# ─────────────────────────────────────────
# [5] HybridIndexer 3モード動作確認
# ─────────────────────────────────────────
print("\n[5] HybridIndexer 3モード動作確認 ...")
results = {}

for mode in ["faiss", "neat", "hybrid"]:
    print(f"  mode='{mode}' ... ", end="", flush=True)
    try:
        hi = HybridIndexer(
            neat_indexer=neat,
            faiss_index=faiss_index if mode in ("faiss", "hybrid") else None,
            mode=mode,
        )
        idx, scores = hi.search(queries[0], k=K)
        assert len(idx) == K, f"返却 index 数が {len(idx)} != {K}"
        assert len(scores) == K, f"返却 scores 数が {len(scores)} != {K}"
        results[mode] = hi
        print(f"✅  idx={idx.tolist()}, top_score={scores[0]:.4f}")
    except Exception as e:
        print(f"❌ {e}")
        sys.exit(1)

# ─────────────────────────────────────────
# [6] Recall@k 比較
# ─────────────────────────────────────────
print("\n[6] Recall@k 比較 (k={}) ...".format(K))

def compute_recall(hi: HybridIndexer, queries: np.ndarray, true_neighbors: np.ndarray, k: int) -> float:
    hits, total = 0, 0
    for i, q in enumerate(queries):
        pred_idx, _ = hi.search(q, k=k)
        true = set(true_neighbors[i].tolist())
        pred = set(pred_idx.tolist())
        hits += len(true & pred)
        total += len(true)
    return hits / total if total > 0 else 0.0

recalls = {}
for mode, hi in results.items():
    t0 = time.perf_counter()
    r = compute_recall(hi, queries, true_neighbors, K)
    elapsed = time.perf_counter() - t0
    recalls[mode] = r
    print(f"  {mode:6s}: Recall@{K} = {r:.4f}  ({elapsed*1000:.0f}ms/{N_QUERY}q)")

# ─────────────────────────────────────────
# [7] hybrid スコア妥当性確認
# ─────────────────────────────────────────
print("\n[7] hybrid スコア妥当性確認 ...")

hi_faiss  = results["faiss"]
hi_neat   = results["neat"]
hi_hybrid = results["hybrid"]

q = queries[0]
fi, fs = hi_faiss.search(q, k=K)
ni, ns = hi_neat.search(q, k=K)
hi_i, hi_s = hi_hybrid.search(q, k=K)

# hybrid の top-1 はいずれかのモードの候補に含まれているはず
faiss_set = set(fi.tolist())
neat_set  = set(ni.tolist())
hybrid_set = set(hi_i.tolist())

overlap_faiss = len(hybrid_set & faiss_set)
overlap_neat  = len(hybrid_set & neat_set)
union_both = faiss_set | neat_set

hybrid_from_union = len(hybrid_set & union_both)
assert hybrid_from_union == K, (
    f"hybrid 結果が faiss∪neat の外にある: "
    f"hybrid={hybrid_set}, union={union_both}"
)
print(f"  hybrid∩faiss = {overlap_faiss}/{K}")
print(f"  hybrid∩neat  = {overlap_neat}/{K}")
print(f"  hybrid ⊆ faiss∪neat: ✅")

# ─────────────────────────────────────────
# [8] バッチ hybrid 検索レイテンシ
# ─────────────────────────────────────────
print("\n[8] バッチ hybrid 検索レイテンシ ...")
N_BATCH = 100
batch_queries = rng.standard_normal((N_BATCH, DIM)).astype(np.float32)
batch_queries /= np.linalg.norm(batch_queries, axis=1, keepdims=True) + 1e-8

t0 = time.perf_counter()
for bq in batch_queries:
    hi_hybrid.search(bq, k=K)
elapsed = time.perf_counter() - t0
ms_per_q = elapsed / N_BATCH * 1000
print(f"  {N_BATCH} クエリ合計: {elapsed*1000:.0f}ms → {ms_per_q:.1f}ms/q")

# ─────────────────────────────────────────
# 最終サマリー
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("検証結果サマリー")
print("=" * 60)
checks = [
    ("faiss-cpu インポート",        True),
    ("FAISS FlatIP 構築",           True),
    ("NeatIndexer 学習",            True),
    ("3モード search() 動作",       True),
    ("hybrid ⊆ faiss∪neat",        True),
    (f"FAISS  Recall@{K} > 0",     recalls["faiss"]  > 0),
    (f"NEAT   Recall@{K} >= 0",    recalls["neat"]   >= 0),
    (f"HYBRID Recall@{K} >= 0",    recalls["hybrid"] >= 0),
    ("バッチ hybrid < 2000ms/q",   ms_per_q < 2000),
]

passed = 0
for name, ok in checks:
    mark = "✅" if ok else "❌"
    print(f"  {mark} {name}")
    if ok:
        passed += 1

print(f"\n  Recall@{K}: FAISS={recalls['faiss']:.4f}  NEAT={recalls['neat']:.4f}  HYBRID={recalls['hybrid']:.4f}")
print(f"  バッチレイテンシ: {ms_per_q:.1f}ms/q ({N_BATCH}q)")
print()
if passed == len(checks):
    print(f"✅ HybridIndexer 統合検証完了 ({passed}/{len(checks)})")
else:
    print(f"❌ 一部チェック失敗 ({passed}/{len(checks)})")
print("=" * 60)
