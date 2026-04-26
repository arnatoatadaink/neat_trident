"""
MED統合 検証スクリプト (MED DomainIndex準拠版)
StubMED を使って TRIDENT → MED アダプタ層の動作を確認する。

MED準拠シグネチャ:
  search(query: (dim,), k=5) → list[tuple[doc_id: str, score: float]]
  add(doc_ids: list[str], embeddings: (n, dim)) → None

MED実装差し替え時の手順:
  StubMEDIndexer  → MED の DomainIndex に置き換える
  StubMEDSkillStore → MED のスキルストアに置き換える

実行方法:
  poetry run python scripts/med_integration_verify.py
"""

import sys
import numpy as np

sys.path.insert(0, ".")

results = {}

# ──────────────────────────────────────────────
# 1. Protocol インポート確認
# ──────────────────────────────────────────────

print("=" * 60)
print("1. Protocol / スタブ インポート確認")
print("=" * 60)

try:
    from src.med_integration.interfaces import MEDIndexerProtocol, MEDSkillStoreProtocol
    from src.med_integration.stub_med import StubMEDIndexer, StubMEDSkillStore
    from src.med_integration.trident_adapter import TRIDENTMEDAdapter, HybridIndexerMEDAdapter
    print("  ✅ 全モジュールインポート成功")
    results["import"] = True
except Exception as e:
    print(f"  ❌ インポートエラー: {e}")
    results["import"] = False
    sys.exit(1)

# ──────────────────────────────────────────────
# 2. StubMEDIndexer — MED DomainIndex準拠確認
# ──────────────────────────────────────────────

print("\n2. StubMEDIndexer 動作確認 (DomainIndex準拠)")
print("-" * 40)

DIM = 16
N = 30
rng = np.random.default_rng(42)

try:
    stub_idx = StubMEDIndexer(dimension=DIM)
    assert isinstance(stub_idx, MEDIndexerProtocol), "Protocol 互換性エラー"
    print(f"  Protocol 互換     : ✅")

    corpus = rng.standard_normal((N, DIM)).astype(np.float32)
    doc_ids = [f"doc_{i:03d}" for i in range(N)]

    # MED準拠: add(doc_ids, embeddings)
    stub_idx.add(doc_ids, corpus)
    print(f"  add(doc_ids, emb) : ✅  ntotal={stub_idx.ntotal}")

    query = rng.standard_normal(DIM).astype(np.float32)

    # MED準拠: search(query, k) → list[tuple[str, float]]
    results_search = stub_idx.search(query, k=5)
    assert isinstance(results_search, list), f"戻値が list でない: {type(results_search)}"
    assert len(results_search) == 5
    assert isinstance(results_search[0][0], str), "doc_id が str でない"
    assert isinstance(results_search[0][1], float), "score が float でない"
    # スコア降順確認
    scores = [s for _, s in results_search]
    assert scores == sorted(scores, reverse=True), "スコアが降順でない"

    print(f"  search(q, k=5)    : ✅  → list[tuple[str, float]]")
    for doc_id, score in results_search[:3]:
        print(f"    {doc_id}: {score:.4f}")
    results["stub_indexer"] = True
except Exception as e:
    print(f"  ❌ エラー: {e}")
    import traceback; traceback.print_exc()
    results["stub_indexer"] = False

# ──────────────────────────────────────────────
# 3. StubMEDSkillStore 単体確認
# ──────────────────────────────────────────────

print("\n3. StubMEDSkillStore 動作確認")
print("-" * 40)

try:
    store = StubMEDSkillStore()
    assert isinstance(store, MEDSkillStoreProtocol), "Protocol 互換性エラー"
    print(f"  Protocol 互換     : ✅")

    sid = store.store_skill({
        "type": "indexer", "fitness": 0.75,
        "descriptor": [0.3, 0.7], "skill_obj": "dummy",
    })
    assert isinstance(sid, int)
    assert store.get_skill(sid)["fitness"] == 0.75
    print(f"  store/get         : ✅  id={sid}")

    for i, stype in enumerate(["gate", "slot_filler", "indexer"]):
        store.store_skill({"type": stype, "fitness": float(i) * 0.1,
                           "descriptor": [0.5, 0.5], "skill_obj": None})
    print(f"  list_skills(indexer): {len(store.list_skills('indexer'))} 件  ✅")
    print(f"  best_skill: fitness={store.best_skill('indexer')['fitness']:.2f}  ✅")
    results["stub_store"] = True
except Exception as e:
    print(f"  ❌ エラー: {e}")
    results["stub_store"] = False

# ──────────────────────────────────────────────
# 4. HybridIndexerMEDAdapter — MED準拠I/F確認
# ──────────────────────────────────────────────

print("\n4. HybridIndexerMEDAdapter (DomainIndex準拠) 確認")
print("-" * 40)

try:
    from src.interfaces.neat_indexer import NeatIndexer, HybridIndexer

    ni = NeatIndexer(
        input_dim=DIM, pop_size=10, species_size=3,
        max_nodes=30, max_conns=50, generation_limit=5, seed=42,
    )
    corpus_fit = rng.standard_normal((N, DIM)).astype(np.float32)
    ni.fit(corpus_fit)
    hi = HybridIndexer(neat_indexer=ni, mode="neat")

    adapter = HybridIndexerMEDAdapter(hybrid_indexer=hi, dimension=DIM)
    assert isinstance(adapter, MEDIndexerProtocol), "Protocol 互換性エラー"
    print(f"  Protocol 互換     : ✅")

    doc_ids = [f"doc_{i:03d}" for i in range(N)]

    # MED準拠 add()
    adapter.add(doc_ids, corpus_fit)
    print(f"  add(doc_ids, emb) : ✅  ntotal={adapter.ntotal}")

    # MED準拠 search() → list[tuple[str, float]]
    q = rng.standard_normal(DIM).astype(np.float32)
    res = adapter.search(q, k=3)
    assert isinstance(res, list)
    assert isinstance(res[0][0], str), f"doc_id が str でない: {type(res[0][0])}"
    print(f"  search(q, k=3)    : ✅  → {res}")

    # sync_corpus でも同じ結果
    adapter.sync_corpus(doc_ids, corpus_fit)
    res2 = adapter.search(q, k=3)
    assert res == res2
    print(f"  sync_corpus()     : ✅  結果一致")
    results["adapter"] = True
except Exception as e:
    print(f"  ❌ エラー: {e}")
    import traceback; traceback.print_exc()
    results["adapter"] = False

# ──────────────────────────────────────────────
# 5. TRIDENTMEDAdapter 統合フロー
# ──────────────────────────────────────────────

print("\n5. TRIDENTMEDAdapter 統合フロー確認")
print("-" * 40)

try:
    med_idx = StubMEDIndexer(dimension=DIM)
    med_store = StubMEDSkillStore()
    doc_ids = [f"doc_{i:03d}" for i in range(N)]

    full_adapter = TRIDENTMEDAdapter(
        hybrid_indexer=hi,
        med_indexer=med_idx,
        med_skill_store=med_store,
        dimension=DIM,
    )

    # MED準拠 sync_indexer(doc_ids, corpus)
    full_adapter.sync_indexer(doc_ids=doc_ids, corpus=corpus_fit)
    print(f"  sync_indexer()    : ✅")
    print(f"    TRIDENT ntotal  : {full_adapter.indexer_adapter.ntotal}")
    print(f"    MED ntotal      : {full_adapter.med_indexer.ntotal}")

    q = rng.standard_normal(DIM).astype(np.float32)

    # TRIDENT 検索 → list[tuple[str, float]]
    t_res = full_adapter.search(q, k=5)
    assert isinstance(t_res[0][0], str)
    print(f"  TRIDENT search()  : ✅  {[d for d, _ in t_res]}")

    # MED (スタブ) 検索
    m_res = full_adapter.search_med(q, k=5)
    assert isinstance(m_res[0][0], str)
    print(f"  MED search()      : ✅  {[d for d, _ in m_res]}")

    # スキルエクスポート
    sid = full_adapter.export_skill(
        skill_obj=ni, skill_type="indexer", fitness=0.8,
        descriptor=np.array([0.6, 0.4], dtype=np.float32),
    )
    print(f"  export_skill()    : ✅  id={sid}")
    print(f"  summary: {full_adapter.summary()}")
    results["full_adapter"] = True
except Exception as e:
    print(f"  ❌ エラー: {e}")
    import traceback; traceback.print_exc()
    results["full_adapter"] = False

# ──────────────────────────────────────────────
# 6. TRIDENTArchive → MED 一括エクスポート
# ──────────────────────────────────────────────

print("\n6. TRIDENTArchive 一括エクスポート確認")
print("-" * 40)

try:
    from src.map_elites_archive import TRIDENTArchive

    archive = TRIDENTArchive(grid_sizes={"indexer": 4, "gate": 4, "slot_filler": 4})
    for i in range(5):
        archive.add_indexer("idx_skill", float(i) * 0.2, rng.random(2).astype(np.float32))
    for i in range(3):
        archive.add_gate("gate_skill", float(i) * 0.3, rng.random(2).astype(np.float32))

    med_store2 = StubMEDSkillStore()
    full_adapter2 = TRIDENTMEDAdapter(
        hybrid_indexer=hi,
        med_indexer=StubMEDIndexer(DIM),
        med_skill_store=med_store2,
        dimension=DIM,
    )

    exported = full_adapter2.export_archive(archive)
    total_exported = sum(len(v) for v in exported.values())
    assert med_store2.size == total_exported
    print(f"  エクスポート件数  : ✅  {total_exported} スキル  {exported}")
    results["archive_export"] = True
except Exception as e:
    print(f"  ❌ エラー: {e}")
    import traceback; traceback.print_exc()
    results["archive_export"] = False

# ──────────────────────────────────────────────
# 結果サマリー
# ──────────────────────────────────────────────

print("\n" + "=" * 60)
print("MED統合 検証結果サマリー (DomainIndex準拠)")
print("=" * 60)

checks = [
    ("import",         "Protocol/スタブ インポート"),
    ("stub_indexer",   "StubMEDIndexer (DomainIndex準拠)"),
    ("stub_store",     "StubMEDSkillStore"),
    ("adapter",        "HybridIndexerMEDAdapter (MED準拠I/F)"),
    ("full_adapter",   "TRIDENTMEDAdapter 統合フロー"),
    ("archive_export", "TRIDENTArchive 一括エクスポート"),
]

passed = sum(1 for k, _ in checks if results.get(k))
total = len(checks)

for key, label in checks:
    mark = "✅" if results.get(key) else "❌"
    print(f"  {mark} {label}")

print(f"\n  {passed}/{total} チェック通過")
sys.exit(0 if passed == total else 1)
