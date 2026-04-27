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
# 7. DomainIndexAdapter — 実DomainIndex 統合確認
# ──────────────────────────────────────────────

print("\n7. DomainIndexAdapter (実 MED DomainIndex) 確認")
print("-" * 40)

import os
med_root = os.environ.get("MED_ROOT", "")

if not med_root or not os.path.isdir(med_root):
    print("  ⏭  MED_ROOT が未設定または存在しないためスキップ")
    print("     実行例: MED_ROOT=/path/to/MED poetry run python scripts/med_integration_verify.py")
    results["domain_index"] = None   # None = skipped
else:
    try:
        import sys as _sys
        # MED と TRIDENT が同じ "src" パッケージ名を使うため sys.modules を一時退避して衝突を回避する
        _saved_src_mods = {k: _sys.modules.pop(k) for k in list(_sys.modules.keys())
                           if k == "src" or k.startswith("src.")}
        _sys.path.insert(0, med_root)
        try:
            from src.memory.faiss_index import DomainIndex
            from src.common.config import FAISSIndexConfig
        finally:
            # MED の src.* を sys.modules から除去し TRIDENT の src.* を復元する
            for k in list(_sys.modules.keys()):
                if k == "src" or k.startswith("src."):
                    del _sys.modules[k]
            _sys.modules.update(_saved_src_mods)
        from src.med_integration.domain_index_adapter import DomainIndexAdapter, _check_protocol

        cfg = FAISSIndexConfig(dim=DIM, initial_type="Flat", metric="inner_product")
        di  = DomainIndex(cfg)
        adapter = DomainIndexAdapter(di, dimension=DIM)

        assert _check_protocol(adapter), "Protocol 非準拠"
        assert isinstance(adapter, MEDIndexerProtocol), "isinstance チェック失敗"
        print(f"  Protocol 互換     : ✅")

        doc_ids_r = [f"real_{i:03d}" for i in range(N)]
        corpus_r  = rng.standard_normal((N, DIM)).astype(np.float32)
        adapter.add(doc_ids_r, corpus_r)
        assert adapter.ntotal == N, f"ntotal={adapter.ntotal}, expected {N}"
        assert adapter.dimension == DIM
        print(f"  add + ntotal      : ✅  ntotal={adapter.ntotal}, dim={adapter.dimension}")

        q_r   = rng.standard_normal(DIM).astype(np.float32)
        res_r = adapter.search(q_r, k=5)
        assert len(res_r) == 5
        assert isinstance(res_r[0][0], str)
        assert res_r[0][1] >= res_r[-1][1], "スコアが降順でない"
        print(f"  search(q, k=5)    : ✅  top={res_r[0]}")

        # TRIDENTMEDAdapter に実アダプタを渡してエンドツーエンド確認
        med_store_r = StubMEDSkillStore()
        full_r = TRIDENTMEDAdapter(
            hybrid_indexer=hi,
            med_indexer=adapter,
            med_skill_store=med_store_r,
            dimension=DIM,
        )
        full_r.sync_indexer(doc_ids=doc_ids_r, corpus=corpus_r)
        med_res_r = full_r.search_med(q_r, k=5)
        assert isinstance(med_res_r[0][0], str)
        print(f"  TRIDENTMEDAdapter : ✅  med_ntotal={adapter.ntotal}")
        print(f"  summary: {full_r.summary()}")
        results["domain_index"] = True
    except Exception as e:
        print(f"  ❌ エラー: {e}")
        import traceback; traceback.print_exc()
        results["domain_index"] = False

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
    ("domain_index",   "DomainIndexAdapter (実MED DomainIndex)"),
]

passed  = sum(1 for k, _ in checks if results.get(k) is True)
skipped = sum(1 for k, _ in checks if results.get(k) is None)
total   = len(checks) - skipped

for key, label in checks:
    v    = results.get(key)
    mark = "✅" if v is True else ("⏭ " if v is None else "❌")
    print(f"  {mark} {label}")

print(f"\n  {passed}/{total} チェック通過", end="")
if skipped:
    print(f"  ({skipped} スキップ)", end="")
print()
sys.exit(0 if passed == total else 1)
