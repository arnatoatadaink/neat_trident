# TRIDENT — ユーザーリファレンス

> このファイルはユーザー向け日本語参照資料です。Claude は CLAUDE.md を読みます。

---

## プロジェクト概要

TRIDENT は MED Framework 内で NEAT を主体とした自律的スキル探索・収集を担うサブシステムです。
MED の FAISS メモリに対して、NEAT 進化によって最適化された検索・連想関数を提供します。

詳細設計: `plan_trident.md`  
仮説文書: `trident_hyp_neat.md`

---

## 現在のフェーズ進捗

| フェーズ | 内容 | 状態 |
|---------|------|------|
| Phase 0 | 環境調査 | ✅ 完了 |
| Phase 1 | A型 NeatIndexer 実装 | ✅ 完了 |
| Phase 2 | B型 NeatGate / C型 NeatSlotFiller | ✅ 完了 |
| Phase 3 | ES-HyperNEAT カスタム拡張 | ✅ 完了 |
| Phase 4 | MAP-Elites + Novelty Search 統合 | ✅ 完了 |
| Phase 5 | FAISS / HybridIndexer 統合 | ✅ 完了 |
| Phase 6 | MED 統合アダプタ層 | ✅ 完了 |
| 仮説E | AssociationFn (ContextSensitiveSearch) | ✅ Phase1/2/4 完了、Phase3(Hyperbolic)保留 |
| GPU設定 | WSL2 CUDA + JAX GPU 有効化 | ✅ 完了 (GTX 1650) |

---

## 技術スタック詳細

| コンポーネント | ライブラリ | バージョン | 状態 |
|-------------|----------|-----------|------|
| NEAT / HyperNEAT | TensorNEAT | 0.1.0 (GitHub) | ✅ |
| MAP-Elites | QDax (JAX) | 0.5.0 | ✅ |
| Novelty Search | カスタム実装 (JAX) | — | ✅ |
| ES-HyperNEAT | TensorNEAT 拡張 | — | ✅ |
| JAX | jax | 0.9.2 | ✅ CPU + GPU |
| FAISS | faiss-cpu | 1.13.2 | ✅ |
| GPU | GTX 1650 / CUDA 12 | — | ✅ (WSL2) |
| T4 GPU | — | — | ❌ 専用ドライバ未対応 |

---

## 実装済みファイル一覧

```
src/
├── interfaces/
│   ├── neat_indexer.py       # A型 + HybridIndexer (FAISS/NEAT/hybrid)
│   ├── neat_gate.py          # B型 + NeatAugmentedReward
│   └── neat_slot_filler.py   # C型 + NeatKGWriter
├── med_integration/
│   ├── interfaces.py         # MEDIndexerProtocol / MEDSkillStoreProtocol
│   ├── stub_med.py           # StubMEDIndexer / StubMEDSkillStore
│   ├── trident_adapter.py    # HybridIndexerMEDAdapter / TRIDENTMEDAdapter
│   └── context_search.py     # AssociationFn / ContextSensitiveSearch (仮説E)
├── map_elites_archive.py     # SkillRepertoire / TRIDENTArchive / EvolutionLoop
├── novelty_search.py         # knn_novelty_scores / NoveltyArchive / NoveltyFitness
└── es_hyperneat.py           # make_trident_substrate / ESHyperNEATProjector / ESHyperNEATIndexer

tests/
├── test_neat_indexer.py      # 17 tests
├── test_neat_gate.py         # 13 tests
├── test_neat_slot_filler.py  # 17 tests
├── test_novelty_search.py    # 21 tests
├── test_map_elites.py        # 21 tests
├── test_es_hyperneat.py      # 19 tests
└── test_context_search.py    # 28 tests  ← 新規 (仮説E)

scripts/
├── neat_benchmark.py         # 30分ベンチマーク (CPU/GPU 対応)
├── med_integration_verify.py # MED統合 6チェック
└── ...
```

---

## ベンチマーク結果

詳細ログ・スケーリング調査は `LOG.md` を参照。

要約 (pop=100, RTX 4060): Spiral **14ms/gen** (CPU比 17x)、VectorNeighbor **8.2s 収束** (CPU比 16x)

---

## 注意事項

- **tensorneat**: PyPI 未公開 → `pip install git+https://github.com/EMI-Group/tensorneat.git`
- **JAX GPU**: `LD_LIBRARY_PATH` に venv 内の nvidia パッケージ lib を追加済み (`~/.bashrc`)
- **T4 GPU**: サーバー用カード・専用ドライバ未インストール → 使用不可
- **ES-HyperNEAT**: TensorNEAT 0.1.0 未対応 → `HyperNEAT` クラスをベースにカスタム実装
- **faiss-cpu**: `python<3.15` の制約あり → pyproject.toml で指定済み

---

## 次のセッションでの作業候補

1. **仮説E Phase 3** — HyperbolicAssociationFn (geoopt) の実装
2. **384次元統合テスト** — all-MiniLM-L6-v2 実際の embedding で ContextSensitiveSearch 評価
3. **NEAT→AssociationFn 進化ループ** — TRIDENT が NEAT で association_fn を進化させる実装
4. **MED 実統合** — StubMED を実際の MED DomainIndex に差し替え

---

## 環境確認コマンド

```bash
# テスト全実行
poetry run pytest tests/ -q

# GPU 確認
poetry run python -c "import jax; print(jax.devices())"

# MED 統合確認
poetry run python scripts/med_integration_verify.py
```
