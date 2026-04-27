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
| MED実統合 | DomainIndexAdapter (実DomainIndex) | ✅ 完了 |
| 仮説E | AssociationFn (ContextSensitiveSearch) | ✅ Phase1/2/3/4 全完了 |
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
│   ├── context_search.py     # AssociationFn / ContextSensitiveSearch (仮説E)
│   ├── hyperbolic_association.py  # HyperbolicAssociationFn (仮説E Phase3)
│   ├── neat_assoc_evolver.py # AssociationFnEvolver / NEATAssociationFn (NEAT進化ループ)
│   └── domain_index_adapter.py   # DomainIndexAdapter (実MED DomainIndex統合)
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
├── test_context_search.py       # 28 tests (仮説E)
├── test_hyperbolic_association.py # 16 tests (仮説E Phase3)
├── test_384dim_interface.py     # 11 tests (384次元統合)
├── test_neat_assoc_evolver.py   # 19 tests (NEAT進化ループ)
└── test_domain_index_adapter.py # 11 tests (実MED DomainIndex統合)

scripts/
├── neat_benchmark.py         # 30分ベンチマーク (CPU/GPU 対応)
├── neat_optuna_tune.py       # Optuna NEAT ハイパーパラメーター最適化
├── med_integration_verify.py # MED統合 7チェック
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

1. **Optuna チューニング実行** — 本番サイズ (dim=16, corpus=100) で 30〜50 試行
   ```bash
   poetry run python scripts/neat_optuna_tune.py --n-trials 50
   optuna-dashboard logs/optuna_neat.db
   ```
2. **埋め込み空間最適化 (Task 2)** — MED 側 FAISS 分布可視化後に実施
3. **species_size=20 固定 pop スケーリング検証** — pop=200/400/600/1000/2000 で性能比較
   ```bash
   poetry run python scripts/neat_benchmark.py --max-seconds 600 --pop-size <N> --species-size 20
   ```
4. **MED SkillStore 実装** — MED 側に MEDSkillStoreProtocol 相当のコンポーネントが必要 (現在 StubMEDSkillStore を継続使用)

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
