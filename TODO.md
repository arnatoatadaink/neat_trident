# TRIDENT — TODO

> 実装・実験タスクの管理ファイル。詳細ログは LOG.md、設計は CLAUDE_USER.md を参照。

---

## 実験・パラメーター探索

### [ ] species_size 固定 20 での pop スケーリング検証

**背景**  
Experiment 4 (LOG.md) で species_size=√pop を試した結果、
種あたり個体数 ≈ 20 がスイートスポットと示唆された (pop=400/sp=20 が全試験最良 fitness -0.586)。

**仮説**  
species_size を 20 に固定したまま pop を変えると、
pop×gen や best fitness がより安定して改善するか？

**テスト条件**

| pop | species_size | 種あたり個体数 |
|-----|-------------|-------------|
| 200 | 20 | 10 |
| 400 | 20 | 20 ← 現状最良 |
| 600 | 20 | 30 |
| 1000 | 20 | 50 |
| 2000 | 20 | 100 |

**実行方法**
```bash
poetry run python scripts/neat_benchmark.py --max-seconds 600 --pop-size <N> --species-size 20
```

**確認観点**
- pop×gen 総評価数のピーク位置
- best fitness の推移
- GPU使用率・メモリコントローラー使用率の変化
- 種あたり個体数 20 が最適かどうかの再検証

---

## 実装

### [x] Optuna NEAT ハイパーパラメーター最適化 ✅ 2026-04-27

`scripts/neat_optuna_tune.py` — TPESampler + SQLite (logs/optuna_neat.db)

探索空間:
- `pop_size` int log[50,1000], `species_size` int[5,100], `generation_limit` int[10,80]
- `max_nodes` categorical[64,128,256], `max_conns` categorical[128,512,1024]

実行:
```bash
poetry run python scripts/neat_optuna_tune.py --n-trials 30 --dim 16 --corpus-size 100
```

可視化: `optuna-dashboard logs/optuna_neat.db`

---

### [x] 仮説E Phase 3 — HyperbolicAssociationFn (オプション) ✅ 2026-04-27

geoopt 0.5.1 / torch 2.11.0 でポアンカレ球ベースの 3 項スコア関数を実装。  
`src/med_integration/hyperbolic_association.py` + 16 テスト通過。

### [x] 384 次元統合テスト ✅ 2026-04-27

all-MiniLM-L6-v2 (384次元) で ContextSensitiveSearch を定量評価。

**知見:**
- alpha=0.5 では FAISS ベーススコアが支配的で文脈効果は限定的
- Hyperbolic 版は英語の曖昧クエリ ("python") で top-5 変化を示した
- MLP デフォルト等重みは文脈感度なし → **NEAT進化で重み最適化の動機を定量的に確認**
- 日本語コーパスは all-MiniLM-L6-v2 (英語特化) との相性不足

実装: `scripts/eval_384dim.py`, `tests/test_384dim_interface.py` (11テスト)

### [x] NEAT → AssociationFn 進化ループ ✅ 2026-04-27

TRIDENT が NEAT で association_fn アーキテクチャを進化させる実装。

- `AssociationFnProblem` — fitness=-MSE(score, label), input=[cos(q,c), cos(q,ctx), cos(c,ctx), cos(q-ctx,c)]
- `NEATAssociationFn` — AssociationFnProtocol 準拠、pickle 永続化、swap_association_fn 対応
- `AssociationFnEvolver` — feedback_pairs → NEAT 進化ループ

実装: `src/med_integration/neat_assoc_evolver.py`, 19テスト全通過

### [x] MED 実統合 ✅ 2026-04-27

DomainIndexAdapter で MED の DomainIndex を MEDIndexerProtocol に適合させた。

**実装内容:**
- `pydantic` / `pydantic-settings` を TRIDENT venv に追加 (MED 依存解消)
- `DomainIndexAdapter`: `DomainIndex.count → ntotal`、`dimension` コンストラクタ注入
- `sys.modules` 退避で `src` パッケージ名衝突を回避して MED インポート成功
- `med_integration_verify.py` チェック7追加: MED_ROOT 環境変数で実 DomainIndex テスト
- `tests/test_domain_index_adapter.py` 11テスト通過

**注意: SkillStore には MED 側の対応コンポーネントがなく StubMEDSkillStore を継続使用。**

---

## 完了済み

- [x] Optuna NEAT ハイパーパラメーター最適化 (neat_optuna_tune.py, SQLite永続化)
- [x] MED 実統合 (DomainIndexAdapter + 11テスト, med_integration_verify 7/7通過)
- [x] NEAT → AssociationFn 進化ループ (neat_assoc_evolver.py + 19テスト)
- [x] 384次元統合テスト (eval_384dim.py + 11テスト)
- [x] 仮説E Phase 3 — HyperbolicAssociationFn (geoopt 0.5.1)
- [x] A/B/C 型 NEAT インターフェース実装 + pytest 136 テスト通過
- [x] ES-HyperNEAT カスタム拡張
- [x] MAP-Elites + Novelty Search 統合
- [x] FAISS / HybridIndexer 統合
- [x] MED 統合アダプタ層 (stub)
- [x] 仮説E Phase 1/2/4 — AssociationFn + ContextSensitiveSearch 実装
- [x] WSL2 CUDA + JAX GPU 有効化 (RTX 4060)
- [x] NEAT ベンチマーク CPU/GPU 比較 (LOG.md Exp.1)
- [x] pop スケーリング調査 100〜2000 (LOG.md Exp.2/3)
- [x] species_size=√pop 効果検証 (LOG.md Exp.4)
