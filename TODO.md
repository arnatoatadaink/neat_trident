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

### [ ] 仮説E Phase 3 — HyperbolicAssociationFn (オプション)

geoopt を使ったポアンカレ球ベースの 3 項スコア関数。  
詳細: `trident_plan_hyp_e.md` Phase 3

```bash
poetry run pip install geoopt
```

### [ ] 384 次元統合テスト

all-MiniLM-L6-v2 の実際の embedding で ContextSensitiveSearch を評価。  
文脈あり/なしの検索結果差を定量測定。

### [ ] NEAT → AssociationFn 進化ループ

TRIDENT が NEAT で association_fn のアーキテクチャを進化させる実装。  
`AssociationFnProtocol` と `swap_association_fn()` は実装済み。

### [ ] MED 実統合

StubMEDIndexer / StubMEDSkillStore を実際の MED DomainIndex に差し替え。  
前提: MED 側の統合テスト完了後。

---

## 完了済み

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
