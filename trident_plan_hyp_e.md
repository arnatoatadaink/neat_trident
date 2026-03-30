# trident_plan_hyp_e.md
> Status: plan | Created: 2026-03-26
> 元仮説: trident_hyp_neat.md > 仮説E（連想関数としてのCPPN）
> 実装はClaude Code CLI（WSL2）で行う
> 旧ファイル: plan_neat_hyp_e.md をTRIDENT側に再配置・更新

---

## 目的

FAISSの検索結果を `association_fn(query, candidate, context)` で
再スコアリングし、文脈依存・非対称な連想検索を実現する。

**短期（MED側）**: MLP固定アーキテクチャで実装  
**長期（TRIDENT側）**: アーキテクチャをNEATで進化させる

---

## 設計

### インターフェース

```python
results = searcher.search(
    query_emb=q,       # 検索クエリのembedding（all-MiniLM-L6-v2の384次元）
    context_emb=ctx,   # 文脈（直前の会話 / タスクタイプなど）
    k=5
)
```

### パイプライン

```
query_emb (384次元)
    │
    ▼
FAISS.search(k * 3件) ─── 候補を広く取る
    │
    ▼
association_fn(query, candidate, context)  ← ここがCPPN的な3項関数
    │   現状: MLP固定アーキテクチャ（MED側が管理）
    │   将来: TRIDENTがNEATでアーキテクチャを進化
    ▼
rerank(top-k)
    │
    ▼
List[SearchResult]
```

### association_fn の設計（現状 → 将来）

```
現状 (MLP固定アーキテクチャ / MED側):
    score = w0 * cosine(q, c)
          + w1 * cosine(q, ctx)
          + w2 * cosine(c, ctx)
          + w3 * cosine(q-ctx, c)   ← 文脈差分ベクトルとの類似度
    w_i は学習可能パラメータ（初期値は等重み）

Hyperbolic版（geoopt使用 / MED側オプション）:
    manifold = geoopt.PoincareBall(c=1.0)
    score = 1/(1+manifold.dist(h_q, h_c))
          + 0.3 * (1/(1+manifold.dist(h_q, h_ctx))
                 + 1/(1+manifold.dist(h_c, h_ctx)))
    ※ h_x = manifold.expmap0(x_euclidean)

将来 (NEAT進化 / TRIDENT側):
    score = CPPN(q, c, ctx)
    ↑ このアーキテクチャ自体をTRIDENTがNEATで進化させる
    フィットネス = FAISSの検索精度 + KGのエッジ品質
```

### SearchResult

```python
@dataclass
class SearchResult:
    index:       int
    text:        str
    base_score:  float    # FAISSのコサイン類似度
    assoc_score: float    # association_fn の出力
    final_score: float    # alpha * base + (1-alpha) * assoc
    embedding:   np.ndarray
```

---

## ファイル構成

```
context_search.py       # メイン実装
  └─ AssociationFn      # 3項スコア関数（TRIDENT進化のプレースホルダー）
  └─ ContextSensitiveSearch  # FAISSラッパー
  └─ SearchResult       # 結果データクラス

test_context_search.py  # 動作確認スクリプト
  └─ numpy fallback テスト（faiss/torch なしで動作確認）
  └─ 文脈ありなしの比較テスト
  └─ Hyperbolic版との比較（オプション）
```

---

## 実装タスク（Claude CLI向け）

### Phase 1: コア実装（MED側）

- [ ] `AssociationFn` — numpy版（torch不要）を先に実装
  - `fit(feedback_pairs)` で重みを更新できる設計
  - 重みをJSONで保存・ロードできる
  - TRIDENT進化用のアーキテクチャメタデータを保持
- [ ] `ContextSensitiveSearch`
  - `build_index(embeddings, texts)`
  - `search(query_emb, context_emb, k, alpha=0.5)`
  - faiss未インストール時はnumpyブルートフォースにフォールバック
- [ ] `SearchResult` dataclass

### Phase 2: テスト

- [ ] `test_context_search.py`
  - ダミーembedding（np.random）で動作確認
  - `context_emb=None`（文脈なし）と `context_emb=ctx`（文脈あり）で
    結果の差を比較出力
  - スコアの内訳（base / assoc / final）をログ出力

### Phase 3: Hyperbolic統合（オプション）

- [ ] geoopt インストール確認: `pip install geoopt`
- [ ] `HyperbolicAssociationFn` — ポアンカレ球ベースの3項関数
  - float64で動作することを確認
  - MLP版との精度比較

### Phase 4: TRIDENTインターフェース準備

- [ ] アーキテクチャをJSONでシリアライズする設計
  ```json
  {
    "arch_type": "mlp",
    "weights": [0.25, 0.25, 0.25, 0.25],
    "fitness_history": [],
    "generation": 0
  }
  ```
- [ ] TRIDENTがアーキテクチャを差し替えられるプラグイン構造

---

## context_emb の生成元候補

| 候補 | 内容 | 実装コスト |
|------|------|---------|
| A | 直前のTeacher応答のembedding | 低（すでにembeddingあり） |
| B | タスクタイプの固定embedding（"creative"/"factual"等） | 低（辞書で管理） |
| C | 直近N件のやり取りのmean pooling | 中（N件の集約処理） |
| D | スタイルベクトル（仮説G由来） | 高（StyleExtractor必要） |

**推奨初期実装**: 候補A（最も実装が簡単）

---

## 評価指標

```
定量:
  文脈ありなしでの検索結果の差（順位変動率）
  同一クエリ・異なるcontextでの結果多様性
  Chance-Level Threshold（IDEA-009）との連携精度

定性:
  「犬」+ context="科学"  → 哺乳類・条件反射 が上位に来るか
  「犬」+ context="創作"  → 忠誠・孤独・友情 が上位に来るか
  
日本語固有:
  敬語context と 素体context で異なる検索結果が得られるか
```

---

## 昇格履歴

| From | To | Date | 理由 |
|------|----|------|------|
| hyp_h_neat.md 仮説E | plan_neat_hyp_e.md | 2026-03-26 | MLP版が数十行で実装可能 |
| plan_neat_hyp_e.md | trident_plan_hyp_e.md | 2026-03-26 | TRIDENT/MED分離に合わせて再配置・Hyperbolic追加 |

---

## Update Log

| Date | Note |
|------|------|
| 2026-03-26 | plan_neat_hyp_e.md からTRIDENT側に移動・更新 |
| 2026-03-26 | Hyperbolic版（geoopt）とTRIDENTインターフェース設計を追加 |
| 2026-03-26 | context_emb候補表、Phase 3/4タスクを追加 |
