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

### Phase 1: コア実装（MED側） ✅ 完了 (2026-04-26)

- [x] `AssociationFn` — numpy版（torch不要）を先に実装
  - `fit(feedback_pairs)` で重みを更新できる設計
  - 重みをJSONで保存・ロードできる
  - TRIDENT進化用のアーキテクチャメタデータを保持
- [x] `ContextSensitiveSearch`
  - `build_index(embeddings, texts)`
  - `search(query_emb, context_emb, k, alpha=0.5)`
  - faiss未インストール時はnumpyブルートフォースにフォールバック
- [x] `SearchResult` dataclass

実装: `src/med_integration/context_search.py`

### Phase 2: テスト ✅ 完了 (2026-04-26)

- [x] `test_context_search.py` — 28テスト全通過
  - ダミーembedding（np.random）で動作確認
  - `context_emb=None`（文脈なし）と `context_emb=ctx`（文脈あり）で結果の差を比較
  - スコアの内訳（base / assoc / final）、alpha=0/1 境界テスト含む

実装: `tests/test_context_search.py`

### Phase 3: Hyperbolic統合（オプション） ✅ 完了 (2026-04-27)

- [x] geoopt インストール確認: `pip install geoopt` (0.5.1 + torch 2.11.0)
- [x] `HyperbolicAssociationFn` — ポアンカレ球ベースの3項関数
  - float64で動作することを確認
  - MLP版との精度比較

実装: `src/med_integration/hyperbolic_association.py`  
テスト: `tests/test_hyperbolic_association.py` — 16テスト全通過

### Phase 4: TRIDENTインターフェース準備 ✅ 完了 (2026-04-26)

- [x] アーキテクチャをJSONでシリアライズする設計
  ```json
  {
    "arch_type": "mlp",
    "weights": [0.25, 0.25, 0.25, 0.25],
    "fitness_history": [],
    "generation": 0
  }
  ```
- [x] TRIDENTがアーキテクチャを差し替えられるプラグイン構造
  - `AssociationFnProtocol` (Protocol定義)
  - `ContextSensitiveSearch.swap_association_fn(fn)` メソッド

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

## 推論グラフ化設計（RLTF + NEAT統合）

### 背景

S7（RLTF）とS6（Observer/Solver）の知見から、
「どうして？どうするの推論構造をFAISS上でNEATを使ってグラフ化する」設計を追加。

### 推論ステップのノード定義

```python
@dataclass
class ReasoningNode:
    node_id: str
    node_type: str   # "observe" | "why" | "evidence" | "infer" | "conclude"
    content: str     # ステップの内容
    embedding: np.ndarray  # FAISSに登録するベクトル
    reward: float    # このステップへのRLTF報酬
    
# ノード間のエッジ（推論の繋がり）
@dataclass  
class ReasoningEdge:
    from_id: str
    to_id: str
    weight: float    # association_fn が出力するスコア
    critique: str    # Teacherのテキスト批評（RLTFの報酬信号）
```

### 推論グラフのパイプライン

```
Student の推論ログ:
  A: 何を検索したか（observe）
  B: なぜそれを検索したか（why）← 新規追加
  C: 何が見つかったか（evidence）← FAISSのヒット
  D: どう判断したか（infer）
  E: 結論（conclude）

↓ グラフ化

KGに登録:
  ノード = 各推論ステップ
  エッジ = association_fn(node_i, node_j, context) のスコア
  
Teacher のRLTF批評:
  「B→Dの推論が根拠なく飛躍している」
  → エッジB→Dの重みを下げる負の報酬信号
  → NEATのフィットネス関数に組み込む
```

### NEATへのフィットネス関数設計

```python
def neat_fitness(genome, reasoning_logs, teacher_critiques):
    """
    RLVRのフィットネス関数 = 推論グラフの品質スコア
    
    components:
      1. observer_score: FAISSが正しい根拠を取得できたか（IDEA-002と連動）
      2. reasoning_coherence: 推論ステップの接続が論理的か
      3. teacher_alignment: Teacherの批評と推論の乖離が小さいか（RLTF）
    """
    observer_score      = eval_faiss_retrieval(reasoning_logs)
    coherence_score     = eval_graph_coherence(reasoning_logs)
    teacher_score       = eval_teacher_alignment(reasoning_logs, teacher_critiques)
    
    return (0.4 * observer_score 
          + 0.3 * coherence_score 
          + 0.3 * teacher_score)
```

### NEAT開始タイミング（RLVR知見による前倒し）

```
旧計画:
  Phase 4（IDEA-002/004/005完成後）にNEAT開始

新計画（RLVR知見により前倒し）:
  Phase 2でRLVRフィットネス関数（neat_fitness）が安定したら即開始
  
  理由:
    RLVRが「知識不要の推論能力訓練」であれば
    NEATは「推論グラフのトポロジー探索」として同時に動かせる
    フィットネス関数が固まれば進化を始めてよい
    → FAISSのk値実験（IDEA-002）と並行実行が可能

  最初の世代:
    ゲノム = AssociationFnの重みベクトル [w0, w1, w2, w3]
    突然変異 = 重みの微小変化
    交叉 = 2つの重みセットの混合
    → TensorNEATで数百個体を並列評価（GPU）
```

---

## Phase 5: 推論グラフ実装（追加フェーズ）

- [ ] `ReasoningNode` / `ReasoningEdge` dataclassの実装
- [ ] 推論ログをSQLの `thought_logs` に構造化して保存（IDEA-001拡張）
- [ ] `neat_fitness()` 関数の実装（observer / coherence / teacher の3成分）
- [ ] TensorNEATとの接続（ゲノム = AssociationFnの重みベクトル）
- [ ] Teacherのテキスト批評をエッジ重みにマッピングする変換関数

---

## 評価指標（更新）

```
定量:
  文脈ありなしでの検索結果の差（順位変動率）
  同一クエリ・異なるcontextでの結果多様性
  Chance-Level Threshold（IDEA-009）との連携精度
  Observer精度とSolver精度の独立スコア（IDEA-004拡張A）
  NEAT世代ごとのフィットネス推移

定性:
  「犬」+ context="科学"  → 哺乳類・条件反射 が上位に来るか
  「犬」+ context="創作"  → 忠誠・孤独・友情 が上位に来るか
  推論グラフで「飛躍のあるステップ」が可視化できるか
  
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
| 2026-04-26 | 推論グラフ化設計を追加（RLTF+NEAT統合）、NEATフィットネス関数設計、Phase 5追加、NEAT開始タイミングをPhase 2並行に前倒し |
| 2026-04-26 | Phase 1/2/4 実装完了 (context_search.py + test_context_search.py 28テスト通過) |
| 2026-04-27 | Phase 3 実装完了 (hyperbolic_association.py + test_hyperbolic_association.py 16テスト通過) |
