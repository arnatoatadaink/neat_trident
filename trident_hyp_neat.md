# trident_hyp_neat.md
> Status: mixed (hypothesis / sketch / plan) | Created: 2026-03-26
> 対象システム: TRIDENT (Topology-Routing Interface for Dynamic Neuro-Evolution Tasks)
> 旧ファイル: hyp_h_neat.md から TRIDENT担当部分を分離

---

## 概要

NEAT（NeuroEvolution of Augmenting Topologies）/ HyperNEAT / NAS の設計思想を
TRIDENTのトポロジー進化・ルーティング制御に応用する仮説群。

提唱者: Kenneth Stanley（UCF → OpenAI）
実装基盤: TensorNEAT（JAX/GPU並列化）+ ES-HyperNEAT + Novelty Search + MAP-Elites

---

## NEAT系譜（参照マップ）

```
NEAT (Stanley & Miikkulainen, 2002)
  ├─ rtNEAT (2003) — リアルタイム進化、lifetimeタイマー
  ├─ HyperNEAT (2007/2009) — CPPNによる間接エンコード（数百万接続を小ゲノムで表現）
  │    └─ ES-HyperNEAT (Risi et al., 2010)
  │           ノード配置・密度をweight patternから自動導出
  │           └─ Iterated ES-HyperNEAT (2011) — 計算コスト削減、複雑ドメインでHyperNEAT超え
  │           └─ Adaptive ES-HyperNEAT — シナプス可塑性を統合
  ├─ DeepNEAT / CoDeepNEAT (Miikkulainen et al., 2019)
  │    遺伝子粒度: ノード→層に抽象化。勾配降下と統合
  │    CoDeepNEAT: モジュール + Blueprint染色体の共進化
  │    実績: CIFAR-10・言語モデリングで最先端同等
  ├─ RankNEAT (2022) — 主観ラベルデータ（感情・嗜好）への適用
  ├─ Hybrid Self-Attention NEAT (2021)
  │    self-attentionを間接エンコードとして統合
  │    Atariピクセル入力 → NEATで扱えるようにした
  └─ TensorNEAT (2024) ← TRIDENTの実装基盤
       JAXベースのGPU並列化。NEAT-Pythonより最大500倍高速
       NEAT / CPPN / HyperNEAT / ES-HyperNEATをGPU上で全対応
```

**遺伝子粒度の変遷**

| 系統 | 遺伝子の単位 | 勾配降下との関係 |
|------|------------|--------------|
| NEAT | ノード・接続 | なし（進化のみ） |
| HyperNEAT | CPPN関数 | なし |
| DeepNEAT | 層（Conv/Dense/LSTM） | フィットネス評価時に使用 |
| CoDeepNEAT | モジュール+Blueprint | フィットネス評価時に使用 |
| TensorNEAT | 上記すべてをGPUテンソルで表現 | 両立可能 |

---

## 仮説A: Student × NEAT構造進化 → TRIDENTのrank進化
> **Status: hypothesis → sketch（AdaLoRA確認後）**
> **担当: TRIDENT**

### 核心的な問い
> LoRAのrankを固定せず、GRPOの報酬に応じて動的に拡張できないか？
> これはNEATの「最小構造から必要に応じて複雑化」に対応する。

### TRIDENT内での位置づけ
```
TRIDENT インターフェースコントラクト:
  Type A (float vector): Studentへの入力
  Type B (binary activation): ルーティング判定
  Type C (named key-value): KG/FAISS検索パラメータ

仮説A はType Aの次元をNEAT的に進化させる
  rank=4 → rank=8 → rank=16 の段階的拡張
  GRPOの報酬 < threshold → rank UP（Type Aの次元が増える）
```

### 現実的な着地点
```
候補rank集合: {2, 4, 8, 16}
  GRPO報酬 > threshold → rank維持（収束）
  GRPO報酬 < threshold → rank UP（構造拡張）

NEATそのものではなくNAS/DARTS寄りの実装:
  DARTSは微分可能なアーキテクチャ探索
  AdaLoRAはrank予算の適応的配分
```

### 確認すべき論文
- [ ] DARTS: Differentiable Architecture Search (Liu et al., 2019)
- [ ] AdaLoRA: Adaptive Budget Allocation for LoRA (Zhang et al., 2023)
- [ ] AdaLoRA実装: https://github.com/QingruZhang/AdaLoRA

---

## 仮説B: CPPN × KGエッジ重み生成
> **Status: hypothesis → sketch（HippoRAG確認後）**
> **担当: TRIDENT（長期）/ MED（短期MLP版）**

### 核心的な問い
> FAISSのembedding空間を「座標」として、
> KGのエッジ重みをCPPN的な関数で間接生成できないか？

### TRIDENTとMEDの分担
```
MED側（短期・実装可能）:
  MLP固定アーキテクチャで3項スコア関数を実装
  f(emb_A, emb_B, context) → edge_weight
  → plan_neat_hyp_e.md / trident_plan_hyp_e.md 参照

TRIDENT側（長期・進化）:
  上記MLPのアーキテクチャをNEATで進化させる
  CPPNが「どの座標間の関係が重要か」を学習
  → TRIDENTのType C（named key-value）でKGクエリを制御
```

### 実装イメージ（MED短期版）
```python
def compute_edge_weight(emb_a, emb_b, context_emb):
    base_sim = cosine(emb_a, emb_b)
    context_a = cosine(emb_a, context_emb)
    context_b = cosine(emb_b, context_emb)
    return base_sim + 0.3 * (context_a + context_b)
    # → このアーキテクチャをTRIDENTがNEATで進化させる（長期）
```

### Hyperbolic Embeddingとの統合（MED側追加）
```python
import geoopt
manifold = geoopt.PoincareBall(c=1.0)

def hyperbolic_edge_weight(emb_a, emb_b, context_emb):
    ha = manifold.expmap0(torch.tensor(emb_a, dtype=torch.float64))
    hb = manifold.expmap0(torch.tensor(emb_b, dtype=torch.float64))
    hc = manifold.expmap0(torch.tensor(context_emb, dtype=torch.float64))
    # 双曲距離は階層関係をユークリッドより忠実に表現
    base = 1.0 / (1.0 + manifold.dist(ha, hb))
    ctx  = 1.0 / (1.0 + manifold.dist(ha, hc)) + 1.0 / (1.0 + manifold.dist(hb, hc))
    return base + 0.3 * ctx
```

### 確認すべき論文
- [ ] HippoRAG (Gutierrez et al., 2024) — KG+embedding統合実装
- [ ] R-GCN: Relational Graph Convolutional Networks
- [ ] Multi-Relational Hyperbolic Word Embeddings (EACL 2024)
  GitHub: https://github.com/neuro-symbolic-ai/multi_relational_hyperbolic_word_embeddings

---

## 仮説C: アナログ↔デジタル境界でのNEAT適用
> **Status: hypothesis（仮説Bと合流）**
> **担当: TRIDENT**

### 核心的な観察
NEATが最も強いのは「最適なトポロジーを事前設計できない」問題。
アナログ↔デジタルの変換境界はその典型。

```
アナログ→デジタル（エンコード側）:
  センサー信号 → embedding
  例: 音声波形→MFCC、網膜信号→V1特徴マップ

デジタル→アナログ（デコード側）:
  latent vector → モーター制御 / 連続出力

→ 最適な変換構造を人間が設計するより
  NEATが進化で発見する方が適切
```

```
生物的対応:
  網膜（アナログ）→ V1（位置・方位マップ）
  = CPPN(x, y, orientation) → connection weight と構造が一致
```

### TRIDENTへの接続
```
all-MiniLM-L6-v2（384次元）→ FAISSへの入力変換層
  この変換をCPPNで進化させる
  → タスクごとに最適な表現空間を自動発見
  → TRIDENTのルーティングが変換層のアーキテクチャを制御
```

---

## 仮説D: LLMの「考え方」＝推論トポロジー
> **Status: hypothesis（長期保留）**
> **担当: TRIDENT（概念的基盤）**

### 核心的な観察
LLMの重みはデジタル的に固定されているが、
推論がたどる「軌跡の構造」はアナログ的な連続多様体として存在する。

```
LLMのデジタルな部分: トークン予測・Attention計算
LLMのアナログな部分（考え方）:
  活性化空間の中を推論がたどる連続的な軌跡
  = 「思考スタイル・推論パターン」に対応

推論トポロジーと思考スタイルの対応:
  演繹的思考   → フィードフォワード的トポロジー
  アナロジー   → 遠い領域を橋渡しするスキップ接続
  創造的跳躍   → 予期しないショートカット接続
  帰納的推論   → フィードバックループを含む再帰構造
```

### TRIDENTとの対応
```
TRIDENTがルーティングするトポロジー
  = 推論の「考え方」を動的に切り替えるメカニズム

Type B (binary activation): どの推論パスを使うか
Type C (named key-value): どの記憶コンテキストで考えるか
```

### 確認すべき研究
- [ ] Mechanistic Interpretability (Anthropic/EleutherAI)
- [ ] Sparse Autoencoders for LLM features

---

## 仮説F: 言語処理 × NEATハイブリッド
> **Status: hypothesis（仮説E実装後にsketch検討）**
> **担当: TRIDENT（アーキテクチャ探索）/ MED（embedding処理）**

### 3つの組み合わせ案

```
案F-1: embedding + NEAT（TRIDENTが担当）
  固定embeddingモデル → ベクトル空間
  そのベクトル空間上でNEATがトポロジーを進化

案F-2: 軽量モデル + NEAT（TRIDENTが担当）
  軽量LM（7B LoRA / TinyLLaMA）の層構造を
  CoDeepNEAT的に進化させる
  MEDのStudentが対象モデルの自然な候補

案F-3: NEAT × embedding共進化（長期・野心的）
  embeddingとトポロジーを同時進化
```

### 仮説Eとの関係
```
仮説E（連想関数 / plan_neat_hyp_e.md):
  FAISS検索後の再スコアリング
  f(query, candidate, context) のアーキテクチャをNEATで進化
  → 案F-1の「embedding空間上でNEAT」の縮小版

仮説F:
  言語処理パイプライン全体にNEATを組み込む
  仮説Eが「検索層」なら、仮説Fは「理解・生成層」まで拡張
```

### 日本語固有の課題
```
有利:
  膠着語（形態素が線形・規則的に連結）
  → CPPNの「座標→パターン」写像と親和性が高い
  漢字の部首構造 → Hyperbolic Embeddingと相性良

不利:
  書記体系の多様性（漢字・ひらがな・カタカナ混在）
  語境界が空白で区切られない → 形態素解析が前提
  敬語システム → 社会的文脈が形態に埋め込まれる
  → 32次元圧縮で文体・敬語情報が失われるリスク
```

### 確認すべき研究
- [ ] Elman Network + 進化的アルゴリズム（初期の言語×NE研究）
- [ ] EvoTransformer系の論文
- [ ] Semantic NEAT（あれば）

---

## TRIDENT組み込み箇所マップ

```
all-MiniLM-L6-v2（384次元）
    │
    ▼
[変換層] ← 仮説C: CPPNで進化（長期）/ 現状は固定線形変換
    │
    ▼
FAISS インデクサ（384次元 or 32次元圧縮後）
    │
    ├─→ TRIDENT Type A（float vector）
    │       → 仮説A: rank動的拡張でStudent入力次元を制御
    │
    ├─→ TRIDENT Type B（binary activation）
    │       → IN-DEDUCTIVE Gate: 演繹/帰納パス切り替え
    │       → Chance-Level Threshold（1/n_groups）で判定
    │
    └─→ TRIDENT Type C（named key-value）
            → 仮説B: KGエッジ重みをCPPN/MLPで生成
            → KG検索パラメータとして使用

[CPPN進化ループ] ← TensorNEATが管理（長期）
    GRPOの報酬 → 適応度として使用
    仮説A: rank変化 → Student性能変化 → 報酬
    仮説B: KGエッジ精度 → 報酬
```

---

## Hyperbolic Embeddingライブラリ（利用可能な実装）

### geoopt（最優先・PyTorchネイティブ）
```bash
pip install geoopt
```
```python
import geoopt
manifold = geoopt.PoincareBall(c=1.0)  # ポアンカレ球（負曲率）
# または
manifold = geoopt.Lorentz()            # 双曲面モデル

# ユークリッド → 双曲空間
x_hyp = manifold.expmap0(x_euclidean)

# 双曲距離（コサイン距離の代替）
dist = manifold.dist(x_hyp_a, x_hyp_b)

# Riemannian最適化（torch.optimのドロップイン代替）
optimizer = geoopt.optim.RiemannianAdam(params, lr=1e-3)
```

**注意**: 数値安定性のためfloat64推奨

### 利用可能な関連実装
| リポジトリ | 内容 | 用途 |
|-----------|------|------|
| geoopt/geoopt | Riemannian最適化全般 | **TRIDENT/MED両方のHyperbolic基盤** |
| neuro-symbolic-ai/multi_relational_hyperbolic_word_embeddings | NLP定義からHyperbolic embedding | KGエッジ生成の参考 |
| FranxYao/PoincareProbe | BERTのHyperbolic解析 | embeddingの構造検証 |
| nalexai/hyperlib | TensorFlowベースのHyperbolic NN | 参考実装 |
| TensorNEAT | JAX/GPUベースのNEAT | TRIDENTの進化エンジン |

### all-MiniLM-L6-v2 + Hyperbolicの組み合わせ方
```python
# 現在の流れ（変更なし）
embeddings = sentence_model.encode(texts)     # (N, 384) Euclidean
faiss_index.add(embeddings)                   # FAISSはそのまま維持

# Hyperbolic投影（KGエッジ生成時のみ使用）
import geoopt, torch
manifold = geoopt.PoincareBall(c=1.0)

def to_hyperbolic(emb: np.ndarray) -> torch.Tensor:
    emb_t = torch.tensor(emb, dtype=torch.float64)
    # 境界内に収める（ノルム < 1）
    emb_t = emb_t / (emb_t.norm(dim=-1, keepdim=True) + 1e-8) * 0.9
    return manifold.expmap0(emb_t)
```

---

## LLMレイヤー代替研究の現況

「層を進化で代替していき最終的にLLM全体を置換する」研究は現時点では存在しない。

### 近接する研究
- **LoLCATs (2024)**: Attention層を線形Attentionに差し替え+LoRAで品質回復
- **Hybrid Self-Attention NEAT (2021)**: NEATにAttentionを間接エンコード統合
- **Post-Transformer全般（Mamba・RWKV・xLSTM）**: Attention層をSSM・RNN系で代替

### 完全代替が存在しない理由
```
障壁1: スケールの数桁差
  NEAT実績: 数十〜数百ノード
  LLM規模: 数十億パラメータ

障壁2: 離散進化 vs 連続最適化の非互換
  NEATの突然変異: 離散グラフ操作
  LLMの学習: 勾配降下による連続最適化

障壁3: 世代あたりの評価コスト
  LLMの評価: 1世代に数千〜数万トークン生成が必要
  世代数 × 個体数 × 評価コストが非現実的
```

### 視覚処理下位モデルへの適用（実績あり）
```
実績あり:
  ロボット制御（歩行・姿勢・自動運転） ← HyperNEATの代表的成果
  Atariゲームのピクセル入力処理       ← Hybrid Self-Attention NEAT
  画像分類（CIFAR-10レベル）         ← CoDeepNEATで最先端同等

適用困難:
  ImageNetスケールのCNN             ← 計算コストが非現実的
  Vision Transformer代替            ← Attentionの間接エンコードが未解決
```

---

## 昇格候補の整理

| 仮説 | 現状 | 昇格条件 |
|------|------|---------|
| 仮説A（rank動的拡張） | hypothesis | AdaLoRA論文確認後 → sketch |
| 仮説B（KGエッジCPPN） | hypothesis | HippoRAG実装確認後 → sketch |
| 仮説C（アナログ境界） | hypothesis | 仮説Bと合流 |
| 仮説D（推論トポロジー） | hypothesis | Mech.Interp研究の進展待ち |
| 仮説F（言語×NEAT） | hypothesis | 仮説E実装後 → sketch |

---

## Update Log

| Date | Note |
|------|------|
| 2026-03-26 | hyp_h_neat.md からTRIDENT担当部分を分離、trident_hyp_neat.md として再構成 |
| 2026-03-26 | geoopt/Hyperbolic Embeddingライブラリ情報を追加 |
| 2026-03-26 | TRIDENT組み込み箇所マップを追加（Type A/B/C対応） |
| 2026-03-26 | 仮説F補足（日本語×CPPN分析）を統合 |
