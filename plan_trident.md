# TRIDENT — Topology-Routing Interface for Dynamic Neuro-Evolution
> MED Framework 進化的トポロジ探索サブシステム  
> 作成日: 2026-03-29  
> ステータス: 設計フェーズ

---

## 1. 概要と命名

**TRIDENT**（*Topology-Routing Interface for Dynamic Neuro-Evolution Tasks*）は、
MEDフレームワーク内でNEATを主体とした自律的スキル探索・収集・組み合わせを担うサブシステムです。

名前の由来:
- 三叉（3つのI/F型: A / B / C）
- Dynamic = TensorNEATによるGPU並列進化
- Neuro-Evolution = ES-HyperNEAT + Novelty Search + MAP-Elites の統合

---

## 2. 設計思想

### コアコンセプト

FAISS（ベクトル近傍探索）の索引構造をNEATトポロジで段階的に置き換え、
最終的にLLMの推論機能の一部をNEATモジュール群で代替する。

```
現状のMED                         TRIDENTによる発展形
─────────────────────             ──────────────────────────────────
FAISS（kNN索引）          →       NEAT トポロジ（進化的近傍探索）
GRPO報酬判定（ルールベース） →   NEAT ゲートモジュール（B型 I/F）
KG / SQL 操作（手動設計）    →   NEAT スロット生成（C型 I/F）
```

### 3つのI/Fタイプとその意味

| I/F型 | データ型 | 接続先空間 | 概念 |
|-------|----------|-----------|------|
| **A: float vector** | `f32[N]` 固定次元 | 言語空間（LLM埋め込み） | 密・連続 |
| **B: binary activation** | `bool[N]` スパース発火 | 圧縮空間（条件・スキル選択） | 疎・離散 |
| **C: named slots** | `{key: f32}` 構造化 | 中間空間（KG/SQL接合） | 構造化・意味付き |

**タスク区切り基準**: 入出力がどのI/F型で完結するかで区切る。
人間が意味を理解できるかではなく、「モジュールの出力が次の入力になれるか」が基準。

---

## 3. コア技術スタック

### 必須コンポーネント

| コンポーネント | 役割 | 使用ライブラリ |
|-------------|------|--------------|
| **TensorNEAT** | GPU並列NEAT（最大500×高速化） | JAX + TensorNEAT |
| **ES-HyperNEAT** | 幾何学的トポロジ探索 | TensorNEAT内蔵 |
| **Novelty Search** | 多様なスキルの維持・発見 | カスタム実装（既存nsライブラリ検討）|
| **MAP-Elites** | 質×多様性アーカイブ | pymap_elites or カスタム実装 |

### 既存ツールの調査と判断（Phase 0）

```
調査対象:
  - TensorNEAT (github.com/EMI-Group/tensorneat)  → ES-HyperNEAT対応済み ✅
  - pymap_elites                                   → 要バージョン確認
  - neat-python                                    → TensorNEATで代替予定 → 不要
  - evosax (JAX-based)                             → NS実装の参考に
  - QD-Gym / qdax                                  → MAP-Elites + JAX → 検討

判断基準:
  - JAX/TensorNEATとの統合性（同一デバイスでバッチ処理できるか）
  - MED既存パイプライン（FAISS/GRPO/KG）との接合のしやすさ
  - WSL2 + GPU環境での動作確認
```

---

## 4. I/F別タスク詳細

### A型タスク（float vector）

```python
# NEATの出力層 = f32[N]
# N = FAISSの索引次元（例: 384 for MiniLM）

class NeatIndexer:
    """NEATトポロジによるクエリ→近傍ベクトル変換"""
    # 入力: クエリ埋め込み f32[384]
    # 出力: 近傍候補スコア f32[384]（FAISSの代替）
    # BCS (MAP-Elites): 次元ごとの活性化分布 × 精度
```

対応タスク:
- FAISS索引の置き換え（kNN → 進化的近傍探索）
- クエリルーティング（どの記憶領域に問うか）
- LoRA delta-weight生成（重み行列をflattenしてベクトルとして扱う）

### B型タスク（binary activation）

```python
# NEATの出力層 = bool[N]（閾値超えで発火）
# N = スキル数 or 条件数

class NeatGate:
    """条件判定・スキル選択ゲート"""
    # 入力: 文脈ベクトル f32[M]
    # 出力: スキル有効化マスク bool[K]
    # BCS: 発火率 × 特異性（特定タスクへの感度）
```

対応タスク:
- GRPOの報酬判定（正解か否かの二値判定）
- スキル選択ゲーティング（複数スキルの有効化マスク）
- 信頼度フィルタ（閾値超えのみ次段へ）

### C型タスク（named slots）

```python
# NEATの出力ニューロン = 名前付きスロット
# 例: {entity_id: f32, relation_type: f32, confidence: f32}

class NeatSlotFiller:
    """KG/SQL操作のスロット生成"""
    # 入力: 文脈 f32[M]
    # 出力: {node: f32, relation: f32, weight: f32}
    # BCS: スロット充填率 × 正確性
```

対応タスク:
- KGエンティティ操作（node / relation / weight）
- SQLスロット埋め（table / col / condition）
- GRPO報酬スロット（reward / reason / target）

---

## 5. MAP-Elites 行動特性空間（BCS）設計

```
A型 BCS:
  軸1: ベクトル次元ごとの平均活性化（0〜1、384次元→PCAで2次元に圧縮）
  軸2: 検索精度（Recall@k）
  格子: 64×64 = 4096セル

B型 BCS:
  軸1: 発火率（0〜1）
  軸2: 特異性（1タスクへの集中度、エントロピーの逆数）
  格子: 16×16 = 256セル

C型 BCS:
  軸1: スロット充填率（埋まったスロット数 / 全スロット数）
  軸2: 出力正確性（Teacher評価スコア）
  格子: 16×16 = 256セル
```

---

## 6. 既存ツール調査フェーズ（Phase 0）

### 調査チェックリスト

```bash
# TensorNEAT の動作確認
pip install tensorneat  # or git clone
python -c "from tensorneat import NEAT; print('ok')"

# ES-HyperNEAT 対応確認
grep -r "ESHyperNEAT\|es_hyperneat" tensorneat/

# QDax（MAP-Elites + JAX）確認
pip install qdax
python -c "from qdax.core.map_elites import MAPElites; print('ok')"

# Novelty Search 既存実装確認
# → nsga2-torch, evosax のNS実装を確認
```

### 判断マトリクス

| ツール | JAX統合 | GPU並列 | NS対応 | MAP-Elites | 判定 |
|--------|---------|---------|--------|------------|------|
| TensorNEAT | ✅ | ✅ | ❌ | ❌ | 採用（NEAT/ES-HyperNEAT） |
| QDax | ✅ | ✅ | ⚠️ | ✅ | 採用候補（MAP-Elites） |
| evosax | ✅ | ✅ | ⚠️ | ⚠️ | 参考（NSの実装参考） |
| neat-python | ❌ | ❌ | ❌ | ❌ | 不採用（TensorNEATで代替） |

---

## 7. フェーズ計画

### Phase 0: 環境整備（1〜2週間）

```bash
# 目標: TensorNEAT + QDax の WSL2/GPU 動作確認
# 成果物: インストール済み環境 + 動作確認スクリプト

tasks:
  - TensorNEAT インストール・基本動作確認
  - ES-HyperNEAT サンプル実行（Braxタスク）
  - QDax MAP-Elites サンプル実行
  - JAX GPU設定（WSL2 + CUDA）確認
  - NSの既存実装有無の最終確認 → なければカスタム実装設計
```

### Phase 1: A型 I/F 実装（2〜3週間）

```python
# 目標: FAISSの部分代替
# 成果物: NeatIndexer（A型）の動作するプロトタイプ

# FAISS並列稼働モード（既存を壊さない）
class HybridIndexer:
    faiss_index: FAISSIndex      # 既存
    neat_indexer: NeatIndexer    # 新規（A型）
    mode: Literal["faiss", "neat", "hybrid"]
```

### Phase 2: B型 I/F 実装（2〜3週間）

```python
# 目標: GRPOゲートの一部をNEATに置き換え
# 成果物: NeatGate（B型）の動作するプロトタイプ

# GRPO報酬関数にNeatGateを組み込み
class NeatAugmentedReward:
    neat_gate: NeatGate          # スキル選択（B型）
    base_reward: RewardFunction  # 既存GRPO報酬
```

### Phase 3: C型 I/F 実装（3〜4週間）

```python
# 目標: KG操作のスロット生成をNEATに置き換え
# 成果物: NeatSlotFiller（C型）の動作するプロトタイプ

class NeatKGWriter:
    neat_filler: NeatSlotFiller  # スロット生成（C型）
    kg_store: KnowledgeGraphStore  # 既存KG
```

### Phase 4: 統合・スキルライブラリ化（継続）

```
- MAP-Elitesアーカイブ = スキルライブラリ
- Novelty Search = 新規スキル発見ループ
- ES-HyperNEAT = 幾何的スキル空間の探索
- 最終目標: Studentモデルの推論の一部をTRIDENTモジュール群で代替
```

---

## 8. MEDフレームワークとの接点

```
MEDコンポーネント          TRIDENTの対応
────────────────────────  ────────────────────────────────
FAISS索引             →   A型: NeatIndexer（進化的近傍探索）
ModelRouter           →   B型: NeatGate（スキル選択・ルーティング）
KnowledgeGraphStore   →   C型: NeatSlotFiller（KGスロット生成）
GRPO報酬設計          →   B型: NeatGate（報酬判定ゲート）
TinyLoRA delta-weight →   A型: LoRA重み生成（ベクトル出力）
```

---

## 9. 未決事項と優先度

| 項目 | 選択肢 | 優先度 |
|------|--------|--------|
| NS（Novelty Search）実装 | QDax流用 / evosax参考 / カスタム | 高（Phase 0で決定） |
| MAP-Elites ライブラリ | QDax採用 / カスタム | 高（Phase 0で決定） |
| A型の次元数N | FAISSと同じ（384）/ 独立設計 | 中（Phase 1で決定） |
| B型の閾値戦略 | 固定閾値 / 進化で学習 | 中（Phase 2で決定） |
| C型のスロット名定義 | KGスキーマと同期 / 独立定義 | 低（Phase 3で決定） |
| LLM代替のスコープ | RAG層のみ / 推論層まで | 低（長期計画） |

---

## 10. 作業開始チェックリスト（Claude Code用）

```bash
# 1. 現状確認
cat CLAUDE.md
cat plan_trident.md   # このファイル

# 2. Phase 0 開始: ツール調査
pip install tensorneat qdax --quiet
python -c "import tensorneat; print(tensorneat.__version__)"
python -c "from qdax.core.map_elites import MAPElites; print('qdax ok')"

# 3. ES-HyperNEAT サンプル確認
# TensorNEAT のexamples/hyperneat/ を確認

# 4. WSL2 GPU 確認
python -c "import jax; print(jax.devices())"
```

---

*このplanはMED Framework TRIDENTサブシステムの基本設計です。*  
*Phase 0の調査結果に応じてツール選定を更新してください。*
