# Phase 0 調査結果レポート

> 実施日: 2026-03-28
> ステータス: 完了

---

## 1. 環境

| 項目 | 結果 |
|------|------|
| Python | 3.11 |
| JAX | 0.9.2 |
| デバイス | CPU only (`[CpuDevice(id=0)]`) |
| GPU (CUDA) | 未認識 → 要WSL2 + CUDA設定 |

---

## 2. ツール調査結果

### TensorNEAT

```
インストール方法: pip install git+https://github.com/EMI-Group/tensorneat.git
バージョン: 0.1.0
状態: ✅ インストール成功
```

**利用可能コンポーネント:**
- `tensorneat.Pipeline` — メインパイプライン
- `tensorneat.algorithm.hyperneat.HyperNEAT` — **HyperNEAT 対応済み** ✅
- `tensorneat.genome.DefaultGenome` — 標準ゲノム
- `tensorneat.genome.RecurrentGenome` — リカレントゲノム

**ES-HyperNEAT の状況:**
- 専用クラス `ESHyperNEAT` は存在しない ⚠️
- `HyperNEAT` クラスが提供されており、Substrate (MLP) ベースのトポロジ探索は可能
- ES-HyperNEAT は現バージョン (0.1.0) では未実装 → **カスタム実装が必要**

### QDax

```
インストール方法: pip install qdax
状態: ✅ インストール成功
```

- `qdax.core.map_elites.MAPElites` — MAP-Elites ✅
- JAX ネイティブ、GPU並列対応
- **採用決定: MAP-Elites の実装として QDax を使用**

### evosax

```
インストール方法: pip install evosax
バージョン: 0.2.0
状態: ✅ インストール成功（ただし NS 実装なし）
```

- Novelty Search (NS) の専用実装は evosax 0.2.0 には含まれない ⚠️
- Evolution Strategies (CMA-ES, NES 等) の実装あり
- **判定: NS はカスタム実装が必要**

### neat-python

- 不採用（TensorNEAT で代替）

---

## 3. 判断マトリクス（更新版）

| ツール | JAX統合 | GPU並列 | NS対応 | MAP-Elites | 判定 |
|--------|---------|---------|--------|------------|------|
| TensorNEAT | ✅ | ✅ | ❌ | ❌ | **採用**（NEAT/HyperNEAT）|
| QDax | ✅ | ✅ | ⚠️ | ✅ | **採用**（MAP-Elites）|
| evosax | ✅ | ✅ | ❌ | ❌ | **参考のみ**（NS実装参考）|
| neat-python | ❌ | ❌ | ❌ | ❌ | **不採用** |

---

## 4. 未決事項の解決（Phase 0 で決定）

| 項目 | 決定内容 |
|------|---------|
| **NS 実装** | **カスタム実装**（JAX + QDax のアーカイブと統合）|
| **MAP-Elites** | **QDax 採用** |
| **ES-HyperNEAT** | TensorNEAT の `HyperNEAT` をベースに **カスタム拡張** |
| **GPU** | 現在 CPU のみ — WSL2 + CUDA 設定が必要（後続タスク）|

---

## 5. Phase 1 への引き継ぎ事項

1. GPU 環境整備（WSL2 CUDA）は並行タスクとして継続
2. A型 I/F (`NeatIndexer`) の実装は CPU でも開始可能
3. NS カスタム実装の設計は Phase 1 冒頭で着手
4. TensorNEAT `HyperNEAT` の Substrate 設計を FAISSの 384次元に合わせる

---

## 6. 動作確認スクリプト

`scripts/phase0_verify.py` を参照。

