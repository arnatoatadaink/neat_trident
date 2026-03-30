# TRIDENT — 次セッション引き継ぎメモ

作成日: 2026-03-30
作業ブランチ: `claude/review-git-plan-uaNrY`

---

## 完了済み作業（このセッションまで）

| コミット | 内容 |
|---------|------|
| `84f9f58` | ES-HyperNEAT カスタム拡張（13/13 チェック通過） |
| `cb9159f` | 長期進化ループ 120iter × 9ms/iter（5/5 通過） |
| `2a95451` | faiss-cpu 統合 / HybridIndexer 3モード（9/9 通過） |

### 実装済みファイル

```
src/
├── interfaces/
│   ├── neat_indexer.py      # A型 + HybridIndexer
│   ├── neat_gate.py         # B型
│   └── neat_slot_filler.py  # C型
├── map_elites_archive.py    # TRIDENTArchive (MAP-Elites)
├── novelty_search.py        # NoveltyArchive (NS カスタム実装)
└── es_hyperneat.py          # ES-HyperNEAT カスタム拡張

scripts/
├── faiss_hybrid_verify.py   # faiss-cpu + HybridIndexer 統合検証
├── long_term_loop.py        # 長期進化ループ検証
└── es_hyperneat_verify.py   # ES-HyperNEAT 検証
```

---

## 次セッションでやること: MED Framework 統合プロトタイプ

### 前提

- MED の git リポジトリをセッションに追加済みの状態で開始
- まず **MED の FAISS インデクサ実装を CLI で確認** してから設計着手
- 統合テストは **保留**（メモリ成熟作業が進行中のため）
- 目標: **「I/F 先行設計」** — TRIDENT 側の MED アダプタ層を設計・実装

### 作業手順

#### Step 1: MED の FAISS I/F 確認（CLI で実施）

確認すべき箇所:
```
- FAISS インデックスの構築方法（IndexFlat* / IVF 等）
- search() の引数・返り値シグネチャ
- コーパスベクトルの次元・正規化方針
- MED 内でインデクサを呼ぶモジュール・クラス名
```

#### Step 2: アダプタ層の設計・実装

```
neat_trident/src/med_integration/
├── interfaces.py        # MED側に期待する抽象 I/F (typing.Protocol)
├── trident_adapter.py   # TRIDENT → MED アダプタ
└── stub_med.py          # MED スタブ (後で実装に差し替え)
```

**`interfaces.py` の骨格例:**
```python
from typing import Protocol
import numpy as np

class MEDIndexerProtocol(Protocol):
    """MED の FAISS インデクサに期待するインターフェース"""
    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]: ...
    def add(self, vectors: np.ndarray) -> None: ...

class MEDSkillStoreProtocol(Protocol):
    """MED のスキルストアに期待するインターフェース"""
    def get_skill(self, skill_id: int) -> dict: ...
    def store_skill(self, skill: dict) -> int: ...
```

**`trident_adapter.py` の役割:**
- `HybridIndexer` を `MEDIndexerProtocol` に適合させる
- A/B/C 型スキルを MED のスキルストア形式に変換
- MED の既存 FAISS を `HybridIndexer(mode="hybrid")` にスワップイン

#### Step 3: スタブ検証スクリプト

```
scripts/med_integration_verify.py
```

- `stub_med.py` を使って MED 統合フローをシミュレーション
- MED 実装差し替え後も同じスクリプトで検証可能な設計にする

---

## 技術メモ（次セッションで参照）

### NeatIndexer の使い方（基本）

```python
from src.interfaces.neat_indexer import NeatIndexer, HybridIndexer
import faiss

# 学習
neat = NeatIndexer(input_dim=384, pop_size=50, generation_limit=100)
neat.fit(corpus)  # corpus: (C, 384) np.ndarray

# FAISS と統合 (hybrid モード)
faiss_index = faiss.IndexFlatIP(384)
faiss_index.add(corpus_normalized)

hi = HybridIndexer(neat_indexer=neat, faiss_index=faiss_index, mode="hybrid")
indices, scores = hi.search(query, k=10)
```

### パフォーマンス注意点

- `NeatIndexer.search()` は内部で `transform()` を毎回呼ぶ → 遅い
- バッチ推論には `algo.transform()` を事前キャッシュ + `jax.jit + vmap` が必要
  → `scripts/long_term_loop.py` の実装パターンを参照

### 現在の制限

- GPU 未認識（CPU のみ / WSL2 + CUDA は別途対応）
- ES-HyperNEAT は TensorNEAT 0.1.0 向けパッチ済み (`src/es_hyperneat.py`)
- faiss-cpu 1.13.2 インストール済み

---

## 環境確認コマンド

```bash
# 環境確認
python scripts/phase0_verify.py

# 統合検証一覧
python scripts/faiss_hybrid_verify.py   # HybridIndexer (9/9)
python scripts/es_hyperneat_verify.py   # ES-HyperNEAT (13/13)
python scripts/long_term_loop.py        # 長期進化ループ (5/5)
```
