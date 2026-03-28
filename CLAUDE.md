# TRIDENT — Claude Code 作業ガイド

## プロジェクト概要

TRIDENT は MED Framework 内で NEAT を主体とした自律的スキル探索・収集を担うサブシステムです。
詳細設計: `plan_trident.md`

## 現在のフェーズ

**Phase 0 完了 → Phase 1 (A型 NeatIndexer) 開始可能**

Phase 0 調査結果: `phase0_results.md`

## 技術スタック

| コンポーネント | ライブラリ | 状態 |
|-------------|----------|------|
| NEAT / HyperNEAT | TensorNEAT 0.1.0 | ✅ |
| MAP-Elites | QDax (JAX) | ✅ |
| Novelty Search | カスタム実装 (JAX) | 未実装 |
| ES-HyperNEAT | TensorNEAT HyperNEAT を拡張 | 未実装 |
| JAX | 0.9.2 | ✅ (CPU) |
| FAISS (MED既存) | faiss | 未インストール |

## ディレクトリ構成

```
neat_trident/
├── CLAUDE.md              # このファイル
├── plan_trident.md        # 設計プラン
├── phase0_results.md      # Phase 0 調査結果
├── scripts/
│   └── phase0_verify.py   # 環境確認スクリプト
└── src/                   # 実装 (Phase 1 以降)
    ├── interfaces/        # A/B/C 型 I/F
    │   ├── neat_indexer.py    # A型: float vector
    │   ├── neat_gate.py       # B型: binary activation
    │   └── neat_slot_filler.py # C型: named slots
    └── novelty_search.py  # NS カスタム実装
```

## 作業開始手順

```bash
# 環境確認
python scripts/phase0_verify.py

# Phase 1 開始
# → src/interfaces/neat_indexer.py を実装
```

## 注意事項

- GPU 未認識（CPU のみ）— WSL2 + CUDA 設定は別途対応
- ES-HyperNEAT: TensorNEAT 0.1.0 未対応 → `HyperNEAT` クラスをベースにカスタム実装
- NS: evosax に実装なし → JAX + QDax アーカイブと統合したカスタム実装
- FAISS は MED 統合時に必要 (Phase 1 で `pip install faiss-cpu`)
