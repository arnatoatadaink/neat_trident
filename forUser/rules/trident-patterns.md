# TRIDENT 実装パターン（.claude/rules/trident-patterns.md の日本語版）

対象パス: `src/**/*.py`

- NEAT インターフェースは `BaseProblem` を継承。`evaluate()` が JAX 互換なら `jitable = True` を設定
- HybridIndexer モード: `"faiss"` | `"neat"` | `"hybrid"` — `__init__` でバリデーション必須
- AssociationFn の重みは更新後も合計 1.0 を保つ（正規化を常に適用）
- MED Protocol 準拠: `search()` は `list[tuple[str, float]]` を返す、`add()` は `doc_ids: list[str]` を受け取る
- TRIDENTMEDAdapter: 大量登録は `sync_indexer()` を使う（`add()` を繰り返さない）
- GPU 上での重い NEAT ループ: `jax.jit(...).lower(state).compile()` で AOT コンパイルする
