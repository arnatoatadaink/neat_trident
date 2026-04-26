# コミットルール（.claude/rules/commit.md の日本語版）

対象パス: `.git/**`

- コミット前に `poetry run pytest tests/ -q` を必ず実行する
- メッセージ形式: `<type>(<scope>): <what>` (例: `feat(indexer): HybridIndexer hybrid モード追加`)
- type: feat / fix / test / refactor / docs / chore
- 1 コミット = 1 論理変更。無関係な修正をまとめない
