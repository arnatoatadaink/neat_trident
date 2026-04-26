---
paths: .git/**
---
- Do not commit without running `poetry run pytest tests/ -q` first
- Message format: `<type>(<scope>): <what>` (e.g. `feat(indexer): add HybridIndexer hybrid mode`)
- Types: feat / fix / test / refactor / docs / chore
- Each commit should cover one logical change; do not bundle unrelated fixes
