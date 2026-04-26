---
paths: src/**/*.py
---
- NEAT interfaces must inherit from BaseProblem; set `jitable = True` when evaluate() is JAX-compatible
- HybridIndexer modes: `"faiss"` | `"neat"` | `"hybrid"` — always validate mode in __init__
- AssociationFn weights must sum to 1.0 after any update (enforce via normalization)
- MED Protocol compliance: search() returns `list[tuple[str, float]]`, add() takes `doc_ids: list[str]`
- TRIDENTMEDAdapter: use `sync_indexer()` for bulk registration, not repeated `add()` calls
- JAX on GPU: wrap heavy NEAT loops with `jax.jit(...).lower(state).compile()` for AOT compilation
