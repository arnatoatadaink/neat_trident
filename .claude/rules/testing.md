---
paths: tests/**/*.py
---
- Test naming: `test_<what>_<condition>` (e.g. `test_score_no_context`)
- Use small fixed parameters: DIM≤32, POP≤10, GEN≤5 to keep tests fast
- Minimize mocks; prefer real JAX/numpy computation
- Share fixtures via conftest.py when used across 3+ test files
- JAX JIT tests: always call `jax.device_get()` before numpy assertions
- For stochastic tests, fix seeds (e.g. `jax.random.PRNGKey(0)`, `np.random.default_rng(0)`)
