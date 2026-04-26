# テストルール（.claude/rules/testing.md の日本語版）

対象パス: `tests/**/*.py`

- テスト名: `test_<何を>_<条件>` (例: `test_score_no_context`)
- 速度のため小さなパラメータを固定: DIM≤32, POP≤10, GEN≤5
- モックは最小化。実際の JAX/numpy 計算を優先
- 3つ以上のテストファイルで共有するフィクスチャは `conftest.py` に集約
- JAX JIT テスト: numpy アサーション前に必ず `jax.device_get()` を呼ぶ
- 確率的テスト: シードを固定する (`jax.random.PRNGKey(0)`, `np.random.default_rng(0)`)
