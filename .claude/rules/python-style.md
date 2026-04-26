---
paths: src/**/*.py,tests/**/*.py
---
- Type hints required on all public functions and class attributes
- No `Any` type unless unavoidable; document why if used
- Split file when it exceeds 300 lines
- No multi-line docstrings; one short comment only when the WHY is non-obvious
- JAX arrays: annotate with shape comments where non-trivial (e.g. `# (n, dim)`)
