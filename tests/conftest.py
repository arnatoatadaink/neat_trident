import os

# Must run before any test module imports jax. The ROCm plugin hangs during
# device enumeration on this machine's heterogeneous dual-GPU (gfx1200+gfx1201)
# setup, and even when scoped to a single GPU it crashes with
# HIP_ERROR_InvalidValue on the second device->host transfer within a process
# (see docs/jax_rocm_fulltest_workaround_decision_20260714.md). Forcing the
# CPU backend avoids both failure modes; these are algorithm-correctness
# tests, not GPU-hardware verification, so CPU execution is sufficient.
# Override with `JAX_PLATFORMS=rocm poetry run pytest ...` to run against the
# GPU deliberately.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
