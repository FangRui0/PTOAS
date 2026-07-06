DeepSeek V4 decode PTO kernels for A3, generated from `hw-native-sys/pypto-lib` `models/deepseek/v4` at commit `ccbdc4fa5cafd1eda7784c9585f9dc876791778b`.

Scope:
- compile-regression inputs for `ptoas`
- board-validation inputs with per-case custom golden wrappers

Notes:
- The vendored sample set in this directory comes from the `pypto-lib` export at `ccbdc4fa5cafd1eda7784c9585f9dc876791778b`, which produced 54 raw `.pto` fragments for this arch.
- This directory vendors the 27 fragments from that export that compile on rebased `PTOAS` `main` with `--pto-level=level3 --pto-arch=a3`.
- As of `pypto-lib/main` `628dcf68d3776e61341d09088063e426cd5ef69c`, the A3 DeepSeek V4 decode export has evolved into a larger 72-kernel topology, so updating beyond this pinned sample set needs a separate sample/golden migration instead of a straight raw file refresh.
- Raw kernels emitted under `aic/` or `aiv/` are flattened to top-level sample files via `<section>_<kernel>.pto` naming.
- Each current fragment has a sibling `<case>_golden.py`; shared reference logic lives in `deepseek_v4_decode_golden_lib.py`.
- The shared helper generates deterministic inputs only; board-validation falls back to first-run output capture when no `golden_*.bin` is emitted.
- `runop.sh` defaults these cases to `--pto-level=level3` and skips the A3/A5 directory on the opposite arch.

Latest export fragments not yet vendored:
- `aiv/csa_cmp_rope.pto`
- `aiv/csa_rope_step.pto`
- `aiv/csa_sparse_idx_tile.pto`
- `aiv/gather_kv.pto`
- `aiv/hca_overlay_topk.pto`
- `aiv/hca_rope.pto`
- `aiv/kv_and_cache_write.pto`
- `aiv/kv_and_cache_write_0.pto`
- `aiv/kv_finalize.pto`
- `aiv/kv_rope_fused.pto`
- `aiv/q_head_rope_fused.pto`
- `aiv/qr_rope.pto`
- `aiv/rmsnorm_rope.pto`
- `aiv/rmsnorm_rope_0.pto`
- `aiv/rope.pto`
- `aiv/softmax_pool.pto`
- `aiv/softmax_pool_0.pto`
- `aiv/state_scatter_paged.pto`
- `aiv/state_scatter_paged_0.pto`
- `aiv/state_scatter_pre.pto`
- `aiv/swa_overlay_topk.pto`
- `aiv/topk.pto`
- `linear.pto`
- `proj_a.pto`
- `proj_b.pto`
- `qproj.pto`
- `qr_proj.pto`
