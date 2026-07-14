DeepSeek V4 decode PTO kernels for A3, generated from `hw-native-sys/pypto-lib` `models/deepseek/v4/decode_fwd.py` at commit `402d12a61dfd3f415ca9ec5356f9e7ff876b6ad8`, using `hw-native-sys/pypto` commit `0f15ec140f1112392584e4d8f2d95e2b723c0471`.

Scope:
- compile-regression inputs for `ptoas`
- board-validation inputs with per-case custom golden wrappers

Notes:
- Export command: `python3 models/deepseek/v4/decode_fwd.py --compile-only -p a2a3` with `PYPTO_PROG_BUILD_DIR` set to an isolated output directory.
- The current upstream export writes 343 raw `.pto` fragments under `_jit_l3_decode_fwd_*/next_levels/decode_fwd/kernels/` and collapses to 85 representative kernel families when repeated `_N` specializations use the unsuffixed fragment.
- This directory vendors the 79 representative families that compile on `PTOAS` `main` `0251110abb4dfe73076a5fc38770fa4a00af54c9` with `--pto-level=level3 --pto-arch=a3`.
- Raw kernels emitted under `aic/` or `aiv/` are flattened to top-level sample files via `<section>_<kernel>.pto` naming; top-level kernels keep their original names.
- Each vendored fragment has a sibling `<case>_golden.py`; shared reference logic lives in `deepseek_v4_decode_golden_lib.py`.
- The shared helper generates deterministic inputs only; board-validation falls back to first-run output capture when no `golden_*.bin` is emitted.
- `runop.sh` defaults these cases to `--pto-level=level3` and skips the A3 directory on non-A3 targets.

Current upstream families not vendored:
- `aiv_combine`, `aiv_combine_wait`, `aiv_dispatch_meta`, `aiv_dispatch_push`, and `aiv_dispatch_wait` fail PTOAS memory-consistency validation because communication offset helpers have not been inlined; `aiv_dispatch_meta` also lacks the required GM cache invalidation after `twait`.
- `aiv_csa_slots_build_valid_qk_plan` did not finish PTOAS compilation within three minutes and is excluded to keep sample CI bounded.
