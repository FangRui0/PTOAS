DeepSeek V4 decode PTO kernels for A3, generated from `hw-native-sys/pypto-lib` `models/deepseek/v4/decode_fwd.py` at commit `d3340a080dec891d6ba71425b934bfaadd6d2371`.

Scope:
- compile-regression inputs for `ptoas`
- board-validation inputs with per-case custom golden wrappers

Notes:
- Export command: `PYTHONPATH=/tmp/pypto-lib-pr799 python3 /tmp/pypto-lib-pr799/models/deepseek/v4/decode_fwd.py --compile-only -p a2a3`.
- The latest successful upstream export wrote 299 raw `.pto` fragments under `build_output/_jit_l3_decode_fwd_*/next_levels/decode_fwd/kernels/`.
- This directory vendors 80 representative kernel families from that export, using the unsuffixed fragment when a family also has repeated `_N` specializations.
- All 80 representative families compile on rebased `PTOAS` `main` with `--pto-level=level3 --pto-arch=a3`.
- Raw kernels emitted under `aic/` or `aiv/` are flattened to top-level sample files via `<section>_<kernel>.pto` naming; top-level kernels keep their original names.
- Each vendored fragment has a sibling `<case>_golden.py`; shared reference logic lives in `deepseek_v4_decode_golden_lib.py`.
- The shared helper generates deterministic inputs only; board-validation falls back to first-run output capture when no `golden_*.bin` is emitted.
- `runop.sh` defaults these cases to `--pto-level=level3` and skips the A3 directory on non-A3 targets.
