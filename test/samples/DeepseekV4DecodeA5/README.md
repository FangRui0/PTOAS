DeepSeek V4 decode PTO kernels for A5, aligned to the latest `hw-native-sys/pypto-lib` `models/deepseek/v4/decode_fwd.py` topology at commit `d3340a080dec891d6ba71425b934bfaadd6d2371`.

Scope:
- compile-regression inputs for `ptoas`
- board-validation inputs with per-case custom golden wrappers

Notes:
- Direct latest-main export command `PYTHONPATH=/tmp/pypto-lib-pr799 python3 /tmp/pypto-lib-pr799/models/deepseek/v4/decode_fwd.py --compile-only -p a5` currently fails upstream before kernel emission on `hc_pre_fused` with `HardSyncallOccupancy`, so this directory mirrors the latest successful raw topology exported with `-p a2a3` instead of claiming a direct A5 dump.
- That latest successful upstream export wrote 299 raw `.pto` fragments under `build_output/_jit_l3_decode_fwd_*/next_levels/decode_fwd/kernels/` and collapses to 80 representative kernel families by the same unsuffixed-fragment rule used for A3.
- This directory vendors the 66 representative families from that latest raw topology that compile on rebased `PTOAS` `main` with `--pto-level=level3 --pto-arch=a5`.
- Raw kernels emitted under `aic/` or `aiv/` are flattened to top-level sample files via `<section>_<kernel>.pto` naming; top-level kernels keep their original names.
- Each vendored fragment has a sibling `<case>_golden.py`; shared reference logic lives in `deepseek_v4_decode_golden_lib.py`.
- The shared helper generates deterministic inputs only; board-validation falls back to first-run output capture when no `golden_*.bin` is emitted.
- `runop.sh` defaults these cases to `--pto-level=level3` and skips the A5 directory on non-A5 targets.

Latest representative families excluded on A5 because they fail `ptoas --pto-arch=a5` today:
- `aic_exp_gate_mm`
- `aic_exp_up_mm`
- `aic_exp_w2_mm`
- `aic_kv_proj_matmul`
- `aic_kv_score_proj`
- `aic_proj_a_mm`
- `aic_proj_b_mm`
- `aic_qproj_matmul`
- `aic_qr_proj_matmul`
- `aic_sh_gate_mm`
- `aic_sh_up_mm`
- `aic_sh_w2_mm`
- `qk_pv`
- `weights_proj`
