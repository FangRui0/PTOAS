Qwen3 decode PTO kernels for A3, generated from `pypto-lib/models/qwen3/32b/qwen3_32b_decode.py` at `ccbdc4fa5cafd1eda7784c9585f9dc876791778b`.

Scope:
- compile-regression inputs for `ptoas`
- board-validation inputs with per-case custom golden

Notes:
- Current `pypto-lib/main` export for the A3 lowering maps to 14 raw `.pto` fragments.
- This directory vendors the 13 fragments that currently compile on `PTOAS` `main` with `--pto-level=level3`.
- Latest A3 `rope_kv_cache.pto` is intentionally omitted here for now because current PTO verification rejects its `pto.textract` source layout on A2/A3.
- The upstream kernel topology changed from the old 17-case `qwen3_decode_incore_*` layout to a mixed set including `rmsnorm`, `out_proj_residual`, `post_rmsnorm`, and `down_proj_residual`.
- `runop.sh` defaults these cases to `--pto-level=level3`.
- `runop.sh` skips this directory on A5 / Ascend950 targets.
- Each vendored fragment has a sibling `<case>_golden.py`; shared reference logic lives in `qwen3_decode_golden_lib.py`.
