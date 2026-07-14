Qwen3 decode PTO kernels for A3, generated from `pypto-lib/models/qwen3/32b/qwen3_32b_decode.py` at `402d12a61dfd3f415ca9ec5356f9e7ff876b6ad8`, using `hw-native-sys/pypto` commit `0f15ec140f1112392584e4d8f2d95e2b723c0471`.

Scope:
- compile-regression inputs for `ptoas`
- board-validation inputs with per-case custom golden

Notes:
- Current `pypto-lib/main` export for the A3 lowering maps to 14 raw `.pto` fragments.
- This directory vendors all 14 fragments, which compile on `PTOAS` `main` `0251110abb4dfe73076a5fc38770fa4a00af54c9` with `--pto-level=level3 --pto-arch=a3`.
- The vendored sample filenames keep the legacy `qwen3_decode_incore_*` compatibility layout where applicable, but the raw PTO contents come directly from the current upstream kernel names such as `q_proj`, `kv_proj`, `softmax`, `rope_kv_cache`, and `post_rmsnorm`.
- The upstream kernel topology changed from the old 17-case `qwen3_decode_incore_*` layout to a mixed set including `rmsnorm`, `rope_kv_cache`, `out_proj_residual`, `post_rmsnorm`, and `down_proj_residual`.
- `runop.sh` defaults these cases to `--pto-level=level3`.
- `runop.sh` skips this directory on A5 / Ascend950 targets.
- Each vendored fragment has a sibling `<case>_golden.py`; shared reference logic lives in `qwen3_decode_golden_lib.py`.
