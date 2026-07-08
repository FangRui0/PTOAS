#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os
import re
import sys
from pathlib import Path

import numpy as np


ROWS_PER_BLOCK = 8
COLS = 256
EPS = 0.0001


def _load_i32_scalar(name: str, default: int) -> int:
    text = Path("main.cpp").read_text(encoding="utf-8")
    match = re.search(rf"\bint32_t\s+{re.escape(name)}\s*=\s*(-?\d+)\s*;", text)
    return int(match.group(1)) if match else int(default)


def _compare_valid_window(golden_path: str, output_path: str) -> bool:
    if not os.path.exists(output_path):
        print(f"[ERROR] Output missing: {output_path}")
        return False
    if not os.path.exists(golden_path):
        print(f"[ERROR] Golden missing: {golden_path}")
        return False

    golden = np.fromfile(golden_path, dtype=np.float32)
    output = np.fromfile(output_path, dtype=np.float32)
    if golden.shape != output.shape:
        print(f"[ERROR] Shape mismatch: {golden_path} {golden.shape} vs {output_path} {output.shape}")
        return False

    block_idx = _load_i32_scalar("v3", 1)
    start = block_idx * ROWS_PER_BLOCK * COLS
    count = ROWS_PER_BLOCK * COLS
    stop = start + count
    if golden.size < stop or output.size < stop:
        print(f"[ERROR] Valid window out of range: need [{start}:{stop}], got {golden.size}")
        return False

    golden_valid = golden[start:stop]
    output_valid = output[start:stop]
    if np.allclose(golden_valid, output_valid, atol=EPS, rtol=EPS, equal_nan=True):
        return True

    close = np.isclose(golden_valid, output_valid, atol=EPS, rtol=EPS, equal_nan=True)
    bad = np.nonzero(~close)[0]
    diff = np.abs(golden_valid.astype(np.float64) - output_valid.astype(np.float64))
    diff_for_report = np.where(np.isfinite(diff), diff, np.inf)
    rel_idx = int(bad[np.argmax(diff_for_report[bad])]) if bad.size else 0
    abs_idx = start + rel_idx
    row, col = divmod(abs_idx, COLS)
    print(
        f"[ERROR] Mismatch(valid window): {golden_path} vs {output_path}, "
        f"max diff={float(diff_for_report[rel_idx])} at idx={abs_idx}, row={row}, col={col} "
        f"(golden={float(golden_valid[rel_idx])}, out={float(output_valid[rel_idx])}, dtype=float32)"
    )
    return False


def main():
    ok = _compare_valid_window("golden_v2.bin", "v2.bin")
    if not ok:
        if os.getenv("COMPARE_STRICT", "1") != "0":
            print("[ERROR] compare failed")
            sys.exit(2)
        print("[WARN] compare failed (non-gating)")
        return False
    print("[INFO] compare passed")
    return True


if __name__ == "__main__":
    main()
