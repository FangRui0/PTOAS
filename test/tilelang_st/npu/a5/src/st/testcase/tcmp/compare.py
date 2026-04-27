#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

import os
import sys
import numpy as np

from cases import CASES
from st_common import result_cmp, style_fail, style_pass, validate_cases


ALIGN_STRIDE = 32


def main():
    validate_cases(CASES)
    case_filter = sys.argv[1] if len(sys.argv) > 1 else None

    all_passed = True
    for case in CASES:
        if case_filter is not None and case["name"] != case_filter:
            continue

        case_dir = case["name"]
        shape = case["shape"]
        out_dtype = case["out_dtype"]
        vr, vc = case["valid_shape"]
        packed_shape = (vr, ALIGN_STRIDE)
        packed_size = vr * ALIGN_STRIDE

        golden = np.fromfile(os.path.join(case_dir, "golden.bin"), dtype=out_dtype).reshape(packed_shape)
        output_full = np.fromfile(os.path.join(case_dir, "output.bin"), dtype=out_dtype)
        output = output_full[:packed_size].reshape(packed_shape)

        ok = result_cmp(golden, output, case["eps"])
        if ok:
            print(style_pass(f"[INFO] {case['name']}: compare passed"))
        else:
            print(style_fail(f"[ERROR] {case['name']}: compare failed"))
            all_passed = False

    if not all_passed:
        sys.exit(2)
    print(style_pass("[INFO] all cases passed"))


if __name__ == "__main__":
    main()