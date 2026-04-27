#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You can not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

"""Single source of truth for tcmp ST test cases.

Each case defines:
  - name:        case identifier
  - dtype:       numpy dtype for inputs (np.float32 or np.int32)
  - out_dtype:   numpy dtype for output (np.int8)
  - shape:       (rows, cols) — tile buffer dimensions
  - valid_shape: (valid_rows, valid_cols) — effective computation region
  - eps:         tolerance for comparison
  - cmp_mode:    comparison mode string (eq, gt, le)
"""

import numpy as np

CASES = [
    {
        "name": "f32_1x64_eq",
        "dtype": np.float32,
        "out_dtype": np.int8,
        "shape": (1, 64),
        "valid_shape": (1, 64),
        "eps": 0,
        "cmp_mode": "eq",
    },
    {
        "name": "f32_8x64_gt",
        "dtype": np.float32,
        "out_dtype": np.int8,
        "shape": (8, 64),
        "valid_shape": (8, 64),
        "eps": 0,
        "cmp_mode": "gt",
    },
    {
        "name": "i32_16x32_eq",
        "dtype": np.int32,
        "out_dtype": np.int8,
        "shape": (16, 32),
        "valid_shape": (16, 32),
        "eps": 0,
        "cmp_mode": "eq",
    },
    {
        "name": "i32_32x32_eq",
        "dtype": np.int32,
        "out_dtype": np.int8,
        "shape": (32, 32),
        "valid_shape": (32, 32),
        "eps": 0,
        "cmp_mode": "eq",
    },
]