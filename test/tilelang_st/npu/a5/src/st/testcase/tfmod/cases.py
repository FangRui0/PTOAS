#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You can not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

"""Single source of truth for tfmod ST test cases.

Each case defines:
  - name:        case identifier
  - dtype:       numpy dtype (np.float32)
  - dst_tile:    (rows, cols) — dst tile buffer dimensions
  - src0_tile:   (rows, cols) — src0 tile buffer dimensions
  - src1_tile:   (rows, cols) — src1 tile buffer dimensions
  - valid_shape: (valid_rows, valid_cols) — effective computation region
  - eps:         tolerance for numpy.allclose (atol and rtol)

Note: src0/src1/dst tile buffer physical sizes can differ,
      but valid_shape must be the same for all.
"""

import numpy as np

CASES = [
    {
        "name": "f32_16x64_16x128_16x128_16x64",
        "dtype": np.float32,
        "dst_tile": (16, 64),
        "src0_tile": (16, 128),
        "src1_tile": (16, 128),
        "valid_shape": (16, 64),
        "eps": 1e-3,
    },
    {
        "name": "f32_16x32_16x64_16x32_16x32",
        "dtype": np.float32,
        "dst_tile": (16, 32),
        "src0_tile": (16, 64),
        "src1_tile": (16, 32),
        "valid_shape": (16, 32),
        "eps": 1e-3,
    },
    {
        "name": "f32_16x64_16x128_16x128_16x63",
        "dtype": np.float32,
        "dst_tile": (16, 64),
        "src0_tile": (16, 128),
        "src1_tile": (16, 128),
        "valid_shape": (16, 63),
        "eps": 1e-3,
    },
    {
        "name": "f32_2x32_2x64_2x32_2x31",
        "dtype": np.float32,
        "dst_tile": (2, 32),
        "src0_tile": (2, 64),
        "src1_tile": (2, 32),
        "valid_shape": (2, 31),
        "eps": 1e-3,
    },
    {
        "name": "f32_1x8192_1x8192_1x8192_1x8192",
        "dtype": np.float32,
        "dst_tile": (1, 8192),
        "src0_tile": (1, 8192),
        "src1_tile": (1, 8192),
        "valid_shape": (1, 8192),
        "eps": 1e-3,
    },
]