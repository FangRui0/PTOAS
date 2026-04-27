#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

"""Single source of truth for tadd ST test cases.

Each case defines:
  - name:        case identifier, used as subdirectory name and by main.cpp kCases[].
  - dtype:       numpy dtype (e.g. np.float32).
  - dst_tile:    (rows, cols) — dst tile buffer dimensions.
  - src0_tile:   (rows, cols) — src0 tile buffer dimensions.
  - src1_tile:   (rows, cols) — src1 tile buffer dimensions.
  - valid_shape: (valid_rows, valid_cols) — effective computation region.
  - eps:         tolerance for numpy.allclose (atol and rtol).

Note: src0/src1/dst tile buffer physical sizes can differ,
      but valid_shape must be the same for all.

gen_data.py and compare.py both import this list to avoid redundant definitions.
"""

import numpy as np

CASES = [
    {
        "name": "f32_64x128_64x128_64x128_64x128",
        "dtype": np.float32,
        "dst_tile": (64, 128),
        "src0_tile": (64, 128),
        "src1_tile": (64, 128),
        "valid_shape": (64, 128),
        "eps": 1e-6,
    },
    {
        "name": "f32_16x64_16x64_16x64_16x64",
        "dtype": np.float32,
        "dst_tile": (16, 64),
        "src0_tile": (16, 64),
        "src1_tile": (16, 64),
        "valid_shape": (16, 64),
        "eps": 1e-6,
    },
    {
        "name": "f32_32x32_32x32_32x32_32x32",
        "dtype": np.float32,
        "dst_tile": (32, 32),
        "src0_tile": (32, 32),
        "src1_tile": (32, 32),
        "valid_shape": (32, 32),
        "eps": 1e-6,
    },
    {
        "name": "f32_64x64_64x64_64x64_64x64",
        "dtype": np.float32,
        "dst_tile": (64, 64),
        "src0_tile": (64, 64),
        "src1_tile": (64, 64),
        "valid_shape": (64, 64),
        "eps": 1e-6,
    },
    {
        "name": "i32_64x64_64x64_64x64_64x64",
        "dtype": np.int32,
        "dst_tile": (64, 64),
        "src0_tile": (64, 64),
        "src1_tile": (64, 64),
        "valid_shape": (64, 64),
        "eps": 0,
    },
    {
        "name": "i16_64x64_64x64_64x64_64x64",
        "dtype": np.int16,
        "dst_tile": (64, 64),
        "src0_tile": (64, 64),
        "src1_tile": (64, 64),
        "valid_shape": (64, 64),
        "eps": 0,
    },
    {
        "name": "f16_16x256_16x256_16x256_16x256",
        "dtype": np.float16,
        "dst_tile": (16, 256),
        "src0_tile": (16, 256),
        "src1_tile": (16, 256),
        "valid_shape": (16, 256),
        "eps": 1e-3,
    },
    {
        "name": "half_16x64_16x128_16x128_16x64",
        "dtype": np.float16,
        "dst_tile": (16, 64),
        "src0_tile": (16, 128),
        "src1_tile": (16, 128),
        "valid_shape": (16, 64),
        "eps": 1e-3,
    },
]
