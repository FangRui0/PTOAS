#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You can not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

"""Single source of truth for txor ST test cases.

Each case defines:
  - name:        case identifier
  - dtype:       numpy dtype (np.int16, np.int8)
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
        "name": "i16_64x64_64x64_64x64_64x64",
        "dtype": np.int16,
        "dst_tile": (64, 64),
        "src0_tile": (64, 64),
        "src1_tile": (64, 64),
        "valid_shape": (64, 64),
        "eps": 0,
    },
    {
        "name": "i16_32x128_32x128_32x256_32x128",
        "dtype": np.int16,
        "dst_tile": (32, 128),
        "src0_tile": (32, 128),
        "src1_tile": (32, 256),
        "valid_shape": (32, 128),
        "eps": 0,
    },
    {
        "name": "i16_32x128_32x128_32x256_32x127",
        "dtype": np.int16,
        "dst_tile": (32, 128),
        "src0_tile": (32, 128),
        "src1_tile": (32, 256),
        "valid_shape": (32, 127),
        "eps": 0,
    },
    {
        "name": "i8_32x128_32x128_32x256_32x127",
        "dtype": np.int8,
        "dst_tile": (32, 128),
        "src0_tile": (32, 128),
        "src1_tile": (32, 256),
        "valid_shape": (32, 127),
        "eps": 0,
    },
]