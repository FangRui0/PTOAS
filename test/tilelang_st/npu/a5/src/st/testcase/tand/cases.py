#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You can not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

"""Single source of truth for tand ST test cases.

Each case defines:
  - name:        case identifier
  - dtype:       numpy dtype (np.int16, np.int8)
  - shape:       (rows, cols) — allocated tile dimensions
  - valid_shape: (valid_rows, valid_cols) — effective computation region
  - eps:         tolerance for numpy.allclose (atol and rtol)
"""

import numpy as np

CASES = [
    {
        "name": "i16_64x64",
        "dtype": np.int16,
        "shape": (64, 64),
        "valid_shape": (64, 64),
        "eps": 0,
    },
    {
        "name": "i8_64x64_valid63x63",
        "dtype": np.int8,
        "shape": (64, 64),
        "valid_shape": (63, 63),
        "eps": 0,
    },
]