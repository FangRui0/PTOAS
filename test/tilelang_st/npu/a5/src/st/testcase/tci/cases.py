#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import numpy as np

# Cases cover the 1D contiguous-integer sequence semantics of pto.tci. The
# shape choices exercise the corner cases of the template's column-chunked
# vector store:
#
#   * tiny shapes (partial mask on the first chunk)
#   * exact single-VReg multiples (mask covers the full lane count, no tail)
#   * one-element tails just past a full VReg (mask of 1 lane on the tail)
#   * multi-VReg shapes (two full chunks, no partial tail)
#
# For i32 the vector lane count is 64; for i16 it is 128.
CASES = [
    # ---- i32 (lanes = 64) ----
    {
        "name": "i32_1x8",
        "dtype": np.int32,
        "shape": (1, 8),
        "valid_shape": (1, 8),
        "start": -5,
        "eps": 0.0,
    },
    {
        "name": "i32_1x32",
        "dtype": np.int32,
        "shape": (1, 32),
        "valid_shape": (1, 32),
        "start": 3,
        "eps": 0.0,
    },
    {
        "name": "i32_1x64",
        "dtype": np.int32,
        "shape": (1, 64),
        "valid_shape": (1, 64),
        "start": 100,
        "eps": 0.0,
    },
    {
        "name": "i32_1x72",
        "dtype": np.int32,
        "shape": (1, 72),
        "valid_shape": (1, 72),
        "start": 0,
        "eps": 0.0,
    },
    {
        "name": "i32_1x80",
        "dtype": np.int32,
        "shape": (1, 80),
        "valid_shape": (1, 80),
        "start": 17,
        "eps": 0.0,
    },
    {
        "name": "i32_1x128",
        "dtype": np.int32,
        "shape": (1, 128),
        "valid_shape": (1, 128),
        "start": -1000,
        "eps": 0.0,
    },
    # ---- i16 (lanes = 128) ----
    {
        "name": "i16_1x16",
        "dtype": np.int16,
        "shape": (1, 16),
        "valid_shape": (1, 16),
        "start": 1000,
        "eps": 0.0,
    },
    {
        "name": "i16_1x64",
        "dtype": np.int16,
        "shape": (1, 64),
        "valid_shape": (1, 64),
        "start": 11,
        "eps": 0.0,
    },
    {
        "name": "i16_1x128",
        "dtype": np.int16,
        "shape": (1, 128),
        "valid_shape": (1, 128),
        "start": -100,
        "eps": 0.0,
    },
    {
        "name": "i16_1x144",
        "dtype": np.int16,
        "shape": (1, 144),
        "valid_shape": (1, 144),
        "start": 0,
        "eps": 0.0,
    },
    {
        "name": "i16_1x160",
        "dtype": np.int16,
        "shape": (1, 160),
        "valid_shape": (1, 160),
        "start": -23,
        "eps": 0.0,
    },
    {
        "name": "i16_1x256",
        "dtype": np.int16,
        "shape": (1, 256),
        "valid_shape": (1, 256),
        "start": 30000,
        "eps": 0.0,
    },
]
