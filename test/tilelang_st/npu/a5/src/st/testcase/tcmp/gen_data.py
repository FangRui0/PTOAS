#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

import numpy as np
from cases import CASES
from st_common import validate_cases, setup_case_rng, save_case_data

validate_cases(CASES)


def compute_cmp(a, b, mode):
    if mode == "gt":
        return (a > b)
    elif mode == "ge":
        return (a >= b)
    elif mode == "lt":
        return (a < b)
    elif mode == "le":
        return (a <= b)
    elif mode == "eq":
        return (a == b)
    elif mode == "ne":
        return (a != b)
    else:
        raise ValueError(f"Unknown cmp_mode: {mode}")


ALIGN_STRIDE = 32


def pack_predicate_mask(cmp_result):
    cmp_result = cmp_result.astype(np.uint8)
    shape = cmp_result.shape
    packed_shape = (shape[0], ALIGN_STRIDE)
    packed = np.zeros(packed_shape, dtype=np.uint8)
    for row in range(shape[0]):
        for vl in range(min(8, shape[1] // 8)):
            lanes = cmp_result[row, vl*8:(vl+1)*8]
            for j in range(8):
                if lanes[j]:
                    byte_idx = j // 2
                    bit_pos = (j % 2) * 4
                    packed[row, vl*4 + byte_idx] |= (1 << bit_pos)
    return packed.view(np.int8)


for case in CASES:
    setup_case_rng(case)

    dtype = case["dtype"]
    out_dtype = case["out_dtype"]
    shape = case["shape"]
    valid_shape = case["valid_shape"]
    cmp_mode = case["cmp_mode"]

    input1 = np.random.choice([-5, -2, -1, 0, 1, 2, 5], size=shape).astype(dtype)
    input2 = np.random.choice([-5, -2, -1, 0, 1, 2, 5], size=shape).astype(dtype)

    vr, vc = valid_shape
    cmp_result = compute_cmp(input1[:vr, :vc], input2[:vr, :vc], cmp_mode)
    golden = pack_predicate_mask(cmp_result)

    save_case_data(case["name"], {"input1": input1, "input2": input2, "golden": golden})
    print(f"[INFO] gen_data: {case['name']} shape={shape} valid_shape={valid_shape} dtype={dtype.__name__} out_dtype={out_dtype.__name__} cmp_mode={cmp_mode}")