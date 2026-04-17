// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include <stdint.h>

#ifndef AICORE
#define AICORE [aicore]
#endif

extern "C" __global__ AICORE void TCI_i32_1x8(int32_t start, __gm__ int32_t *dst);
extern "C" __global__ AICORE void TCI_i32_1x32(int32_t start, __gm__ int32_t *dst);
extern "C" __global__ AICORE void TCI_i32_1x64(int32_t start, __gm__ int32_t *dst);
extern "C" __global__ AICORE void TCI_i32_1x72(int32_t start, __gm__ int32_t *dst);
extern "C" __global__ AICORE void TCI_i32_1x80(int32_t start, __gm__ int32_t *dst);
extern "C" __global__ AICORE void TCI_i32_1x128(int32_t start, __gm__ int32_t *dst);
extern "C" __global__ AICORE void TCI_i16_1x16(int16_t start, __gm__ int16_t *dst);
extern "C" __global__ AICORE void TCI_i16_1x64(int16_t start, __gm__ int16_t *dst);
extern "C" __global__ AICORE void TCI_i16_1x128(int16_t start, __gm__ int16_t *dst);
extern "C" __global__ AICORE void TCI_i16_1x144(int16_t start, __gm__ int16_t *dst);
extern "C" __global__ AICORE void TCI_i16_1x160(int16_t start, __gm__ int16_t *dst);
extern "C" __global__ AICORE void TCI_i16_1x256(int16_t start, __gm__ int16_t *dst);

#define DEFINE_LAUNCH_I32(name)                                                 \
    void Launch##name(const void *start, void *dst, void *stream) {             \
        const int32_t scalar = *reinterpret_cast<const int32_t *>(start);       \
        name<<<1, nullptr, stream>>>(scalar, (__gm__ int32_t *)dst);            \
    }

#define DEFINE_LAUNCH_I16(name)                                                 \
    void Launch##name(const void *start, void *dst, void *stream) {             \
        const int16_t scalar = *reinterpret_cast<const int16_t *>(start);       \
        name<<<1, nullptr, stream>>>(scalar, (__gm__ int16_t *)dst);            \
    }

DEFINE_LAUNCH_I32(TCI_i32_1x8)
DEFINE_LAUNCH_I32(TCI_i32_1x32)
DEFINE_LAUNCH_I32(TCI_i32_1x64)
DEFINE_LAUNCH_I32(TCI_i32_1x72)
DEFINE_LAUNCH_I32(TCI_i32_1x80)
DEFINE_LAUNCH_I32(TCI_i32_1x128)

DEFINE_LAUNCH_I16(TCI_i16_1x16)
DEFINE_LAUNCH_I16(TCI_i16_1x64)
DEFINE_LAUNCH_I16(TCI_i16_1x128)
DEFINE_LAUNCH_I16(TCI_i16_1x144)
DEFINE_LAUNCH_I16(TCI_i16_1x160)
DEFINE_LAUNCH_I16(TCI_i16_1x256)
