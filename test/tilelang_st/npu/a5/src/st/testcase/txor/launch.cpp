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

// Case 0: i16, dst=64x64, src0=64x64, src1=64x64, valid=64x64
extern "C" __global__ AICORE void TXOR_i16_64x64_64x64_64x64_64x64(__gm__ int16_t *a, __gm__ int16_t *b, __gm__ int16_t *c);

void LaunchTXOR_i16_64x64_64x64_64x64_64x64(void *a, void *b, void *c, void *stream) {
    TXOR_i16_64x64_64x64_64x64_64x64<<<1, nullptr, stream>>>((__gm__ int16_t *)a, (__gm__ int16_t *)b, (__gm__ int16_t *)c);
}

// Case 1: i16, dst=32x128, src0=32x128, src1=32x256, valid=32x128
extern "C" __global__ AICORE void TXOR_i16_32x128_32x128_32x256_32x128(__gm__ int16_t *a, __gm__ int16_t *b, __gm__ int16_t *c);

void LaunchTXOR_i16_32x128_32x128_32x256_32x128(void *a, void *b, void *c, void *stream) {
    TXOR_i16_32x128_32x128_32x256_32x128<<<1, nullptr, stream>>>((__gm__ int16_t *)a, (__gm__ int16_t *)b, (__gm__ int16_t *)c);
}

// Case 2: i16, dst=32x128, src0=32x128, src1=32x256, valid=32x127
extern "C" __global__ AICORE void TXOR_i16_32x128_32x128_32x256_32x127(__gm__ int16_t *a, __gm__ int16_t *b, __gm__ int16_t *c);

void LaunchTXOR_i16_32x128_32x128_32x256_32x127(void *a, void *b, void *c, void *stream) {
    TXOR_i16_32x128_32x128_32x256_32x127<<<1, nullptr, stream>>>((__gm__ int16_t *)a, (__gm__ int16_t *)b, (__gm__ int16_t *)c);
}

// Case 3: i8, dst=32x128, src0=32x128, src1=32x256, valid=32x127
extern "C" __global__ AICORE void TXOR_i8_32x128_32x128_32x256_32x127(__gm__ int8_t *a, __gm__ int8_t *b, __gm__ int8_t *c);

void LaunchTXOR_i8_32x128_32x128_32x256_32x127(void *a, void *b, void *c, void *stream) {
    TXOR_i8_32x128_32x128_32x256_32x127<<<1, nullptr, stream>>>((__gm__ int8_t *)a, (__gm__ int8_t *)b, (__gm__ int8_t *)c);
}