// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You can not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include <stdint.h>

#ifndef AICORE
#define AICORE [aicore]
#endif

// Case 0: f32 16x64
extern "C" __global__ AICORE void TDIV_f32_16x64(__gm__ float *a, __gm__ float *b, __gm__ float *c);

void LaunchTDIV_f32_16x64(void *a, void *b, void *c, void *stream) {
    TDIV_f32_16x64<<<1, nullptr, stream>>>((__gm__ float *)a, (__gm__ float *)b, (__gm__ float *)c);
}

// Case 1: f32 32x32
extern "C" __global__ AICORE void TDIV_f32_32x32(__gm__ float *a, __gm__ float *b, __gm__ float *c);

void LaunchTDIV_f32_32x32(void *a, void *b, void *c, void *stream) {
    TDIV_f32_32x32<<<1, nullptr, stream>>>((__gm__ float *)a, (__gm__ float *)b, (__gm__ float *)c);
}

// Case 2: f32 64x64
extern "C" __global__ AICORE void TDIV_f32_64x64(__gm__ float *a, __gm__ float *b, __gm__ float *c);

void LaunchTDIV_f32_64x64(void *a, void *b, void *c, void *stream) {
    TDIV_f32_64x64<<<1, nullptr, stream>>>((__gm__ float *)a, (__gm__ float *)b, (__gm__ float *)c);
}

// Case 3: f16 64x64
extern "C" __global__ AICORE void TDIV_f16_64x64(__gm__ half *a, __gm__ half *b, __gm__ half *c);

void LaunchTDIV_f16_64x64(void *a, void *b, void *c, void *stream) {
    TDIV_f16_64x64<<<1, nullptr, stream>>>((__gm__ half *)a, (__gm__ half *)b, (__gm__ half *)c);
}

// Case 4: f16 64x64 v61x61
extern "C" __global__ AICORE void TDIV_f16_64x64_v61x61(__gm__ half *a, __gm__ half *b, __gm__ half *c);

void LaunchTDIV_f16_64x64_v61x61(void *a, void *b, void *c, void *stream) {
    TDIV_f16_64x64_v61x61<<<1, nullptr, stream>>>((__gm__ half *)a, (__gm__ half *)b, (__gm__ half *)c);
}

// Case 5: f32 64x32 v60x30
extern "C" __global__ AICORE void TDIV_f32_64x32_v60x30(__gm__ float *a, __gm__ float *b, __gm__ float *c);

void LaunchTDIV_f32_64x32_v60x30(void *a, void *b, void *c, void *stream) {
    TDIV_f32_64x32_v60x30<<<1, nullptr, stream>>>((__gm__ float *)a, (__gm__ float *)b, (__gm__ float *)c);
}