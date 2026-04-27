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

// Case 0: f32, dst=16x64, src0=16x128, src1=16x128, valid=16x64
extern "C" __global__ AICORE void TREM_f32_16x64_16x128_16x128_16x64(__gm__ float *a, __gm__ float *b, __gm__ float *c);

void LaunchTREM_f32_16x64_16x128_16x128_16x64(void *a, void *b, void *c, void *stream) {
    TREM_f32_16x64_16x128_16x128_16x64<<<1, nullptr, stream>>>((__gm__ float *)a, (__gm__ float *)b, (__gm__ float *)c);
}

// Case 1: f32, dst=16x32, src0=16x64, src1=16x32, valid=16x32
extern "C" __global__ AICORE void TREM_f32_16x32_16x64_16x32_16x32(__gm__ float *a, __gm__ float *b, __gm__ float *c);

void LaunchTREM_f32_16x32_16x64_16x32_16x32(void *a, void *b, void *c, void *stream) {
    TREM_f32_16x32_16x64_16x32_16x32<<<1, nullptr, stream>>>((__gm__ float *)a, (__gm__ float *)b, (__gm__ float *)c);
}

// Case 2: i32, dst=4x32, src0=4x32, src1=4x32, valid=4x32
extern "C" __global__ AICORE void TREM_i32_4x32_4x32_4x32_4x32(__gm__ int32_t *a, __gm__ int32_t *b, __gm__ int32_t *c);

void LaunchTREM_i32_4x32_4x32_4x32_4x32(void *a, void *b, void *c, void *stream) {
    TREM_i32_4x32_4x32_4x32_4x32<<<1, nullptr, stream>>>((__gm__ int32_t *)a, (__gm__ int32_t *)b, (__gm__ int32_t *)c);
}

// Case 3: i32, dst=16x32, src0=16x64, src1=16x32, valid=16x32
extern "C" __global__ AICORE void TREM_i32_16x32_16x64_16x32_16x32(__gm__ int32_t *a, __gm__ int32_t *b, __gm__ int32_t *c);

void LaunchTREM_i32_16x32_16x64_16x32_16x32(void *a, void *b, void *c, void *stream) {
    TREM_i32_16x32_16x64_16x32_16x32<<<1, nullptr, stream>>>((__gm__ int32_t *)a, (__gm__ int32_t *)b, (__gm__ int32_t *)c);
}

// Case 4: f32, dst=16x64, src0=16x128, src1=16x128, valid=16x63
extern "C" __global__ AICORE void TREM_f32_16x64_16x128_16x128_16x63(__gm__ float *a, __gm__ float *b, __gm__ float *c);

void LaunchTREM_f32_16x64_16x128_16x128_16x63(void *a, void *b, void *c, void *stream) {
    TREM_f32_16x64_16x128_16x128_16x63<<<1, nullptr, stream>>>((__gm__ float *)a, (__gm__ float *)b, (__gm__ float *)c);
}

// Case 5: f32, dst=2x32, src0=2x64, src1=2x32, valid=2x31
extern "C" __global__ AICORE void TREM_f32_2x32_2x64_2x32_2x31(__gm__ float *a, __gm__ float *b, __gm__ float *c);

void LaunchTREM_f32_2x32_2x64_2x32_2x31(void *a, void *b, void *c, void *stream) {
    TREM_f32_2x32_2x64_2x32_2x31<<<1, nullptr, stream>>>((__gm__ float *)a, (__gm__ float *)b, (__gm__ float *)c);
}

// Case 6: i32, dst=16x32, src0=16x64, src1=16x32, valid=16x31
extern "C" __global__ AICORE void TREM_i32_16x32_16x64_16x32_16x31(__gm__ int32_t *a, __gm__ int32_t *b, __gm__ int32_t *c);

void LaunchTREM_i32_16x32_16x64_16x32_16x31(void *a, void *b, void *c, void *stream) {
    TREM_i32_16x32_16x64_16x32_16x31<<<1, nullptr, stream>>>((__gm__ int32_t *)a, (__gm__ int32_t *)b, (__gm__ int32_t *)c);
}

// Case 7: f32, dst=1x8192, src0=1x8192, src1=1x8192, valid=1x8192
extern "C" __global__ AICORE void TREM_f32_1x8192_1x8192_1x8192_1x8192(__gm__ float *a, __gm__ float *b, __gm__ float *c);

void LaunchTREM_f32_1x8192_1x8192_1x8192_1x8192(void *a, void *b, void *c, void *stream) {
    TREM_f32_1x8192_1x8192_1x8192_1x8192<<<1, nullptr, stream>>>((__gm__ float *)a, (__gm__ float *)b, (__gm__ float *)c);
}