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

extern "C" __global__ AICORE void TAND_i16_64x64(__gm__ int16_t *a, __gm__ int16_t *b, __gm__ int16_t *c);

void LaunchTAND_i16_64x64(void *a, void *b, void *c, void *stream) {
    TAND_i16_64x64<<<1, nullptr, stream>>>((__gm__ int16_t *)a, (__gm__ int16_t *)b, (__gm__ int16_t *)c);
}

extern "C" __global__ AICORE void TAND_i8_64x64_valid63x63(__gm__ int8_t *a, __gm__ int8_t *b, __gm__ int8_t *c);

void LaunchTAND_i8_64x64_valid63x63(void *a, void *b, void *c, void *stream) {
    TAND_i8_64x64_valid63x63<<<1, nullptr, stream>>>((__gm__ int8_t *)a, (__gm__ int8_t *)b, (__gm__ int8_t *)c);
}