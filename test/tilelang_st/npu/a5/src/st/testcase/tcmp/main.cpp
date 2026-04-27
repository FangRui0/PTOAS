// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You can not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Host driver for TileLang tcmp ST — case-table driven.

#include "acl/acl.h"
#include "test_common.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/stat.h>

using namespace PtoTestCommon;

void LaunchTCMP_f32_1x64_eq(float *a, float *b, int8_t *c, void *stream);
void LaunchTCMP_f32_8x64_gt(float *a, float *b, int8_t *c, void *stream);
void LaunchTCMP_i32_16x32_eq(int32_t *a, int32_t *b, int8_t *c, void *stream);
void LaunchTCMP_i32_32x32_eq(int32_t *a, int32_t *b, int8_t *c, void *stream);

using LaunchFn = void (*)(void *, void *, void *, void *);

struct TestCase {
    const char *name;
    LaunchFn    launch;
    size_t      rows;
    size_t      cols;
    size_t      validRows;
    size_t      validCols;
    size_t      inputElemSize;
    size_t      outputElemSize;
};

static const TestCase kCases[] = {
    {"f32_1x64_eq",  (LaunchFn)LaunchTCMP_f32_1x64_eq,  1,  64, 1,  64, sizeof(float),   sizeof(int8_t)},
    {"f32_8x64_gt",  (LaunchFn)LaunchTCMP_f32_8x64_gt,  8,  64, 8,  64, sizeof(float),   sizeof(int8_t)},
    {"i32_16x32_eq", (LaunchFn)LaunchTCMP_i32_16x32_eq, 16, 32, 16, 32, sizeof(int32_t), sizeof(int8_t)},
    {"i32_32x32_eq", (LaunchFn)LaunchTCMP_i32_32x32_eq, 32, 32, 32, 32, sizeof(int32_t), sizeof(int8_t)},
};
static constexpr size_t kNumCases = sizeof(kCases) / sizeof(kCases[0]);

static int RunCase(const TestCase &tc, int deviceId, aclrtStream stream) {
    int rc = 0;
    const size_t elemCount = tc.rows * tc.cols;
    const size_t inputFileSize  = elemCount * tc.inputElemSize;
    const size_t outputFileSize = elemCount * tc.outputElemSize;
    const size_t packedMaskSize = tc.validRows * 32 * tc.outputElemSize;

    std::printf("[INFO] === case: %s (shape=%zux%zu, valid=%zux%zu, packed_mask=%zu) ===\n",
                tc.name, tc.rows, tc.cols, tc.validRows, tc.validCols, packedMaskSize);

    std::string caseDir = std::string("./") + tc.name;

    void *src0Host = nullptr, *src1Host = nullptr, *dstHost = nullptr;
    void *src0Device = nullptr, *src1Device = nullptr, *dstDevice = nullptr;

    aclrtMallocHost(&src0Host, inputFileSize);
    aclrtMallocHost(&src1Host, inputFileSize);
    aclrtMallocHost(&dstHost, outputFileSize);

    aclrtMalloc(&src0Device, inputFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&src1Device, inputFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dstDevice, outputFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    size_t readSize = inputFileSize;
    if (!ReadFile((caseDir + "/input1.bin").c_str(), readSize, src0Host, inputFileSize)) {
        std::fprintf(stderr, "[ERROR] failed to read %s/input1.bin\n", caseDir.c_str());
        rc = 1;
    }
    readSize = inputFileSize;
    if (rc == 0 && !ReadFile((caseDir + "/input2.bin").c_str(), readSize, src1Host, inputFileSize)) {
        std::fprintf(stderr, "[ERROR] failed to read %s/input2.bin\n", caseDir.c_str());
        rc = 1;
    }

    if (rc == 0) {
        aclrtMemcpy(src0Device, inputFileSize, src0Host, inputFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(src1Device, inputFileSize, src1Host, inputFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

        tc.launch(src0Device, src1Device, dstDevice, stream);

        aclrtSynchronizeStream(stream);
        aclrtMemcpy(dstHost, outputFileSize, dstDevice, outputFileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    }

    if (rc == 0 && !WriteFile((caseDir + "/output.bin").c_str(), dstHost, outputFileSize)) {
        std::fprintf(stderr, "[ERROR] failed to write %s/output.bin\n", caseDir.c_str());
        rc = 1;
    }

    if (src0Device != nullptr)
        aclrtFree(src0Device);
    if (src1Device != nullptr)
        aclrtFree(src1Device);
    if (dstDevice != nullptr)
        aclrtFree(dstDevice);
    if (src0Host != nullptr)
        aclrtFreeHost(src0Host);
    if (src1Host != nullptr)
        aclrtFreeHost(src1Host);
    if (dstHost != nullptr)
        aclrtFreeHost(dstHost);

    if (rc == 0)
        std::printf("[INFO] case %s done\n", tc.name);
    return rc;
}

int main(int argc, char *argv[]) {
    const char *caseFilter = (argc > 1) ? argv[1] : nullptr;

    int rc = 0;
    int deviceId = 0;
    aclrtStream stream = nullptr;

    aclInit(nullptr);
    if (const char *envDevice = std::getenv("ACL_DEVICE_ID")) {
        deviceId = std::atoi(envDevice);
    }
    aclrtSetDevice(deviceId);
    aclrtCreateStream(&stream);

    for (size_t i = 0; i < kNumCases; ++i) {
        if (caseFilter != nullptr && std::strcmp(kCases[i].name, caseFilter) != 0) {
            continue;
        }
        int ret = RunCase(kCases[i], deviceId, stream);
        if (ret != 0) {
            std::fprintf(stderr, "[ERROR] case %s failed\n", kCases[i].name);
            rc = 1;
            break;
        }
    }

    if (stream != nullptr)
        aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return rc;
}