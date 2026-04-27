// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You can not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Host driver for TileLang trem ST — case-table driven.

#include "acl/acl.h"
#include "test_common.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/stat.h>

using namespace PtoTestCommon;

void LaunchTREM_f32_16x64_16x128_16x128_16x64(void *a, void *b, void *c, void *stream);
void LaunchTREM_f32_16x32_16x64_16x32_16x32(void *a, void *b, void *c, void *stream);
void LaunchTREM_i32_4x32_4x32_4x32_4x32(void *a, void *b, void *c, void *stream);
void LaunchTREM_i32_16x32_16x64_16x32_16x32(void *a, void *b, void *c, void *stream);
void LaunchTREM_f32_16x64_16x128_16x128_16x63(void *a, void *b, void *c, void *stream);
void LaunchTREM_f32_2x32_2x64_2x32_2x31(void *a, void *b, void *c, void *stream);
void LaunchTREM_i32_16x32_16x64_16x32_16x31(void *a, void *b, void *c, void *stream);
void LaunchTREM_f32_1x8192_1x8192_1x8192_1x8192(void *a, void *b, void *c, void *stream);

using LaunchFn = void (*)(void *, void *, void *, void *);

struct TestCase {
    const char *name;
    LaunchFn    launch;
    size_t      src0Rows;
    size_t      src0Cols;
    size_t      src1Rows;
    size_t      src1Cols;
    size_t      dstRows;
    size_t      dstCols;
    size_t      validRows;
    size_t      validCols;
    size_t      elemSize;
};

static const TestCase kCases[] = {
    {"f32_16x64_16x128_16x128_16x64", LaunchTREM_f32_16x64_16x128_16x128_16x64, 16, 128, 16, 128, 16, 64, 16, 64, sizeof(float)},
    {"f32_16x32_16x64_16x32_16x32", LaunchTREM_f32_16x32_16x64_16x32_16x32, 16, 64, 16, 32, 16, 32, 16, 32, sizeof(float)},
    {"i32_4x32_4x32_4x32_4x32", LaunchTREM_i32_4x32_4x32_4x32_4x32, 4, 32, 4, 32, 4, 32, 4, 32, sizeof(int32_t)},
    {"i32_16x32_16x64_16x32_16x32", LaunchTREM_i32_16x32_16x64_16x32_16x32, 16, 64, 16, 32, 16, 32, 16, 32, sizeof(int32_t)},
    {"f32_16x64_16x128_16x128_16x63", LaunchTREM_f32_16x64_16x128_16x128_16x63, 16, 128, 16, 128, 16, 64, 16, 63, sizeof(float)},
    {"f32_2x32_2x64_2x32_2x31", LaunchTREM_f32_2x32_2x64_2x32_2x31, 2, 64, 2, 32, 2, 32, 2, 31, sizeof(float)},
    {"i32_16x32_16x64_16x32_16x31", LaunchTREM_i32_16x32_16x64_16x32_16x31, 16, 64, 16, 32, 16, 32, 16, 31, sizeof(int32_t)},
    {"f32_1x8192_1x8192_1x8192_1x8192", LaunchTREM_f32_1x8192_1x8192_1x8192_1x8192, 1, 8192, 1, 8192, 1, 8192, 1, 8192, sizeof(float)},
};
static constexpr size_t kNumCases = sizeof(kCases) / sizeof(kCases[0]);

static int RunCase(const TestCase &tc, int deviceId, aclrtStream stream) {
    int rc = 0;
    const size_t src0Size = tc.src0Rows * tc.src0Cols * tc.elemSize;
    const size_t src1Size = tc.src1Rows * tc.src1Cols * tc.elemSize;
    const size_t dstSize  = tc.dstRows * tc.dstCols * tc.elemSize;

    std::printf("[INFO] === case: %s (dst=%zux%zu, src0=%zux%zu, src1=%zux%zu, valid=%zux%zu) ===\n",
                tc.name, tc.dstRows, tc.dstCols, tc.src0Rows, tc.src0Cols, tc.src1Rows, tc.src1Cols, tc.validRows, tc.validCols);

    std::string caseDir = std::string("./") + tc.name;

    void *src0Host = nullptr, *src1Host = nullptr, *dstHost = nullptr;
    void *src0Device = nullptr, *src1Device = nullptr, *dstDevice = nullptr;

    aclrtMallocHost(&src0Host, src0Size);
    aclrtMallocHost(&src1Host, src1Size);
    aclrtMallocHost(&dstHost, dstSize);

    aclrtMalloc(&src0Device, src0Size, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&src1Device, src1Size, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dstDevice, dstSize, ACL_MEM_MALLOC_HUGE_FIRST);

    size_t fileSize = 0;
    if (!ReadFile((caseDir + "/input1.bin").c_str(), fileSize, src0Host, src0Size)) {
        std::fprintf(stderr, "[ERROR] failed to read %s/input1.bin\n", caseDir.c_str());
        rc = 1;
    }
    fileSize = 0;
    if (rc == 0 && !ReadFile((caseDir + "/input2.bin").c_str(), fileSize, src1Host, src1Size)) {
        std::fprintf(stderr, "[ERROR] failed to read %s/input2.bin\n", caseDir.c_str());
        rc = 1;
    }

    if (rc == 0) {
        aclrtMemcpy(src0Device, src0Size, src0Host, src0Size, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(src1Device, src1Size, src1Host, src1Size, ACL_MEMCPY_HOST_TO_DEVICE);

        tc.launch(src0Device, src1Device, dstDevice, stream);

        aclrtSynchronizeStream(stream);
        aclrtMemcpy(dstHost, dstSize, dstDevice, dstSize, ACL_MEMCPY_DEVICE_TO_HOST);
    }

    if (rc == 0 && !WriteFile((caseDir + "/output.bin").c_str(), dstHost, dstSize)) {
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