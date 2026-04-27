// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You can not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Host driver for TileLang tadd ST — case-table driven.
// Each case launches a different kernel variant, reads/writes from per-case subdirectory.
// Numerical comparison is done externally by compare.py.

#include "acl/acl.h"
#include "test_common.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/stat.h>

using namespace PtoTestCommon;

// Kernel launch wrappers (defined in launch.cpp)
void LaunchTADD_f32_64x128_64x128_64x128_64x128(void *a, void *b, void *c, void *stream);
void LaunchTADD_f32_16x64_16x64_16x64_16x64(void *a, void *b, void *c, void *stream);
void LaunchTADD_f32_32x32_32x32_32x32_32x32(void *a, void *b, void *c, void *stream);
void LaunchTADD_f32_64x64_64x64_64x64_64x64(void *a, void *b, void *c, void *stream);
void LaunchTADD_i32_64x64_64x64_64x64_64x64(void *a, void *b, void *c, void *stream);
void LaunchTADD_i16_64x64_64x64_64x64_64x64(void *a, void *b, void *c, void *stream);
void LaunchTADD_f16_16x256_16x256_16x256_16x256(void *a, void *b, void *c, void *stream);
void LaunchTADD_half_16x64_16x128_16x128_16x64(void *a, void *b, void *c, void *stream);

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
    {"f32_64x128_64x128_64x128_64x128", LaunchTADD_f32_64x128_64x128_64x128_64x128, 64, 128, 64, 128, 64, 128, 64, 128, sizeof(float)},
    {"f32_16x64_16x64_16x64_16x64", LaunchTADD_f32_16x64_16x64_16x64_16x64, 16, 64, 16, 64, 16, 64, 16, 64, sizeof(float)},
    {"f32_32x32_32x32_32x32_32x32", LaunchTADD_f32_32x32_32x32_32x32_32x32, 32, 32, 32, 32, 32, 32, 32, 32, sizeof(float)},
    {"f32_64x64_64x64_64x64_64x64", LaunchTADD_f32_64x64_64x64_64x64_64x64, 64, 64, 64, 64, 64, 64, 64, 64, sizeof(float)},
    {"i32_64x64_64x64_64x64_64x64", LaunchTADD_i32_64x64_64x64_64x64_64x64, 64, 64, 64, 64, 64, 64, 64, 64, sizeof(int32_t)},
    {"i16_64x64_64x64_64x64_64x64", LaunchTADD_i16_64x64_64x64_64x64_64x64, 64, 64, 64, 64, 64, 64, 64, 64, sizeof(int16_t)},
    {"f16_16x256_16x256_16x256_16x256", LaunchTADD_f16_16x256_16x256_16x256_16x256, 16, 256, 16, 256, 16, 256, 16, 256, sizeof(uint16_t)},
    {"half_16x64_16x128_16x128_16x64", LaunchTADD_half_16x64_16x128_16x128_16x64, 16, 128, 16, 128, 16, 64, 16, 64, sizeof(uint16_t)},
};
static constexpr size_t kNumCases = sizeof(kCases) / sizeof(kCases[0]);

static int RunCase(const TestCase &tc, int deviceId, aclrtStream stream) {
    (void)deviceId;
    int rc = 0;
    const size_t src0Size = tc.src0Rows * tc.src0Cols * tc.elemSize;
    const size_t src1Size = tc.src1Rows * tc.src1Cols * tc.elemSize;
    const size_t dstSize  = tc.dstRows * tc.dstCols * tc.elemSize;

    std::printf("[INFO] === case: %s (dst=%zux%zu, src0=%zux%zu, src1=%zux%zu, valid=%zux%zu) ===\n",
                tc.name, tc.dstRows, tc.dstCols, tc.src0Rows, tc.src0Cols, tc.src1Rows, tc.src1Cols, tc.validRows, tc.validCols);

    // Per-case data directory
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
    // Optional case filter: ./tadd [case_name]
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