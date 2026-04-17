// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "acl/acl.h"
#include "test_common.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace PtoTestCommon;

void LaunchTCI_i32_1x8(const void *start, void *dst, void *stream);
void LaunchTCI_i32_1x32(const void *start, void *dst, void *stream);
void LaunchTCI_i32_1x64(const void *start, void *dst, void *stream);
void LaunchTCI_i32_1x72(const void *start, void *dst, void *stream);
void LaunchTCI_i32_1x80(const void *start, void *dst, void *stream);
void LaunchTCI_i32_1x128(const void *start, void *dst, void *stream);
void LaunchTCI_i16_1x16(const void *start, void *dst, void *stream);
void LaunchTCI_i16_1x64(const void *start, void *dst, void *stream);
void LaunchTCI_i16_1x128(const void *start, void *dst, void *stream);
void LaunchTCI_i16_1x144(const void *start, void *dst, void *stream);
void LaunchTCI_i16_1x160(const void *start, void *dst, void *stream);
void LaunchTCI_i16_1x256(const void *start, void *dst, void *stream);

using LaunchFn = void (*)(const void *, void *, void *);

struct TestCase {
    const char *name;
    LaunchFn    launch;
    size_t      rows;
    size_t      cols;
    size_t      validRows;
    size_t      validCols;
    size_t      elemSize;
    size_t      scalarSize;
};

#define CASE_I32(cols) \
    {"i32_1x" #cols, LaunchTCI_i32_1x##cols, 1, (cols), 1, (cols), sizeof(int32_t), sizeof(int32_t)}
#define CASE_I16(cols) \
    {"i16_1x" #cols, LaunchTCI_i16_1x##cols, 1, (cols), 1, (cols), sizeof(int16_t), sizeof(int16_t)}

static const TestCase kCases[] = {
    CASE_I32(8),
    CASE_I32(32),
    CASE_I32(64),
    CASE_I32(72),
    CASE_I32(80),
    CASE_I32(128),
    CASE_I16(16),
    CASE_I16(64),
    CASE_I16(128),
    CASE_I16(144),
    CASE_I16(160),
    CASE_I16(256),
};
static constexpr size_t kNumCases = sizeof(kCases) / sizeof(kCases[0]);

static int RunCase(const TestCase &tc, aclrtStream stream) {
    int rc = 0;
    const size_t elemCount = tc.rows * tc.cols;
    const size_t fileSize = elemCount * tc.elemSize;

    std::printf("[INFO] === case: %s (shape=%zux%zu, valid=%zux%zu) ===\n",
                tc.name, tc.rows, tc.cols, tc.validRows, tc.validCols);

    std::string caseDir = std::string("./") + tc.name;
    size_t startFileSize = tc.scalarSize;
    std::vector<unsigned char> startHost(tc.scalarSize);
    void *dstHost = nullptr;
    void *dstDevice = nullptr;

    aclrtMallocHost(&dstHost, fileSize);
    aclrtMalloc(&dstDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    if (!ReadFile((caseDir + "/start.bin").c_str(), startFileSize, startHost.data(), tc.scalarSize)) {
        std::fprintf(stderr, "[ERROR] failed to read %s/start.bin\n", caseDir.c_str());
        rc = 1;
    }

    if (rc == 0) {
        tc.launch(startHost.data(), dstDevice, stream);
        aclrtSynchronizeStream(stream);
        aclrtMemcpy(dstHost, fileSize, dstDevice, fileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    }

    if (rc == 0 && !WriteFile((caseDir + "/output.bin").c_str(), dstHost, fileSize)) {
        std::fprintf(stderr, "[ERROR] failed to write %s/output.bin\n", caseDir.c_str());
        rc = 1;
    }

    if (dstDevice != nullptr)
        aclrtFree(dstDevice);
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
        int ret = RunCase(kCases[i], stream);
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
