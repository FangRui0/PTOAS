// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "acl/acl.h"
#include "test_common.h"
#include <cstdio>

using namespace PtoTestCommon;

void LaunchTaddreluA3(float *src0, float *src1, float *dst, void *stream);

int main() {
  constexpr size_t elemCount = 64;
  size_t fileSize = elemCount * sizeof(float);
  float *host[3] = {nullptr, nullptr, nullptr};
  float *device[3] = {nullptr, nullptr, nullptr};
  const char *files[3] = {"./v1.bin", "./v2.bin", "./v3.bin"};
  aclrtStream stream = nullptr;
  int rc = 0;

  aclError status = aclInit(nullptr);
  if (status == ACL_SUCCESS) {
    status = aclrtSetDevice(0);
  }
  if (status == ACL_SUCCESS) {
    status = aclrtCreateStream(&stream);
  }
  if (status != ACL_SUCCESS) {
    return 1;
  }
  for (int i = 0; i < 3; ++i) {
    aclError hostStatus = aclrtMallocHost((void **)&host[i], fileSize);
    aclError deviceStatus = hostStatus == ACL_SUCCESS
                                ? aclrtMalloc((void **)&device[i], fileSize, ACL_MEM_MALLOC_HUGE_FIRST)
                                : hostStatus;
    if (hostStatus != ACL_SUCCESS || deviceStatus != ACL_SUCCESS) {
      rc = 1;
      break;
    }
    ReadFile(files[i], fileSize, host[i], fileSize);
    aclError copyStatus = aclrtMemcpy(device[i], fileSize, host[i], fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if (copyStatus != ACL_SUCCESS) {
      rc = 1;
      break;
    }
  }
  if (rc == 0) {
    LaunchTaddreluA3(device[0], device[1], device[2], stream);
    aclError syncStatus = aclrtSynchronizeStream(stream);
    aclError copyStatus = syncStatus == ACL_SUCCESS
                              ? aclrtMemcpy(host[2], fileSize, device[2], fileSize, ACL_MEMCPY_DEVICE_TO_HOST)
                              : syncStatus;
    if (syncStatus != ACL_SUCCESS || copyStatus != ACL_SUCCESS) {
      rc = 1;
    } else {
      WriteFile("./v3.bin", host[2], fileSize);
    }
  }

  for (int i = 0; i < 3; ++i) {
    if (device[i] != nullptr) {
      aclrtFree(device[i]);
    }
    if (host[i] != nullptr) {
      aclrtFreeHost(host[i]);
    }
  }
  if (stream != nullptr) {
    aclrtDestroyStream(stream);
  }
  aclrtResetDevice(0);
  aclFinalize();
  return rc;
}
