#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import numpy as np

rng = np.random.default_rng(19)
src0 = rng.uniform(-3.0, 3.0, 64).astype(np.float32)
src1 = rng.uniform(-3.0, 3.0, 64).astype(np.float32)
dst = np.zeros(64, dtype=np.float32)
src0.tofile("v1.bin")
src1.tofile("v2.bin")
dst.tofile("v3.bin")
np.maximum(src0 + src1, np.float32(0.0)).tofile("golden_v3.bin")
