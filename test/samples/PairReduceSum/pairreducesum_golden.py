#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms of
# the CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. Please make sure you comply with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO, NON-INFRINGEMENT, MERCHANTABILITY OR
# FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full license text.

import numpy as np
from pathlib import Path
import sys

for search_root in (Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]):
    if (search_root / "validation_runtime.py").is_file():
        sys.path.insert(0, str(search_root))
        break

from validation_runtime import (
    default_buffers,
    float_values,
    load_case_meta,
    matrix32,
    rng,
    single_output,
    write_buffers,
    write_golden,
)


def main():
    meta = load_case_meta()
    [src_name] = meta.inputs
    out_name = single_output(meta)
    generator = rng()
    src = float_values(generator, meta.elem_counts[src_name], style="signed")
    src_m = matrix32(src)
    buffers = default_buffers(meta)
    buffers[src_name] = src
    write_buffers(meta, buffers)

    reduced = np.zeros_like(src_m, dtype=np.float32)
    reduced[:, : src_m.shape[1] // 2] = src_m[:, 0::2] + src_m[:, 1::2]

    out = np.zeros(meta.elem_counts[out_name], dtype=np.float32)
    out[: reduced.size] = reduced.reshape(-1)
    write_golden(meta, {out_name: out})


if __name__ == "__main__":
    main()
