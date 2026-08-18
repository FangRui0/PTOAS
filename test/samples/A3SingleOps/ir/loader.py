# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from pathlib import Path

from ptoas.mlir.dialects import pto
from ptoas.mlir.ir import Context, Module


def print_case(loader_file: str) -> None:
    case_name = Path(loader_file).stem
    ir_path = Path(loader_file).parent / "ir" / f"{case_name}.pto"
    with Context() as context:
        pto.register_dialect(context, load=True)
        ir_text = ir_path.read_text(encoding="utf-8")
        module = Module.parse(ir_text)
        module.operation.verify()
        print(ir_text)
