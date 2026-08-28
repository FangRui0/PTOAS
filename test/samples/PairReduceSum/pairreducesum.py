# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os

from ptoas.mlir.ir import (
    Context,
    InsertionPoint,
    Location,
    Module,
    StringAttr,
    UnitAttr,
)
from ptoas.mlir.dialects import func, arith, pto
from ptoas.mlir.ir import F32Type, IndexType


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            m = Module.create()
            arch = os.environ.get("PTOAS_SAMPLE_ARCH", "a5")
            m.operation.attributes["pto.target_arch"] = StringAttr.get(arch)

            f32 = F32Type.get(ctx)
            ptr_f32 = pto.PtrType.get(f32, ctx)

            tv2_f32 = pto.TensorViewType.get(2, f32, ctx)
            tile_view_32x64 = pto.PartitionTensorViewType.get([32, 64], f32, ctx)
            vec = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC, ctx)
            bl = pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx)
            sl = pto.SLayoutAttr.get(pto.SLayout.NoneBox, ctx)
            pd = pto.PadValueAttr.get(pto.PadValue.Null, ctx)

            cfg = pto.TileBufConfigAttr.get(bl, sl, pto.TileConfig.fractalABSize, pd, ctx)
            tile_buf_32x64 = pto.TileBufType.get([32, 64], f32, vec, [32, 64], cfg, ctx)

            fn_ty = func.FunctionType.get([ptr_f32, ptr_f32], [])
            with InsertionPoint(m.body):
                fn = func.FuncOp("pairreducesum_kernel_2d", fn_ty)
                fn.operation.attributes["pto.entry"] = UnitAttr.get(ctx)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                c0 = arith.ConstantOp(IndexType.get(ctx), 0).result
                c1 = arith.ConstantOp(IndexType.get(ctx), 1).result
                c32 = arith.ConstantOp(IndexType.get(ctx), 32).result
                c64 = arith.ConstantOp(IndexType.get(ctx), 64).result
                src_ptr, dst_ptr = entry.arguments

                tv_src = pto.MakeTensorViewOp(tv2_f32, src_ptr, [c32, c64], [c64, c1]).result
                tv_dst = pto.MakeTensorViewOp(tv2_f32, dst_ptr, [c32, c64], [c64, c1]).result

                sv_src = pto.PartitionViewOp(tile_view_32x64, tv_src, offsets=[c0, c0], sizes=[c32, c64]).result
                sv_dst = pto.PartitionViewOp(tile_view_32x64, tv_dst, offsets=[c0, c0], sizes=[c32, c64]).result

                src_tile = pto.AllocTileOp(tile_buf_32x64).result
                dst_tile = pto.AllocTileOp(tile_buf_32x64).result

                pto.TLoadOp(None, sv_src, src_tile)
                pto.TPairReduceSumOp(src_tile, dst_tile)
                pto.TStoreOp(None, dst_tile, sv_dst)

                func.ReturnOp([])

            m.operation.verify()
            return m


if __name__ == "__main__":
    print(build())
