"""TileLang DSL template for pto.tcmp"""

import sys
from pathlib import Path
import tilelang_dsl as pto


@pto.vkernel(
    target="a5",
    op="pto.tcmp",
    advanced=True,
)
def template_tcmp(src0: pto.Tile, src1: pto.Tile, dst: pto.Tile):
    dtype = src0.element_type
    valid_rows, valid_cols = dst.valid_shape
    cmp_mode = pto.get_op_attr("cmp_mode", "eq")

    lanes = pto.get_lanes(dtype)

    dst_ptr = dst.as_ptr()
    mask_ptr = pto.castptr(dst_ptr, pto.ptr(pto.ui32, pto.MemorySpace.UB))

    align_stride = 32

    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, lanes):
            mask, remained = pto.make_mask(dtype, remained)
            lhs = pto.vlds(src0[row, col:])
            rhs = pto.vlds(src1[row, col:])
            result = pto.vcmp(lhs, rhs, mask, cmp_mode)
            byte_offset = row * align_stride + (col // 8)
            pto.psts(result, mask_ptr, byte_offset)
    return