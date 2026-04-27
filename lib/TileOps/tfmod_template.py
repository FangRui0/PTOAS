"""TileLang DSL template for pto.tfmod"""

import tilelang_dsl as pto


@pto.vkernel(
    target="a5",
    op="pto.tfmod",
    dtypes=[
        (pto.f32, pto.f32, pto.f32),
        (pto.f16, pto.f16, pto.f16),
    ],
)
def template_tfmod(src0: pto.Tile, src1: pto.Tile, dst: pto.Tile):
    dtype = dst.element_type
    valid_rows, valid_cols = dst.valid_shape

    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, pto.get_lanes(dtype)):
            mask, remained = pto.make_mask(dtype, remained)
            lhs = pto.vlds(src0[row, col:])
            rhs = pto.vlds(src1[row, col:])
            quotient = pto.vdiv(lhs, rhs, mask)
            if pto.constexpr(dtype == pto.f32 or dtype == pto.f16):
                quotient = pto.vtrc(quotient, mask, rnd=pto.VcvtRoundMode.Z)
            truncated_mul = pto.vmul(quotient, rhs, mask)
            result = pto.vsub(lhs, truncated_mul, mask)
            pto.vsts(result, dst[row, col:], mask)
    return