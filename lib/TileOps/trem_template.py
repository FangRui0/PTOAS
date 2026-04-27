"""TileLang DSL template for pto.trem"""

import tilelang_dsl as pto


@pto.vkernel(
    target="a5",
    op="pto.trem",
    dtypes=[
        (pto.f32, pto.f32, pto.f32, pto.f32),
        (pto.f16, pto.f16, pto.f16, pto.f16),
        (pto.i32, pto.i32, pto.i32, pto.i32),
    ],
)
def template_trem(src0: pto.Tile, src1: pto.Tile, tmp: pto.Tile, dst: pto.Tile):
    dtype = dst.element_type
    valid_rows, valid_cols = dst.valid_shape

    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, pto.get_lanes(dtype)):
            mask, remained = pto.make_mask(dtype, remained)
            lhs = pto.vlds(src0[row, col:])
            rhs = pto.vlds(src1[row, col:])
            if pto.constexpr(dtype == pto.f32 or dtype == pto.f16):
                quotient = pto.vdiv(lhs, rhs, mask)
                quotient = pto.vtrc(quotient, mask, rnd=pto.VcvtRoundMode.F)
                floored_mul = pto.vmul(quotient, rhs, mask)
                result = pto.vsub(lhs, floored_mul, mask)
            elif pto.constexpr(dtype == pto.i32):
                lhs_f32 = pto.vcvt(lhs, pto.f32, mask, rnd=pto.VcvtRoundMode.R)
                rhs_f32 = pto.vcvt(rhs, pto.f32, mask, rnd=pto.VcvtRoundMode.R)
                quotient = pto.vdiv(lhs_f32, rhs_f32, mask)
                quotient = pto.vtrc(quotient, mask, rnd=pto.VcvtRoundMode.F)
                floored_mul = pto.vmul(quotient, rhs_f32, mask)
                result_f32 = pto.vsub(lhs_f32, floored_mul, mask)
                result = pto.vcvt(result_f32, pto.i32, mask, rnd=pto.VcvtRoundMode.Z, sat=pto.VcvtSatMode.NOSAT)
            pto.vsts(result, dst[row, col:], mask)
    return