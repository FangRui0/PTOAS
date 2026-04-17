"""TileLang DSL template for pto.tci.

`pto.tci` writes the integer sequence ``dst[i, j] = start + linear_index(i, j)``
into a vec tile. The A5 lowering path currently requires ``valid_rows == 1``,
so this template only covers the 1xC row-major case. On the hardware path the
work is carried out with ``pto.vci``, which seeds a per-lane index vector
``[seed, seed + 1, ..., seed + lanes - 1]``. That matches `TAdd`'s structure
exactly: tile the valid columns into VReg-sized chunks, load the mask for the
trailing partial chunk, and use ``pto.vsts`` to write each chunk back to the
destination tile.
"""

import tilelang_dsl as pto


def _supports_tci_layout(dst):
    return (
        dst.shape[0] == 1
        and dst.valid_shape[0] == 1
        and dst.shape[1] > 1
    )


@pto.vkernel(
    target="a5",
    op="pto.tci",
    dtypes=[(pto.i16, pto.i16)],
    constraints=[_supports_tci_layout],
    advanced=True,
)
def template_tci_i16(start: pto.i16, dst: pto.Tile):
    dtype = dst.element_type
    _, valid_cols = dst.valid_shape

    remained = valid_cols
    for col in range(0, valid_cols, pto.get_lanes(dtype)):
        mask, remained = pto.make_mask(dtype, remained)
        # DSL v1 lowering cannot directly cast `index` to `i16`, so stage
        # through `i32` first (`index -> i32 -> i16`).
        col_i16 = pto.i16(pto.i32(col))
        seed = start + col_i16
        indices = pto.vci(seed)
        pto.vsts(indices, dst[0, col:], mask)
    return


@pto.vkernel(
    target="a5",
    op="pto.tci",
    dtypes=[(pto.i32, pto.i32)],
    constraints=[_supports_tci_layout],
    advanced=True,
)
def template_tci_i32(start: pto.i32, dst: pto.Tile):
    dtype = dst.element_type
    _, valid_cols = dst.valid_shape

    remained = valid_cols
    for col in range(0, valid_cols, pto.get_lanes(dtype)):
        mask, remained = pto.make_mask(dtype, remained)
        seed = start + pto.i32(col)
        indices = pto.vci(seed)
        pto.vsts(indices, dst[0, col:], mask)
    return
