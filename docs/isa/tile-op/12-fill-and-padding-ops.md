# 12. Fill and Padding Operations

> **Category:** Tile-local fill, pad, and expansion materialization
> **Pipeline:** PIPE_V

This chapter documents the unified TileLib fill / padding operation. It preserves or materializes valid data and then synthesizes the remaining destination region from the destination tile's padding policy.

The destination tile's `pad` / `pad_value` configuration determines which value is written into the synthesized padding or expansion region.

---

## 12.1 `pto.tfillpad`

- **syntax:**
```mlir
pto.tfillpad ins(%src : !pto.tile_buf<...>)
             outs(%dst : !pto.tile_buf<...>)
             {mode = #pto.tfillpad_mode<normal>}
```
- **semantics:** the `mode` attribute selects normal, in-place, or expand behavior. It defaults to `normal`; PTOAS does not infer it from aliasing or shape.

**Parameter Table:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `pto.tile_buf` | Source tile. |
| `dst` | `pto.tile_buf` | Destination tile carrying the pad configuration. |
| `mode` | `#pto.tfillpad_mode<normal\|in_place\|expand>` | ISA mode; defaults to `normal`. |
| `padValue` | `#pto.pad_value<...>` (optional) | Explicit MAT `TFILLPAD<PadValue>` argument; only valid in normal mode. |

**Mode Table:**

| Mode | Behavior | PTO-ISA mapping |
|------|----------|-----------------|
| `normal` | Copy valid data from `src`, then fill padding in `dst`. | `TFILLPAD(dst, src)` |
| `in_place` | Skip the copy phase and fill padding on shared storage. | `TFILLPAD<pto::TFillPadMode::InPlace>(dst, src)` |
| `expand` | Copy `src` into a destination whose static shape may be larger, then fill the expanded region. | `TFILLPAD<pto::TFillPadMode::Expand>(dst, src)` |

**Constraints:**

- Source and destination element types must be compatible.
- The destination tile must carry a meaningful pad configuration.
- `in_place` and `expand` are VEC-only. Normal mode also supports the homogeneous MAT overload.
- Normal and in-place modes require equal source and destination static shapes.
- Expand mode requires each destination static dimension to be greater than or equal to the source dimension.

**Example:**

```mlir
pto.tfillpad ins(%src : !pto.tile_buf<vec, 8x64xf32, valid=?x?>)
             outs(%dst : !pto.tile_buf<vec, 8x64xf32, pad=1>)

pto.tfillpad ins(%tile : !pto.tile_buf<vec, 32x32xf32, pad=1>)
             outs(%tile : !pto.tile_buf<vec, 32x32xf32, pad=1>)
             {mode = #pto.tfillpad_mode<in_place>}

pto.tfillpad ins(%src_small : !pto.tile_buf<vec, 4x32xf32>)
             outs(%dst_large : !pto.tile_buf<vec, 8x64xf32, pad=1>)
             {mode = #pto.tfillpad_mode<expand>}
```
