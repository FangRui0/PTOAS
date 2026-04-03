TileLang DSL examples live here.

Examples in this subtree should import `tilelang_dsl` as their package
entrypoint once the package wiring is added.

Current example:
- `v1_emit_mlir_demo.py`: define a v1 `@pto.vkernel`, specialize a bare
  `Tile`, and materialize the result as MLIR text or an `.mlir` file

Typical usage from the repository root:

```bash
cmake --build build --target TileLangDSLPackage
python3 tilelang-dsl/examples/v1_emit_mlir_demo.py
python3 tilelang-dsl/examples/v1_emit_mlir_demo.py /tmp/tilelang_demo.mlir
```
