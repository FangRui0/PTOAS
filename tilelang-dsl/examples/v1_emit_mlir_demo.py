"""Minimal TileLang DSL v1 demo that materializes a kernel into MLIR."""

import sys
from pathlib import Path


def _import_tilelang_dsl():
    try:
        import tilelang_dsl as pto

        return pto
    except ModuleNotFoundError:
        repo_root = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(repo_root / "build" / "python"))
        import tilelang_dsl as pto

        return pto


pto = _import_tilelang_dsl()


@pto.vkernel(
    op="eltwise_with_tile",
    dtypes=[(pto.f32, pto.f16, pto.i32)],
    name="tilelang_v1_demo_kernel",
)
def kernel(inp: pto.TensorView, tile: pto.Tile, scale: pto.i32):
    return None


def build_specialized_kernel():
    return kernel.specialize(
        tile=pto.TileSpecialization(
            shape=(16, 32),
            memory_space=pto.MemorySpace.UB,
            config=pto.TileConfig.from_mapping({"layout": "row_major"}),
        )
    )


def main(argv: list[str]) -> int:
    specialized = build_specialized_kernel()

    if len(argv) > 2:
        print(f"usage: {Path(argv[0]).name} [output.mlir]", file=sys.stderr)
        return 2

    if len(argv) == 2:
        output_path = Path(argv[1])
        specialized.emit(output_path)
        print(f"wrote MLIR to {output_path}")
        return 0

    print(specialized.mlir_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
