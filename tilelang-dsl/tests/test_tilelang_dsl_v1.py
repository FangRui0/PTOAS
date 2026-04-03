import tempfile
import unittest
from importlib import util
from pathlib import Path

import tilelang_dsl as pto


class TileLangDSLPackageTests(unittest.TestCase):
    def test_package_exports_surface(self) -> None:
        self.assertIsNotNone(pto.__file__)
        self.assertTrue(hasattr(pto, "vkernel"))
        self.assertTrue(hasattr(pto, "TensorView"))
        self.assertTrue(hasattr(pto, "Tile"))
        self.assertTrue(hasattr(pto, "TileSpecialization"))


class TileLangDSLDescriptorTests(unittest.TestCase):
    def test_descriptor_metadata_and_parameter_binding(self) -> None:
        @pto.vkernel(op="eltwise", dtypes=[(pto.f32, pto.f16, pto.i32)], verify=False)
        def kernel(inp: pto.TensorView, tile: pto.Tile, scale: pto.i32):
            return None

        self.assertEqual(kernel.target, "a5")
        self.assertEqual(kernel.op, "eltwise")
        self.assertEqual(kernel.name, "kernel")
        self.assertFalse(kernel.verify_enabled)
        self.assertEqual(kernel.metadata["verify"], False)
        self.assertEqual(kernel.dtype_signature, (pto.f32, pto.f16, pto.i32))
        self.assertEqual(
            [(param.name, param.kind, param.dtype) for param in kernel.parameters],
            [("inp", "tensorview", pto.f32), ("tile", "tile", pto.f16), ("scale", "scalar", pto.i32)],
        )
        self.assertEqual(kernel.parameters[0].element_dtype, pto.f32)
        self.assertEqual(kernel.parameters[1].element_dtype, pto.f16)
        self.assertIsNone(kernel.parameters[2].element_dtype)

    def test_specialization_enables_materialization_apis(self) -> None:
        @pto.vkernel(op="eltwise", dtypes=[(pto.f32, pto.f16)])
        def kernel(inp: pto.TensorView, tile: pto.Tile):
            return None

        specialized = kernel.specialize(
            tile=pto.TileSpecialization(
                shape=(16, 32),
                memory_space=pto.MemorySpace.UB,
                config=pto.TileConfig.from_mapping({"layout": "row_major"}),
            )
        )

        self.assertIn("tile", specialized.specializations_by_name)
        text = specialized.mlir_text()
        self.assertIn("// tilelang.target = a5", text)
        self.assertIn("// tilelang.specialize tile shape=(16, 32) memory_space=ub", text)
        module = specialized.mlir_module()
        self.assertEqual(type(module).__name__, "MaterializedMLIRModule")
        self.assertTrue(module.verify())
        self.assertTrue(specialized.verify())

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "kernel.mlir"
            specialized.emit(out)
            self.assertEqual(out.read_text(encoding="utf-8"), text)


class TileLangDSLDiagnosticsTests(unittest.TestCase):
    def test_matcher_feature_diagnostics_point_to_follow_up_change(self) -> None:
        cases = [
            lambda: pto.vkernel(op="x", dtypes=[(pto.f32,)], constraints=[])(lambda x: None),
            lambda: pto.vkernel(op="x", dtypes=[(pto.f32,)], priority=1)(lambda x: None),
            lambda: pto.vkernel(op="x", dtypes=[(pto.f32,), (pto.f16,)])(lambda x: None),
            lambda: pto.vkernel(op="x", dtypes=[(pto.AnyFloat,)])(lambda x: None),
            lambda: pto.vkernel(op="x", dtypes=[(pto.TypeVar("T"),)])(lambda x: None),
        ]

        for thunk in cases:
            with self.assertRaises(ValueError) as ctx:
                thunk()
            self.assertIn(
                "extend-tilelang-dsl-matcher-and-advanced-surface",
                str(ctx.exception),
            )

    def test_unsupported_python_syntax_reports_source_location(self) -> None:
        with self.assertRaises(pto.TileLangFrontendError) as ctx:

            @pto.vkernel(op="x", dtypes=[(pto.f32,)])
            def kernel(x: pto.TensorView):
                while True:
                    return None

        self.assertIn("unsupported Python syntax `while`", str(ctx.exception))
        self.assertIn(f"{__file__}:", str(ctx.exception))

    def test_arbitrary_external_call_reports_source_location(self) -> None:
        def helper():
            return None

        with self.assertRaises(pto.TileLangFrontendError) as ctx:

            @pto.vkernel(op="x", dtypes=[(pto.f32,)])
            def kernel(x: pto.TensorView):
                helper()
                return None

        self.assertIn("arbitrary external call `helper`", str(ctx.exception))
        self.assertIn(f"{__file__}:", str(ctx.exception))

    def test_unsupported_pto_surface_reports_source_location(self) -> None:
        with self.assertRaises(pto.TileLangFrontendError) as ctx:

            @pto.vkernel(op="x", dtypes=[(pto.f32,)])
            def kernel(x: pto.TensorView):
                pto.vadd(x)
                return None

        self.assertIn("unsupported op surface `pto.vadd`", str(ctx.exception))
        self.assertIn(f"{__file__}:", str(ctx.exception))

    def test_missing_specialization_reports_source_location(self) -> None:
        @pto.vkernel(op="x", dtypes=[(pto.f32, pto.f16)])
        def kernel(x: pto.TensorView, tile: pto.Tile):
            return None

        with self.assertRaises(pto.TileLangFrontendError) as ctx:
            kernel.mlir_text()

        self.assertIn("requires specialize() bindings for bare Tile parameters", str(ctx.exception))
        self.assertIn(f"{__file__}:", str(ctx.exception))

    def test_dynamic_shape_and_illegal_profile_report_source_location(self) -> None:
        @pto.vkernel(op="x", dtypes=[(pto.f32, pto.f16)])
        def kernel(x: pto.TensorView, tile: pto.Tile):
            return None

        with self.assertRaises(pto.TileLangFrontendError) as dynamic_ctx:
            kernel.specialize(tile={"shape": (16, "n"), "memory_space": "ub"})
        self.assertIn("dynamic physical Tile shape is not supported", str(dynamic_ctx.exception))
        self.assertIn(f"{__file__}:", str(dynamic_ctx.exception))

        with self.assertRaises(pto.TileLangFrontendError) as rank_ctx:
            kernel.specialize(tile={"shape": (4, 4, 4), "memory_space": "ub"})
        self.assertIn("v1 only supports rank-1 or rank-2 Tile shapes", str(rank_ctx.exception))
        self.assertIn(f"{__file__}:", str(rank_ctx.exception))

        with self.assertRaises(pto.TileLangFrontendError) as space_ctx:
            kernel.specialize(tile={"shape": (4, 4), "memory_space": "gm"})
        self.assertIn("v1 only supports MemorySpace.UB", str(space_ctx.exception))
        self.assertIn(f"{__file__}:", str(space_ctx.exception))


if __name__ == "__main__":
    unittest.main()
