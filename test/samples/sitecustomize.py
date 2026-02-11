import importlib.util
import os
import sys


def _load_local_pto():
    ext_root = os.environ.get("PTOAS_PYTHON_EXT_PATH")
    if not ext_root:
        return
    libs_dir = os.path.join(ext_root, "mlir", "_mlir_libs")
    if not os.path.isdir(libs_dir):
        return
    candidates = []
    for name in os.listdir(libs_dir):
        if name.startswith("_pto") and name.endswith((".so", ".dylib", ".pyd")):
            candidates.append(os.path.join(libs_dir, name))
    if not candidates:
        return
    # Load the first match and register as mlir._mlir_libs._pto.
    path = sorted(candidates)[0]
    try:
        spec = importlib.util.spec_from_file_location("mlir._mlir_libs._pto", path)
        if spec is None or spec.loader is None:
            return
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules["mlir._mlir_libs._pto"] = module
    except Exception:
        # Best-effort: fall back to default _pto if load fails.
        return


_load_local_pto()
