# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Acquire the version-sensitive native pybind11 extensions.

Four extensions are shipped prebuilt in the wheel but cannot be abi3 (pybind11 is
interpreter-specific): ``ptoas._core`` and the ``ptoas.mlir`` family (``_mlir``,
``_mlirDialectsLLVM``, ``_site_initialize_0``). On a matching interpreter the
prebuilt binaries load directly (fast path). On a mismatching interpreter a
``sys.meta_path`` finder redirects those imports to an online CMake build of the
whole family (see ``_build_online``) and serves the freshly built binaries from
the build cache.

The online binaries live outside the package dir, so their ``$ORIGIN`` rpath does
NOT point at the shipped ``libPTOASCompiler`` / ``libLLVMSupport`` DSOs. We
therefore preload those with ``RTLD_GLOBAL`` first so the extensions resolve the
compiler and LLVM support symbols regardless of where they were cached.
"""

import ctypes
import importlib.machinery
import importlib.util
import logging
from pathlib import Path
import sys
import threading
from typing import List, Optional

_log = logging.getLogger(__name__)

_QUALIFIED_MODULE = "ptoas._core"

# Shared, Python-independent DSOs to preload with RTLD_GLOBAL, low-level first
# so their symbols land in the global scope before dependents load. libLLVMSupport
# is only present in wheels built with the online family option (shared LLVM); it
# is absent on a plain prebuilt build, where the extensions find it via their own
# rpath. Preload is therefore best-effort per stem.
_DSO_STEMS = ("libLLVMSupport", "libPTOASCompiler")

# The native modules intercepted by the meta path finder, mapped to their
# in-package location relative to the ``ptoas`` package dir.
_MEMBER_PKG_SUBDIR = {
    "ptoas._core": ".",
    "ptoas.mlir._mlir_libs._mlir": "mlir/_mlir_libs",
    "ptoas.mlir._mlir_libs._mlirDialectsLLVM": "mlir/_mlir_libs",
    "ptoas.mlir._mlir_libs._site_initialize_0": "mlir/_mlir_libs",
}

_ensure_lock = threading.Lock()
_ensured_module = None

_preload_lock = threading.Lock()
_preloaded = False

_finder_lock = threading.Lock()
_finder_installed = False


def _package_dir() -> Path:
    return Path(__file__).parent.resolve()


def _probe_in_pkg(stem: str, dirs: List[Path], exts) -> Optional[Path]:
    """First unmangled ``{stem}{ext}`` under any in-package dir, else ``None``."""
    for cand in dirs:
        if not cand.is_dir():
            continue
        hit = next((cand / f"{stem}{ext}" for ext in exts if (cand / f"{stem}{ext}").exists()), None)
        if hit is not None:
            return hit
    return None


def _probe_relocated(stem: str, dirs: List[Path]) -> Optional[Path]:
    """First hash-mangled ``{stem}*`` under a wheel-repair relocation dir."""
    for cand in dirs:
        if not cand.is_dir():
            continue
        globbed = next(
            (p for pattern in (f"{stem}*.so*", f"{stem}*.dylib") for p in sorted(cand.glob(pattern))),
            None,
        )
        if globbed is not None:
            return globbed
    return None


def _find_dsos(pkg_dir: Path) -> List[Path]:
    """Locate the shipped shared DSOs (one per stem, first hit wins).

    Probe the in-package dirs first (unmangled names, e.g. an editable build),
    then the relocation dirs a wheel repair tool moves external DSOs into
    (``auditwheel`` -> ``<dist>.libs``; ``delocate`` -> ``<pkg>/.dylibs``), where
    the file is hash-mangled (e.g. ``libLLVMSupport-<hash>.so.19.1``) and must be
    matched by glob.
    """
    in_pkg = [pkg_dir, pkg_dir / "mlir" / "_mlir_libs"]
    relocated = [pkg_dir.parent / "ptoas.libs", pkg_dir / ".dylibs"]
    exts = (".so", ".dylib")
    found: List[Path] = []
    for stem in _DSO_STEMS:
        hit = _probe_in_pkg(stem, in_pkg, exts) or _probe_relocated(stem, relocated)
        if hit is not None:
            found.append(hit)
    return found


def preload_shared_libs():
    """Preload the shipped, Python-independent DSOs with RTLD_GLOBAL (once)."""
    global _preloaded
    if _preloaded:
        return
    with _preload_lock:
        if _preloaded:
            return
        for dso in _find_dsos(_package_dir()):
            try:
                ctypes.CDLL(str(dso), mode=ctypes.RTLD_GLOBAL)
            except OSError as e:
                _log.warning("Failed to preload %s: %s", dso, e)
        _preloaded = True


def _find_prebuilt(fullname: str) -> Optional[Path]:
    """Return the in-package prebuilt extension for ``fullname`` if its ABI
    suffix matches this interpreter, else ``None``."""
    base = _package_dir() / _MEMBER_PKG_SUBDIR[fullname]
    stem = fullname.rsplit(".", 1)[1]
    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        cand = base / f"{stem}{suffix}"
        if cand.exists():
            return cand
    return None


class _OnlineExtensionFinder:
    """meta_path finder serving the version-sensitive native extensions.

    For each of the four EMBED_CAPI pybind11 extensions it returns the prebuilt
    in-package binary when the ABI suffix matches this interpreter, otherwise it
    triggers a one-shot online CMake build of the whole family and serves the
    freshly built binary from the build cache.
    """

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in _MEMBER_PKG_SUBDIR:
            return None
        so = _find_prebuilt(fullname)
        if so is None:
            preload_shared_libs()
            from ._build_online import get_or_build_member

            so = get_or_build_member(fullname)
        loader = importlib.machinery.ExtensionFileLoader(fullname, str(so))
        return importlib.util.spec_from_file_location(fullname, str(so), loader=loader)


def install_online_finder():
    """Install the online-extension finder at the front of ``sys.meta_path``."""
    global _finder_installed
    if _finder_installed:
        return
    with _finder_lock:
        if _finder_installed:
            return
        sys.meta_path.insert(0, _OnlineExtensionFinder())
        _finder_installed = True


def ensure_core():
    """Return the ``ptoas._core`` module, using the prebuilt binary when possible.

    On an ABI-matching interpreter the finder serves the prebuilt in-package
    ``_core`` (the fast path); otherwise it serves an online-compiled build. The
    online ``_core`` is full-featured (``main`` / TileLib / SoftLib), not
    dialect-only, so the ``ptoas`` CLI works on mismatching interpreters too.

    The result is also registered as ``sys.modules['ptoas._core']`` so that
    ``from ptoas import _core`` works uniformly for both the prebuilt and the
    online-compiled module.
    """
    global _ensured_module
    if _ensured_module is not None:
        return _ensured_module
    with _ensure_lock:
        if _ensured_module is not None:
            return _ensured_module

        preload_shared_libs()
        install_online_finder()
        # The finder serves the prebuilt binary on a matching interpreter, or an
        # online-compiled one otherwise; a genuinely broken install raises here.
        import ptoas._core as core

        _ensured_module = core
        return core
