# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Standalone runtime recipe for the online-compiled pybind11 extension fallback.
#
# This file is staged into the wheel at `ptoas/_online/CMakeLists.txt`. It is
# invoked at first import (by ptoas/_build_online.py) against the shipped,
# Python-version-independent DSO (libPTOASCompiler) that lives in the installed
# `ptoas/` package, to produce extensions matching the running interpreter's ABI.
#
# Two extension groups are built here:
#   * `_core` (installed next to the package root) -- the PTO dialect binding,
#     which uses only the project C-API + a Twine-free patched PybindAdaptors.h
#     staged under `include/`.
#   * the `ptoas.mlir` pybind11 family (`_mlir`, `_mlirDialectsLLVM`,
#     `_site_initialize_0`, installed into `mlir/_mlir_libs/`) -- the genuine
#     upstream MLIR bindings, which need the real PybindAdaptors.h and the full
#     LLVM/MLIR C++ header closure staged under `mlir_include/` + their private
#     sibling headers under `src/mlir/`. These dynamically link the shipped
#     shared LLVMSupport that travels next to libPTOASCompiler.
#
# The Python driver passes these cache variables via -D:
#   Python3_EXECUTABLE            - the target interpreter
#   Python3_EXECUTABLE_VERSION    - "3.<minor>" for an EXACT find_package
#   Python3_MOD_PYBIND11_CMAKE_DIR - pybind11.get_cmake_dir() of that interpreter
#   PTOAS_SHIPPED_LIB_DIR         - dir holding libPTOASCompiler (the package dir)

cmake_minimum_required(VERSION 3.20)
project(ptoas_online_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

find_package(Python3 ${Python3_EXECUTABLE_VERSION} EXACT REQUIRED
  COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED
  PATHS ${Python3_MOD_PYBIND11_CMAKE_DIR} NO_DEFAULT_PATH)

pybind11_add_module(_core
  ${CMAKE_CURRENT_SOURCE_DIR}/src/PTOModule.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/NativeModuleOnline.cpp)

target_compile_definitions(_core PRIVATE
  PTOAS_ONLINE_BUILD=1
  MLIR_PYTHON_PACKAGE_PREFIX=ptoas.mlir.)

target_include_directories(_core PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Link against the shipped, Python-independent DSO living in the package dir.
target_link_directories(_core PRIVATE ${PTOAS_SHIPPED_LIB_DIR})
target_link_libraries(_core PRIVATE PTOASCompiler)

# The cached `_core` is loaded from a directory that is NOT necessarily the
# package dir (see _build_online._compute_cache_dir). ctypes preloads the DSO
# with RTLD_GLOBAL before importing `_core`, so these rpaths are a best-effort
# fallback for the in-package cache case.
if(APPLE)
  set_target_properties(_core PROPERTIES INSTALL_RPATH
    "@loader_path;@loader_path/mlir/_mlir_libs")
else()
  set_target_properties(_core PROPERTIES INSTALL_RPATH
    "$ORIGIN;$ORIGIN/mlir/_mlir_libs")
endif()

install(TARGETS _core LIBRARY DESTINATION ".")

# ---------------------------------------------------------------------------
# The `ptoas.mlir` pybind11 family. These are the genuine upstream MLIR
# bindings; unlike `_core` they cannot use the C-API-only trick and require the
# real PybindAdaptors.h + the full LLVM/MLIR C++ header closure (staged under
# `mlir_include/`) plus their private sibling headers (staged next to the .cpp
# under `src/mlir/`, quote-included). They dynamically link the shipped shared
# LLVMSupport that travels next to libPTOASCompiler.
# ---------------------------------------------------------------------------
set(_ptoas_family_src "${CMAKE_CURRENT_SOURCE_DIR}/src/mlir")

# _mlir core: the 9 upstream binding TUs.
pybind11_add_module(_mlir
  ${_ptoas_family_src}/MainModule.cpp
  ${_ptoas_family_src}/IRAffine.cpp
  ${_ptoas_family_src}/IRAttributes.cpp
  ${_ptoas_family_src}/IRCore.cpp
  ${_ptoas_family_src}/IRInterfaces.cpp
  ${_ptoas_family_src}/IRModule.cpp
  ${_ptoas_family_src}/IRTypes.cpp
  ${_ptoas_family_src}/Pass.cpp
  ${_ptoas_family_src}/Rewrite.cpp)

# _mlirDialectsLLVM: the upstream LLVM dialect binding.
pybind11_add_module(_mlirDialectsLLVM
  ${_ptoas_family_src}/DialectLLVM.cpp)

# _site_initialize_0: our dialect registration TU (repo-owned, staged alongside
# the upstream family sources).
pybind11_add_module(_site_initialize_0
  ${_ptoas_family_src}/PythonRegistration.cpp)

foreach(_ptoas_family_tgt _mlir _mlirDialectsLLVM _site_initialize_0)
  target_compile_definitions(${_ptoas_family_tgt} PRIVATE
    PTOAS_ONLINE_BUILD=1
    MLIR_PYTHON_PACKAGE_PREFIX=ptoas.mlir.)

  # The family header closure lives in its own tree so the real
  # PybindAdaptors.h does not collide with the patched copy `_core` ships under
  # `include/`. Private sibling headers (Globals.h, IRModule.h, ...) are
  # quote-included, so `src/mlir` must also be on the search path.
  target_include_directories(${_ptoas_family_tgt} PRIVATE
    ${_ptoas_family_src}
    ${CMAKE_CURRENT_SOURCE_DIR}/mlir_include)

  # Link the shipped Python-independent DSO (project C-API) plus the shared
  # LLVMSupport that ships next to it; both live in PTOAS_SHIPPED_LIB_DIR's
  # mlir/_mlir_libs, which is also where these extensions install.
  target_link_directories(${_ptoas_family_tgt} PRIVATE ${PTOAS_SHIPPED_LIB_DIR})
  target_link_libraries(${_ptoas_family_tgt} PRIVATE PTOASCompiler LLVMSupport)

  if(APPLE)
    set_target_properties(${_ptoas_family_tgt} PROPERTIES INSTALL_RPATH
      "@loader_path")
  else()
    set_target_properties(${_ptoas_family_tgt} PROPERTIES INSTALL_RPATH
      "$ORIGIN")
  endif()

  # The family extensions live in ptoas/mlir/_mlir_libs/ inside the package.
  install(TARGETS ${_ptoas_family_tgt} LIBRARY DESTINATION "mlir/_mlir_libs")
endforeach()
