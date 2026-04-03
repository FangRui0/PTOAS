# =========================================================
# TileLang DSL package wiring
# =========================================================

set(TILELANG_DSL_PACKAGE_SRC_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/python/tilelang_dsl")
set(TILELANG_DSL_BUILD_ROOT "${CMAKE_BINARY_DIR}/python")
set(TILELANG_DSL_BUILD_PACKAGE_DIR
    "${TILELANG_DSL_BUILD_ROOT}/tilelang_dsl")

add_custom_target(TileLangDSLPackage ALL
  COMMAND ${CMAKE_COMMAND} -E make_directory "${TILELANG_DSL_BUILD_ROOT}"
  COMMAND ${CMAKE_COMMAND} -E remove_directory "${TILELANG_DSL_BUILD_PACKAGE_DIR}"
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          "${TILELANG_DSL_PACKAGE_SRC_DIR}"
          "${TILELANG_DSL_BUILD_PACKAGE_DIR}"
  COMMENT "Staging tilelang_dsl package into build/python"
  VERBATIM
)

install(
  DIRECTORY "${TILELANG_DSL_PACKAGE_SRC_DIR}"
  DESTINATION "."
  COMPONENT PTOAS_Runtime
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pyc" EXCLUDE
)
