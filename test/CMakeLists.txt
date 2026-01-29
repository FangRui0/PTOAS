llvm_canonicalize_cmake_booleans(LLVM_ENABLE_PIC LLVM_ENABLE_PLUGINS)
configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)
add_lit_testsuite(check-pto "Running the PTO regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ptoas FileCheck count not
)
