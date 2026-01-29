add_llvm_executable(ptoas
  ptoas.cpp
)

set_target_properties(ptoas PROPERTIES OUTPUT_NAME "ptoas")

target_link_libraries(ptoas PRIVATE
  PTO             # 链接 Dialect 库
  PTOTransforms       # 链接 Transforms 库 (add_mlir_dialect_library 生成的无前缀别名或 obj 库)
  MLIRMlirOptMain
  MLIRParser
  MLIRSupport
  MLIRTransforms
  MLIREmitCDialect
  MLIRTargetCpp
)
