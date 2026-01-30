# [关键] 库名重命名为 PTOIR，避免与 LLVM 里的 PTODialect/MLIRPTODialect 冲突
add_mlir_dialect_library(PTOIR
  PTO.cpp
  PTOAttrs.cpp
  PTOTypeDefs.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/PTO

  DEPENDS
  PTOOpsIncGen  # 对应 include/PTO/IR/CMakeLists.txt 里定义的那个目标

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRControlFlowInterfaces
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
)
