file(GLOB_RECURSE SRC_FILES "*.cpp")

add_mlir_dialect_library(PTOTransforms
  ${SRC_FILES}

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/pto

  DEPENDS
  PTOPassesIncGen
  PTOIncGen

  LINK_LIBS PUBLIC
  PTO             # 必须与 lib/pto/IR 中的定义一致
  MLIRIR
  MLIRPass
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRArithDialect
  MLIRAffineDialect
  MLIREmitCDialect
  MLIRTargetCpp       # 处理 EmitC -> C++
  MLIRTransformUtils
  MLIRTransforms
  MLIRTensorDialect
  MLIRSCFToEmitC
  MLIRSupport
)
