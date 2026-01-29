add_mlir_dialect_library(PTO
  PTO.cpp
  PTOAttrs.cpp
  PTOTypeDefs.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/pto

  DEPENDS
  PTOIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRControlFlowInterfaces
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
  MLIRCastInterfaces
  MLIREmitCDialect
)
