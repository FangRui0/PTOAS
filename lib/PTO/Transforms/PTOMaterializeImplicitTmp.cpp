// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOMaterializeImplicitTmp.cpp --------------------------------------===//

#include "PTO/Transforms/Passes.h"

#include "PTO/IR/PTO.h"
#include "PTO/IR/PTODialect.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

namespace {

static pto::TileBufConfigAttr makeRowMajorNoneBoxConfig(MLIRContext *ctx) {
  OpBuilder builder(ctx);
  return pto::TileBufConfigAttr::get(
      ctx, pto::BLayoutAttr::get(ctx, pto::BLayout::RowMajor),
      pto::SLayoutAttr::get(ctx, pto::SLayout::NoneBox),
      builder.getI32IntegerAttr(512),
      pto::PadValueAttr::get(ctx, pto::PadValue::Null),
      pto::CompactModeAttr::get(ctx, pto::CompactMode::Null));
}

static unsigned getTCIDstBitWidth(pto::TCIOp op) {
  auto tileTy = dyn_cast<pto::TileBufType>(op.getDst().getType());
  if (!tileTy)
    return 0;
  auto elemTy = dyn_cast<IntegerType>(tileTy.getElementType());
  if (!elemTy)
    return 0;
  return elemTy.getWidth();
}

static pto::TileBufType makeTCITmpType(MLIRContext *ctx, unsigned dstBitWidth) {
  // PTO-ISA TCI A2/A3 vector path needs 768B for b32 dst and 1792B for
  // b16 dst. Use an f32 1xN tmp with the exact minimum capacity.
  int64_t cols = dstBitWidth == 16 ? 448 : 192;
  return pto::TileBufType::get(
      ctx, {1, cols}, Float32Type::get(ctx),
      pto::AddressSpaceAttr::get(ctx, pto::AddressSpace::VEC), {1, cols},
      makeRowMajorNoneBoxConfig(ctx));
}

static std::optional<int64_t> getElemBytes(Type elemTy) {
  unsigned bits = pto::getPTOStorageElemBitWidth(elemTy);
  if (bits == 0 || bits % 8 != 0)
    return std::nullopt;
  return bits / 8;
}

static SmallVector<int64_t, 4> getValidShapeVec(Type ty) {
  if (auto tileTy = dyn_cast<pto::TileBufType>(ty))
    return SmallVector<int64_t, 4>(tileTy.getValidShape().begin(),
                                  tileTy.getValidShape().end());
  return {};
}

static bool validShapesCompatible(ArrayRef<int64_t> lhs,
                                  ArrayRef<int64_t> rhs) {
  if (lhs.size() != rhs.size())
    return false;
  for (auto [l, r] : llvm::zip(lhs, rhs)) {
    if (l != ShapedType::kDynamic && r != ShapedType::kDynamic && l != r)
      return false;
  }
  return true;
}

static bool isRowMajorTile(Value value) {
  auto tileTy = dyn_cast<pto::TileBufType>(value.getType());
  return tileTy && tileTy.getBLayoutValueI32() ==
                       static_cast<int32_t>(pto::BLayout::RowMajor);
}

static bool isColMajorTile(Value value) {
  auto tileTy = dyn_cast<pto::TileBufType>(value.getType());
  return tileTy && tileTy.getBLayoutValueI32() ==
                       static_cast<int32_t>(pto::BLayout::ColMajor);
}

enum class RowExpandMode {
  Unknown,
  Mode1ColMajorScalar,
  Mode2RowMajorBlock,
};

static RowExpandMode classifyTRowExpandBinaryMode(Value src0, Value src1,
                                                  Value dst) {
  auto dstValid = getValidShapeVec(dst.getType());
  auto src0Valid = getValidShapeVec(src0.getType());
  auto src1Valid = getValidShapeVec(src1.getType());
  if (dstValid.size() != 2 || src0Valid.size() != 2 || src1Valid.size() != 2)
    return RowExpandMode::Unknown;

  Value expanded;
  ArrayRef<int64_t> expandedValid;
  if (validShapesCompatible(src0Valid, dstValid)) {
    expanded = src1;
    expandedValid = src1Valid;
  } else if (validShapesCompatible(src1Valid, dstValid)) {
    expanded = src0;
    expandedValid = src0Valid;
  } else {
    return RowExpandMode::Unknown;
  }

  int64_t expandedCols = expandedValid[1];
  if (isColMajorTile(expanded) &&
      (expandedCols == ShapedType::kDynamic || expandedCols == 1))
    return RowExpandMode::Mode1ColMajorScalar;

  auto dstTileTy = dyn_cast<pto::TileBufType>(dst.getType());
  if (!dstTileTy)
    return RowExpandMode::Unknown;
  auto elemBytes = getElemBytes(dstTileTy.getElementType());
  if (!elemBytes || *elemBytes == 0)
    return RowExpandMode::Unknown;
  int64_t expectedMode2Cols = 32 / *elemBytes;
  if (isRowMajorTile(expanded) &&
      (expandedCols == ShapedType::kDynamic ||
       expandedCols == expectedMode2Cols))
    return RowExpandMode::Mode2RowMajorBlock;

  return RowExpandMode::Unknown;
}

static pto::TileBufType makeTRowExpandTmpType(MLIRContext *ctx,
                                              pto::TileBufType dstTy) {
  constexpr int64_t kTmpBytes = 8192;
  std::optional<int64_t> elemBytes = getElemBytes(dstTy.getElementType());
  int64_t cols = elemBytes && *elemBytes > 0 ? kTmpBytes / *elemBytes : 2048;
  return pto::TileBufType::get(
      ctx, {1, cols}, dstTy.getElementType(),
      pto::AddressSpaceAttr::get(ctx, pto::AddressSpace::VEC), {1, cols},
      makeRowMajorNoneBoxConfig(ctx));
}

static void replaceTRowExpandBinaryOpWithTmp(Operation *op, Value src0,
                                             Value src1, Value tmp, Value dst) {
  OpBuilder builder(op);
  OperationState state(op->getLoc(), op->getName());
  state.addOperands({src0, src1, tmp, dst});
  state.addAttribute("operandSegmentSizes",
                     builder.getDenseI32ArrayAttr({1, 1, 1, 1}));
  for (NamedAttribute attr : op->getAttrs()) {
    if (attr.getName() == "operandSegmentSizes")
      continue;
    state.addAttribute(attr.getName(), attr.getValue());
  }
  builder.create(state);
  op->erase();
}

template <typename OpTy>
static LogicalResult materializeTRowExpandTmp(OpTy op, bool requireExplicitTmp,
                                              MLIRContext *ctx) {
  if (op.getTmp())
    return success();
  if (pto::getTargetArch(op.getOperation()) == pto::PTOArch::A5)
    return success();

  RowExpandMode mode =
      classifyTRowExpandBinaryMode(op.getSrc0(), op.getSrc1(), op.getDst());
  if (mode != RowExpandMode::Mode1ColMajorScalar)
    return success();

  if (requireExplicitTmp) {
    return op.emitOpError(
        "requires explicit tmp for A2/A3 row-expand mode 1 when PlanMemory is skipped");
  }

  auto dstTy = dyn_cast<pto::TileBufType>(op.getDst().getType());
  if (!dstTy)
    return op.emitOpError("expects tile_buf dst when materializing implicit tmp");

  OpBuilder builder(op);
  Value tmp =
      builder
          .create<pto::AllocTileOp>(op.getLoc(),
                                    makeTRowExpandTmpType(ctx, dstTy), Value(),
                                    Value(), Value())
          .getResult();
  replaceTRowExpandBinaryOpWithTmp(op.getOperation(), op.getSrc0(), op.getSrc1(),
                                   tmp, op.getDst());
  return success();
}

struct PTOMaterializeImplicitTmpPass
    : public PassWrapper<PTOMaterializeImplicitTmpPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PTOMaterializeImplicitTmpPass)

  PTOMaterializeImplicitTmpPass() = default;
  explicit PTOMaterializeImplicitTmpPass(bool requireExplicitTmp)
      : requireExplicitTmp(requireExplicitTmp) {}

  StringRef getArgument() const final { return "pto-materialize-implicit-tmp"; }
  StringRef getDescription() const final {
    return "Materialize implicit tmp tiles for PTO ops before memplan";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *ctx = func.getContext();
    bool failed = false;

    SmallVector<pto::TCIOp> tciOps;
    func.walk([&](pto::TCIOp op) {
      if (!op.getTmp())
        tciOps.push_back(op);
    });

    for (pto::TCIOp op : tciOps) {
      if (pto::getTargetArch(op.getOperation()) == pto::PTOArch::A5)
        continue;

      if (requireExplicitTmp) {
        op.emitOpError("requires explicit tmp when PlanMemory is skipped");
        failed = true;
        continue;
      }

      OpBuilder builder(op);
      Location loc = op.getLoc();
      auto tmpType = makeTCITmpType(ctx, getTCIDstBitWidth(op));
      Value tmp =
          builder.create<pto::AllocTileOp>(loc, tmpType, Value(), Value(),
                                           Value())
              .getResult();

      auto newOp = builder.create<pto::TCIOp>(
          loc, TypeRange{}, op.getS(), tmp, op.getDst(),
          op.getDescendingAttr());
      for (NamedAttribute attr : op->getAttrs()) {
        if (attr.getName() == "operandSegmentSizes")
          continue;
        newOp->setAttr(attr.getName(), attr.getValue());
      }
      op.erase();
    }

    SmallVector<Operation *> rowExpandOps;
    func.walk([&](Operation *op) {
      if (isa<pto::TRowExpandAddOp, pto::TRowExpandSubOp,
              pto::TRowExpandMulOp, pto::TRowExpandDivOp,
              pto::TRowExpandMaxOp, pto::TRowExpandMinOp>(op))
        rowExpandOps.push_back(op);
    });

    for (Operation *op : rowExpandOps) {
      LogicalResult result =
          llvm::TypeSwitch<Operation *, LogicalResult>(op)
              .Case<pto::TRowExpandAddOp, pto::TRowExpandSubOp,
                    pto::TRowExpandMulOp, pto::TRowExpandDivOp,
                    pto::TRowExpandMaxOp, pto::TRowExpandMinOp>(
                  [&](auto typedOp) {
                    return materializeTRowExpandTmp(typedOp, requireExplicitTmp,
                                                    ctx);
                  })
              .Default([](Operation *) { return success(); });
      if (mlir::failed(result))
        failed = true;
    }

    if (failed)
      signalPassFailure();
  }

private:
  bool requireExplicitTmp = false;
};

} // namespace

std::unique_ptr<Pass>
mlir::pto::createPTOMaterializeImplicitTmpPass(bool requireExplicitTmp) {
  return std::make_unique<PTOMaterializeImplicitTmpPass>(requireExplicitTmp);
}
