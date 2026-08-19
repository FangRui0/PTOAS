// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOLowerLoopHintsPass.cpp ------------------------------------------===//
//
// Translate leftover pto.unroll / pto.unroll_factor loop hint attributes on
// scf.for into LLVM loop metadata reachable form.
//
// pto-unroll-loops consumes the "full" / factor annotations it can unroll
// natively; everything that still carries an annotation when this pass runs
// is forwarded to the compiler as LLVM loop metadata:
//
//   {pto.unroll = "enable"}    -> #llvm.loop_annotation<unroll = <disable = false>>
//   {pto.unroll = "disable"}   -> #llvm.loop_annotation<unroll = <disable = true>>
//   {pto.unroll = "full"}      -> #llvm.loop_annotation<unroll = <full = true>>
//   {pto.unroll_factor = N}    -> #llvm.loop_annotation<unroll = <count = N>>
//
// LLVM 19's convert-scf-to-cf does not propagate llvm.loop_annotation from
// scf.for to the loop latch (that upstream support only exists in newer
// MLIR), so this pass additionally lowers each annotated scf.for to
// control-flow ops itself, attaching the annotation to the latch cf.br.
// Unannotated loops are left for the regular convert-scf-to-cf pass.
// Downstream CF→LLVM lowering preserves branch attributes on llvm.br, and
// MLIR→LLVM IR translation attaches the corresponding !llvm.loop metadata.
//
// This pass must run immediately before convert-scf-to-cf so that no
// intermediate pass can drop the discardable attributes.
//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include <cstdint>

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOLOWERLOOPHINTS
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

#define DEBUG_TYPE "pto-lower-loop-hints"

namespace {

/// Name of the discardable attribute carrying the loop annotation on scf.for.
static constexpr llvm::StringLiteral kLoopAnnotationAttrName =
    "llvm.loop_annotation";

/// Name under which llvm.br / llvm.cond_br declare their loop annotation
/// attribute in ODS.  The MLIR-to-LLVM-IR translation looks the attribute up
/// by this *bare* name (getLoopAnnotationAttr()), so it must be stored without
/// the dialect prefix on the latch branch; convert-cf-to-llvm forwards branch
/// attributes verbatim (op->getAttrs()), preserving the name.
static constexpr llvm::StringLiteral kBranchLoopAnnotationAttrName =
    "loop_annotation";

/// Build the LoopUnrollAttr for one hint, or return nullptr when the hint is
/// malformed (an error has been emitted).
static LLVM::LoopUnrollAttr buildUnrollAttr(scf::ForOp forOp,
                                            StringAttr unrollAttr,
                                            IntegerAttr factorAttr) {
  MLIRContext *ctx = forOp.getContext();
  auto i32 = IntegerType::get(ctx, 32);

  if (unrollAttr) {
    StringRef value = unrollAttr.getValue();
    if (value == pto::kUnrollEnableValue)
      return LLVM::LoopUnrollAttr::get(ctx, BoolAttr::get(ctx, false), {}, {},
                                       {}, {}, {}, {});
    if (value == pto::kUnrollDisableValue)
      return LLVM::LoopUnrollAttr::get(ctx, BoolAttr::get(ctx, true), {}, {},
                                       {}, {}, {}, {});
    if (value == pto::kUnrollFullValue) {
      forOp.emitRemark()
          << "'" << pto::kUnrollAttrName << " = \"full\"' loop was not "
          << "unrolled natively (no constant trip count); forwarding the "
          << "llvm.loop.unroll.full hint to the compiler";
      return LLVM::LoopUnrollAttr::get(ctx, {}, {}, {},
                                       BoolAttr::get(ctx, true), {}, {}, {});
    }
    forOp.emitError() << "unknown '" << pto::kUnrollAttrName << "' value '"
                      << value << "'; expected \"enable\", \"disable\", or "
                      << "\"full\"";
    return nullptr;
  }

  int64_t factor = factorAttr.getInt();
  if (factor < 1) {
    forOp.emitError() << "'" << pto::kUnrollFactorAttrName
                      << "' must be a positive integer, got " << factor;
    return nullptr;
  }
  forOp.emitRemark()
      << "'" << pto::kUnrollFactorAttrName << "' loop was not unrolled "
      << "natively (no constant positive step); forwarding the "
      << "llvm.loop.unroll.count hint to the compiler";
  return LLVM::LoopUnrollAttr::get(ctx, {}, IntegerAttr::get(i32, factor), {},
                                   {}, {}, {}, {});
}

/// Merge *unroll* into the loop's existing llvm.loop_annotation (if any) and
/// set the merged attribute on *forOp*.
static void setMergedLoopAnnotation(scf::ForOp forOp,
                                    LLVM::LoopUnrollAttr unroll) {
  MLIRContext *ctx = forOp.getContext();
  auto existing =
      forOp->getAttrOfType<LLVM::LoopAnnotationAttr>(kLoopAnnotationAttrName);

  LLVM::LoopAnnotationAttr merged;
  if (!existing) {
    merged = LLVM::LoopAnnotationAttr::get(ctx, {}, {}, {}, unroll, {}, {}, {},
                                           {}, {}, {}, {}, {}, {}, {}, {});
  } else {
    if (existing.getUnroll())
      forOp.emitWarning() << "overwriting an existing unroll entry in '"
                          << kLoopAnnotationAttrName << "'";
    merged = LLVM::LoopAnnotationAttr::get(
        ctx, existing.getDisableNonforced(), existing.getVectorize(),
        existing.getInterleave(), unroll, existing.getUnrollAndJam(),
        existing.getLicm(), existing.getDistribute(), existing.getPipeline(),
        existing.getPeeled(), existing.getUnswitch(),
        existing.getMustProgress(), existing.getIsVectorized(),
        existing.getStartLoc(), existing.getEndLoc(),
        existing.getParallelAccesses());
  }
  forOp->setAttr(kLoopAnnotationAttrName, merged);
}

/// Validate the hint attributes on one loop and translate them into an
/// llvm.loop_annotation attribute.  Returns failure when diagnostics were
/// emitted.
static LogicalResult translateLoopHint(scf::ForOp forOp) {
  bool hasUnroll = forOp->hasAttr(pto::kUnrollAttrName);
  bool hasFactor = forOp->hasAttr(pto::kUnrollFactorAttrName);
  if (!hasUnroll && !hasFactor)
    return success();

  auto unrollAttr = forOp->getAttrOfType<StringAttr>(pto::kUnrollAttrName);
  auto factorAttr =
      forOp->getAttrOfType<IntegerAttr>(pto::kUnrollFactorAttrName);

  if (hasUnroll && hasFactor) {
    forOp.emitError() << "'" << pto::kUnrollAttrName << "' and '"
                      << pto::kUnrollFactorAttrName
                      << "' are mutually exclusive on one loop";
    return failure();
  }
  if ((hasUnroll && !unrollAttr) || (hasFactor && !factorAttr)) {
    forOp.emitError() << "loop hint attributes '" << pto::kUnrollAttrName
                      << "' and '" << pto::kUnrollFactorAttrName
                      << "' must be a string and an integer attribute "
                         "respectively";
    return failure();
  }
  // The factor is forwarded into 32-bit LLVM loop metadata, so it must fit
  // the signless i32 contract; a wider attribute would silently truncate
  // (e.g. an i64 2**31 becomes count=-2**31).
  if (factorAttr && !factorAttr.getType().isSignlessInteger(32)) {
    forOp.emitError() << "'" << pto::kUnrollFactorAttrName
                      << "' must be a signless i32 attribute, got "
                      << factorAttr.getType();
    return failure();
  }

  LLVM::LoopUnrollAttr unroll = buildUnrollAttr(forOp, unrollAttr, factorAttr);
  if (!unroll)
    return failure();

  LLVM_DEBUG(llvm::dbgs() << "PTOLowerLoopHints: annotating scf.for at "
                          << forOp.getLoc() << "\n");
  setMergedLoopAnnotation(forOp, unroll);
  forOp->removeAttr(pto::kUnrollAttrName);
  forOp->removeAttr(pto::kUnrollFactorAttrName);
  return success();
}

/// Lower one annotated scf.for to control-flow ops, attaching its
/// LLVM-dialect attributes (llvm.loop_annotation, stored on the latch under
/// the bare ODS name loop_annotation) to the latch cf.br.
///
/// This mirrors convert-scf-to-cf's ForLowering; the latch-attribute copy
/// backports the behavior that upstream MLIR only provides in newer
/// versions.  Unannotated loops are left for the regular conversion pass.
struct LowerAnnotatedForPattern : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    if (!forOp->hasAttr(kLoopAnnotationAttrName))
      return failure();

    Location loc = forOp.getLoc();

    // Start by splitting the block containing the 'scf.for' into two parts.
    // The part before will get the init code, the part after will be the end
    // point.
    auto *initBlock = rewriter.getInsertionBlock();
    auto initPosition = rewriter.getInsertionPoint();
    auto *endBlock = rewriter.splitBlock(initBlock, initPosition);

    // Use the first block of the loop body as the condition block since it is
    // the block that has the induction variable and loop-carried values as
    // arguments.  Split out all operations from the first block into a new
    // block.  Move all body blocks from the loop body region to the region
    // containing the loop.
    auto *conditionBlock = &forOp.getRegion().front();
    auto *firstBodyBlock =
        rewriter.splitBlock(conditionBlock, conditionBlock->begin());
    auto *lastBodyBlock = &forOp.getRegion().back();
    rewriter.inlineRegionBefore(forOp.getRegion(), endBlock);
    auto iv = conditionBlock->getArgument(0);

    // Append the induction variable stepping logic to the last body block and
    // branch back to the condition block.  Loop-carried values are taken from
    // the operands of the loop terminator.
    Operation *terminator = lastBodyBlock->getTerminator();
    rewriter.setInsertionPointToEnd(lastBodyBlock);
    auto stepped =
        rewriter.create<arith::AddIOp>(loc, iv, forOp.getStep()).getResult();
    if (!stepped)
      return failure();

    SmallVector<Value, 8> loopCarried;
    loopCarried.push_back(stepped);
    loopCarried.append(terminator->operand_begin(), terminator->operand_end());
    auto latchBranch =
        rewriter.create<cf::BranchOp>(loc, conditionBlock, loopCarried);

    // Attach the LLVM attributes of the scf.for to the latch branch: LLVM
    // requires loop metadata on the backedge.  The loop annotation is stored
    // under its bare ODS name ("loop_annotation") so that the MLIR-to-LLVM-IR
    // translation picks it up via BrOp::getLoopAnnotationAttr().
    for (const NamedAttribute &attr : forOp->getAttrs()) {
      if (!isa<LLVM::LLVMDialect>(attr.getValue().getDialect()))
        continue;
      StringRef name = attr.getName().getValue();
      if (name == kLoopAnnotationAttrName)
        name = kBranchLoopAnnotationAttrName;
      latchBranch->setAttr(name, attr.getValue());
    }

    rewriter.eraseOp(terminator);

    // Compute loop bounds before branching to the condition.
    rewriter.setInsertionPointToEnd(initBlock);
    Value lowerBound = forOp.getLowerBound();
    Value upperBound = forOp.getUpperBound();
    if (!lowerBound || !upperBound)
      return failure();

    // The initial values of loop-carried values are obtained from the
    // operands of the loop operation.
    SmallVector<Value, 8> destOperands;
    destOperands.push_back(lowerBound);
    llvm::append_range(destOperands, forOp.getInitArgs());
    rewriter.create<cf::BranchOp>(loc, conditionBlock, destOperands);

    // With the body block done, we can fill in the condition block.
    rewriter.setInsertionPointToEnd(conditionBlock);
    auto comparison = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::slt, iv, upperBound);

    rewriter.create<cf::CondBranchOp>(loc, comparison, firstBodyBlock,
                                      ArrayRef<Value>(), endBlock,
                                      ArrayRef<Value>());

    // The result of the loop operation is the values of the condition block
    // arguments except the induction variable on the last iteration.
    rewriter.replaceOp(forOp, conditionBlock->getArguments().drop_front());
    return success();
  }
};

struct PTOLowerLoopHints
    : public pto::impl::PTOLowerLoopHintsBase<PTOLowerLoopHints> {
  using pto::impl::PTOLowerLoopHintsBase<
      PTOLowerLoopHints>::PTOLowerLoopHintsBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    // Step 1: validate and translate pto.* hint attributes into
    // llvm.loop_annotation attributes on scf.for.
    bool failed = false;
    func.walk([&](scf::ForOp forOp) {
      if (mlir::failed(translateLoopHint(forOp)))
        failed = true;
    });
    if (failed) {
      signalPassFailure();
      return;
    }

    // Step 2: lower annotated loops to control flow ourselves, preserving
    // the annotation on the latch branch (LLVM 19's convert-scf-to-cf would
    // drop it).
    //
    // The rewrite is restricted to the annotated loops themselves via
    // applyOpPatternsAndFold + ExistingOps strictness.  A plain
    // applyPatternsAndFoldGreedily on the whole function would also
    // constant-fold every op it visits even when no pattern matches, which
    // silently rewrites unrelated IR (in the VPTO emission pipeline this
    // pass runs after the ub-to-llvm config words are materialized, and
    // folding their arith.ori chains changes the emitted LLVM IR).
    // Annotated loops may nest; the walk is pre-order, so outer loops are
    // rewritten first and inner scf.for ops (moved, not erased, by the outer
    // rewrite) remain valid pointers.
    SmallVector<Operation *, 8> annotated;
    func.walk([&](scf::ForOp forOp) {
      if (forOp->hasAttr(kLoopAnnotationAttrName))
        annotated.push_back(forOp);
    });
    if (annotated.empty())
      return;

    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<LowerAnnotatedForPattern>(ctx);

    GreedyRewriteConfig config;
    config.strictMode = GreedyRewriteStrictness::ExistingOps;

    if (mlir::failed(applyOpPatternsAndFold(
            annotated, FrozenRewritePatternSet(std::move(patterns)), config)))
      signalPassFailure();
  }
};

} // namespace

// ---------------------------------------------------------------------------
// Pass constructor
// ---------------------------------------------------------------------------

std::unique_ptr<Pass> mlir::pto::createPTOLowerLoopHintsPass() {
  return std::make_unique<PTOLowerLoopHints>();
}
