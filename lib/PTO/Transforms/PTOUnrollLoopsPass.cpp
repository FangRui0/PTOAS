// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOUnrollLoopsPass.cpp ---------------------------------------------===//
//
// Unroll explicitly annotated scf.for loops before LLVM lowering.
//
// Consumption contract for the unroll hint attributes (see PTO.h):
//
//   {pto.unroll = "full"}     - fully unrolled here when the trip count is a
//                               positive constant; otherwise the attribute is
//                               kept and degraded to llvm.loop.unroll.full
//                               metadata by pto-lower-loop-hints.
//   {pto.unroll_factor = N}   - unrolled by N here when the step is a positive
//                               constant (dynamic upper bounds are supported;
//                               an epilogue loop threading live-out values is
//                               generated and annotated
//                               {pto.unroll = "disable"}); otherwise the
//                               attribute is kept and degraded to
//                               llvm.loop.unroll.count metadata.
//   {pto.unroll = "enable"} / {pto.unroll = "disable"}
//                             - never unrolled natively; left untouched for
//                               pto-lower-loop-hints.
//
// Loops without any unroll annotation are never modified.  Diagnostics for
// malformed annotations (unknown values, non-positive factors, conflicting
// attributes) are reported by pto-lower-loop-hints, which is the final
// consumer of the attributes.
//
// Historically this pass only handled {pto.unroll = "full"} inside SIMT
// contexts (pto-unroll-simt-for) to eliminate divergent control flow in
// SIMTVF kernels.  The SIMT restriction has been lifted because the
// annotation is always explicit user intent.
//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"

#include <cstdint>
#include <optional>

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOUNROLLLOOPS
#define GEN_PASS_DEF_PTOUNROLLSIMTFOR
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

#define DEBUG_TYPE "pto-unroll-loops"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

namespace {

/// Compute the constant trip count of *forOp*, or std::nullopt when any of
/// the bounds/step is not a compile-time constant.  Mirrors the arithmetic of
/// the historical pto-unroll-simt-for pass.
static std::optional<int64_t> getStaticTripCount(scf::ForOp forOp) {
  std::optional<int64_t> lb = getConstantIntValue(forOp.getLowerBound());
  std::optional<int64_t> ub = getConstantIntValue(forOp.getUpperBound());
  std::optional<int64_t> step = getConstantIntValue(forOp.getStep());
  if (!lb || !ub || !step || *step <= 0 || *ub <= *lb)
    return std::nullopt;
  int64_t tripCount = (*ub - *lb + *step - 1) / *step;
  if (tripCount <= 0)
    return std::nullopt;
  return tripCount;
}

/// Shared implementation for the pto-unroll-loops pass (and its legacy
/// pto-unroll-simt-for alias).
struct PTOUnrollLoopsImpl {
  explicit PTOUnrollLoopsImpl(int64_t maxFullUnrollTripCount)
      : maxFullUnrollTripCount(maxFullUnrollTripCount) {}

  int64_t maxFullUnrollTripCount;

  /// Try to fully unroll a loop annotated {pto.unroll = "full"}.  Returns
  /// failure (keeping the attribute) when the trip count is not constant so
  /// that pto-lower-loop-hints can degrade the hint to loop metadata.
  LogicalResult tryFullUnroll(scf::ForOp forOp,
                              PatternRewriter &rewriter) const {
    std::optional<int64_t> tripCount = getStaticTripCount(forOp);
    if (!tripCount)
      return failure();

    LLVM_DEBUG(llvm::dbgs()
               << "PTOUnrollLoops: fully unrolling annotated scf.for "
                  "tripCount="
               << *tripCount << " at " << forOp.getLoc() << "\n");

    // The loop is erased on success; capture the location for the guardrail
    // warning beforehand.
    Location loc = forOp.getLoc();
    if (failed(loopUnrollByFactor(forOp, static_cast<uint64_t>(*tripCount))))
      return failure();

    if (maxFullUnrollTripCount >= 0 && *tripCount > maxFullUnrollTripCount)
      mlir::emitWarning(loc)
          << "fully unrolled a loop with trip count " << *tripCount
          << ", which exceeds max-full-unroll-trip-count="
          << maxFullUnrollTripCount;

    return success();
  }

  /// Try to unroll a loop annotated {pto.unroll_factor = N} by N.  Requires a
  /// statically known positive step; dynamic upper bounds are supported and
  /// produce an epilogue loop that threads live-out values.  The epilogue is
  /// annotated {pto.unroll = "disable"} to prevent further unrolling.
  LogicalResult tryFactorUnroll(scf::ForOp forOp, int64_t factor,
                                PatternRewriter &rewriter) const {
    if (factor < 1)
      return failure(); // malformed; pto-lower-loop-hints reports the error

    std::optional<int64_t> step = getConstantIntValue(forOp.getStep());
    if (!step || *step <= 0)
      return failure(); // degraded to the llvm.loop.unroll.count hint

    LLVM_DEBUG(llvm::dbgs() << "PTOUnrollLoops: unrolling annotated scf.for "
                               "by factor="
                            << factor << " at " << forOp.getLoc() << "\n");

    // loopUnrollByFactor clones the original loop to create the epilogue
    // (inserted as the main loop's sibling) without reporting it.  Snapshot
    // the sibling scf.for ops so the epilogue can be found afterwards, and
    // drop the factor attribute up front so neither the unrolled main loop
    // nor the epilogue clone keeps it.  The attribute is restored on failure
    // so the hint can still degrade to loop metadata.
    MLIRContext *ctx = forOp.getContext();
    Block *parentBlock = forOp->getBlock();
    llvm::DenseSet<Operation *> preExisting;
    for (Operation &op : *parentBlock)
      if (isa<scf::ForOp>(&op))
        preExisting.insert(&op);

    IntegerAttr factorAttr =
        IntegerAttr::get(IntegerType::get(ctx, 32), factor);
    forOp->removeAttr(pto::kUnrollFactorAttrName);

    // On success the main loop may have been promoted away; only the parent
    // block and the snapshot remain safe to use.
    if (failed(loopUnrollByFactor(forOp, static_cast<uint64_t>(factor)))) {
      forOp->setAttr(pto::kUnrollFactorAttrName, factorAttr);
      return failure();
    }

    // Annotate the freshly created epilogue loop (if it survived single
    // -iteration promotion) so it is not unrolled again downstream.
    for (Operation &op : *parentBlock) {
      auto sibling = dyn_cast<scf::ForOp>(&op);
      if (!sibling || preExisting.contains(&op))
        continue;
      sibling->setAttr(pto::kUnrollAttrName,
                       StringAttr::get(ctx, pto::kUnrollDisableValue));
    }

    return success();
  }

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const {
    auto unrollAttr = forOp->getAttrOfType<StringAttr>(pto::kUnrollAttrName);
    auto factorAttr =
        forOp->getAttrOfType<IntegerAttr>(pto::kUnrollFactorAttrName);

    // Conflicting annotations: left untouched; pto-lower-loop-hints errors.
    if (unrollAttr && factorAttr)
      return failure();

    // loopUnrollByFactor reports success on empty-body loops without changing
    // them, which would make the greedy driver loop forever (and, for factor
    // unrolling, silently drop the attribute).  Leave empty loops for
    // pto-lower-loop-hints so the hint degrades to loop metadata instead.
    if (llvm::hasSingleElement(forOp.getBody()->getOperations()))
      return failure();

    if (unrollAttr) {
      if (unrollAttr.getValue() == pto::kUnrollFullValue)
        return tryFullUnroll(forOp, rewriter);
      // "enable" / "disable" / unknown values are not unrolled natively.
      return failure();
    }

    if (factorAttr) {
      // Out-of-contract factors (wrong type/width, non-positive) are left
      // untouched; pto-lower-loop-hints reports the error.  Unrolling by 1 is
      // a no-op; forward the count hint to the compiler.
      if (!pto::isValidUnrollFactorAttr(factorAttr) || factorAttr.getInt() < 2)
        return failure();
      return tryFactorUnroll(forOp, factorAttr.getInt(), rewriter);
    }

    return failure();
  }

  LogicalResult run(func::FuncOp func, MLIRContext *ctx) const {
    struct UnrollAnnotatedForPattern : public OpRewritePattern<scf::ForOp> {
      UnrollAnnotatedForPattern(MLIRContext *ctx, const PTOUnrollLoopsImpl *impl)
          : OpRewritePattern(ctx), impl(impl) {}

      LogicalResult matchAndRewrite(scf::ForOp forOp,
                                    PatternRewriter &rewriter) const override {
        return impl->matchAndRewrite(forOp, rewriter);
      }

      const PTOUnrollLoopsImpl *impl;
    };

    RewritePatternSet patterns(ctx);
    patterns.add<UnrollAnnotatedForPattern>(ctx, this);
    FrozenRewritePatternSet frozen(std::move(patterns));

    GreedyRewriteConfig config;
    config.strictMode = GreedyRewriteStrictness::ExistingOps;

    // Only annotated loops ever enter the worklist: a function-wide greedy
    // run would constant-fold / simplify unrelated unannotated ops even when
    // no pattern matches, violating the "no hint -> no IR change" contract.
    // Unrolling an outer loop clones annotated inner loops; the clones are
    // new ops and thus excluded under ExistingOps, so re-walk each round to
    // pick them up until a round makes no more changes (the remaining hints
    // are enable/disable or not natively unrollable and degrade to loop
    // metadata in pto-lower-loop-hints).
    //
    // Run to a true fixpoint rather than capping the rounds: a cap would
    // silently leave hints behind on deeply nested loops, and a leftover
    // "full" hint degrading to metadata violates the native-unroll contract.
    // Every changing round consumes the annotation of at least one loop, so
    // the loop terminates.
    while (true) {
      SmallVector<Operation *, 8> annotated;
      func.walk([&](scf::ForOp forOp) {
        if (forOp->hasAttr(pto::kUnrollAttrName) ||
            forOp->hasAttr(pto::kUnrollFactorAttrName))
          annotated.push_back(forOp);
      });
      if (annotated.empty())
        return success();

      bool changed = false;
      if (failed(applyOpPatternsAndFold(annotated, frozen, config, &changed)))
        return failure();
      if (!changed)
        return success();
    }
  }
};

struct PTOUnrollLoops
    : public pto::impl::PTOUnrollLoopsBase<PTOUnrollLoops> {
  using pto::impl::PTOUnrollLoopsBase<PTOUnrollLoops>::PTOUnrollLoopsBase;

  void runOnOperation() override {
    PTOUnrollLoopsImpl impl(maxFullUnrollTripCount);
    if (failed(impl.run(getOperation(), &getContext())))
      signalPassFailure();
  }
};

/// Legacy alias pass kept under the historical name "pto-unroll-simt-for".
struct PTOUnrollSIMTFor
    : public pto::impl::PTOUnrollSIMTForBase<PTOUnrollSIMTFor> {
  using pto::impl::PTOUnrollSIMTForBase<
      PTOUnrollSIMTFor>::PTOUnrollSIMTForBase;

  void runOnOperation() override {
    PTOUnrollLoopsImpl impl(maxFullUnrollTripCount);
    if (failed(impl.run(getOperation(), &getContext())))
      signalPassFailure();
  }
};

} // namespace

// ---------------------------------------------------------------------------
// Pass constructors
// ---------------------------------------------------------------------------

std::unique_ptr<Pass> mlir::pto::createPTOUnrollLoopsPass() {
  return std::make_unique<PTOUnrollLoops>();
}

std::unique_ptr<Pass> mlir::pto::createPTOUnrollSIMTForPass() {
  return std::make_unique<PTOUnrollSIMTFor>();
}
