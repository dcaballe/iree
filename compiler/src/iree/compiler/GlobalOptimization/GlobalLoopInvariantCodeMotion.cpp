// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/GlobalOptimization/PassDetail.h"
#include "iree/compiler/GlobalOptimization/Passes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/TopologicalSortUtils.h"

#include "llvm/Support/Debug.h"

using namespace mlir;

#define DEBUG_TYPE "global-loop-invariant-code-motion"
#define LICM_DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")

// Check if the op is a root op we always want to hoist.
static bool isHoistableRootOp(Operation *op) {
  // Currently it's limited to a small set of ops related to constant pack op.
  return isa<tensor::PackOp>(op);
}

// Check if the op is hoistable (but we might not want to hoist it alone).
static bool isHoistableOp(Operation *op) {
  // Currently it's limited to a small set of ops related to constant pack op.
  return op->hasTrait<OpTrait::ConstantLike>() || isa<tensor::EmptyOp>(op) ||
         isHoistableRootOp(op);
}

// Check if the op and its producers are loop invariants and hoistable. Results
// are cached in hoistableOpMap to avoid repeated traversals.
static bool isBackwardSliceHoistable(
    LoopLikeOpInterface loopOp, Operation *op,
    llvm::SmallDenseMap<Operation *, bool> &hoistableOpMap) {
  // First check if the op has been analyzed.
  if (hoistableOpMap.contains(op))
    return hoistableOpMap[op];

  // Currently we don't handle implicit captures, so don't hoist ops with
  // regions.
  if (op->getNumRegions() > 0) {
    LLVM_DEBUG(LICM_DBGS() << "Non-hoistable: " << *op << "\n");
    hoistableOpMap[op] = false;
    return false;
  }

  bool hoistable = true;
  // Check if all producers are hoistable.
  for (OpOperand &operand : op->getOpOperands()) {
    Value value = operand.get();
    // Ignore values defined outside the loop.
    if (loopOp.isDefinedOutsideOfLoop(value))
      continue;

    Operation *producer = value.getDefiningOp();
    // If the producer is not an operation, don't hoist it; otherwise
    // recursively check if the producer is hoistable.
    if (!producer ||
        !isBackwardSliceHoistable(loopOp, producer, hoistableOpMap)) {
      hoistable = false;
      break;
    }
  }
  hoistableOpMap[op] = hoistable;

  LLVM_DEBUG(LICM_DBGS() << (hoistable ? "Hoistable: " : "Non-hoistable: ")
                         << *op << "\n");
  return hoistable;
}

// Hoist ops and their backward slices out of the loop.
static LogicalResult hoistOps(LoopLikeOpInterface loopOp,
                              SmallVectorImpl<Operation *> &opsToHoist) {
  llvm::SmallDenseSet<Operation *> visitedOps;
  SmallVector<Operation *> worklist(opsToHoist.begin(), opsToHoist.end());
  // Collect ops and their backward slices.
  while (!worklist.empty()) {
    Operation *op = worklist.back();
    worklist.pop_back();

    if (visitedOps.contains(op))
      continue;
    visitedOps.insert(op);

    assert(op->getNumRegions() == 0 && "don't expect ops with regions yet");

    for (OpOperand &operand : op->getOpOperands()) {
      if (Operation *producer = operand.get().getDefiningOp()) {
        // Only include producers inside the loop.
        if (loopOp->isAncestor(producer))
          worklist.push_back(producer);
      }
    }
  }

  // Sort the collected ops in topological order.
  SmallVector<Operation *> orderedOpsToHoist = llvm::to_vector(visitedOps);
  mlir::computeTopologicalSorting(orderedOpsToHoist);

  // Hoist ops in topological order.
  for (Operation *op : orderedOpsToHoist) {
    loopOp.moveOutOfLoop(op);
  }
  return success();
}

static LogicalResult hoistLoopInvariants(LoopLikeOpInterface loopOp,
                                         RewriterBase &rewriter) {
  // First find the root ops can be hoisted. The root op needs to satisfy:
  // 1. It is a root op having benefits to be hoisted (e.g. tensor.pack)
  // 2. Its backward slice can be hoisted (e.g. they are loop invariant)
  SmallVector<Operation *> rootOpsToHoist;
  llvm::SmallDenseMap<Operation *, bool> hoistableOpMap;
  for (Region *region : loopOp.getLoopRegions()) {
    // Consider only the top-level ops in the region.
    for (Operation &rootOp : region->getOps()) {
      if (!isHoistableRootOp(&rootOp))
        continue;

      if (isBackwardSliceHoistable(loopOp, &rootOp, hoistableOpMap)) {
        LLVM_DEBUG(LICM_DBGS() << "Found hoistable root: " << rootOp << "\n");
        rootOpsToHoist.push_back(&rootOp);
      }
    }
  }
  if (rootOpsToHoist.empty())
    return success();

  // Wrap the loop in zero-trip-check so the hoisted ops will only run when the
  // loop condition is ever satisfied.
  FailureOr<LoopLikeOpInterface> wrappedLoop =
      TypeSwitch<Operation *, FailureOr<LoopLikeOpInterface>>(
          loopOp.getOperation())
          .Case<scf::WhileOp>([&](scf::WhileOp op) {
            return scf::wrapWhileLoopInZeroTripCheck(op, rewriter);
          })
          .Default([&](Operation *op) { return failure(); });
  if (failed(wrappedLoop))
    return failure();

  return hoistOps(*wrappedLoop, rootOpsToHoist);
}

namespace mlir::iree_compiler::GlobalOptimization {

namespace {

struct GlobalLoopInvariantCodeMotionPass
    : public GlobalLoopInvariantCodeMotionBase<
          GlobalLoopInvariantCodeMotionPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    FunctionOpInterface funcOp = getOperation();

    SmallVector<LoopLikeOpInterface> candidateLoops;
    // Candidate loops are visited in post-order so a loop invariant has chances
    // to move across multiple loop levels.
    funcOp.walk([&](LoopLikeOpInterface op) {
      // Check if the loop type is supported.
      if (isa<scf::WhileOp>(op))
        candidateLoops.push_back(op);
      return;
    });

    IRRewriter rewriter(context);
    for (auto loopOp : candidateLoops) {
      if (failed(hoistLoopInvariants(loopOp, rewriter)))
        return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createGlobalLoopInvariantCodeMotionPass() {
  return std::make_unique<GlobalLoopInvariantCodeMotionPass>();
}
} // namespace mlir::iree_compiler::GlobalOptimization
