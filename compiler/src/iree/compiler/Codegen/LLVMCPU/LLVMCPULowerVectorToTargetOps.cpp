// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMCPU/LLVMCPUPasses.h"
#include "iree/compiler/Codegen/PassDetail.h"
// #include "llvm/Support/CommandLine.h"
// #include "mlir/Dialect/Affine/IR/AffineOps.h"
// #include "mlir/Dialect/Arith/IR/Arith.h"
// #include "mlir/Dialect/Linalg/Transforms/Transforms.h"
// #include "mlir/Dialect/Linalg/Utils/Utils.h"
// #include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/X86Vector/Transforms.h"
// #include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
// #include "mlir/Dialect/SCF/Transforms/Transforms.h"
// #include "mlir/IR/Iterators.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-llvmcpu-lower-vector-to-target-ops"

using namespace mlir;
using namespace mlir::iree_compiler;

namespace {

struct LLVMCPULowerVectorToTargetOps
    : LLVMCPULowerVectorToTargetOpsBase<LLVMCPULowerVectorToTargetOps> {

  void runOnOperation() override;
};

void LLVMCPULowerVectorToTargetOps::runOnOperation(){
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  x86vector::avx512::populateMaskBroadcastLoweringPatterns(patterns);
  (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
}

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::iree_compiler::createLLVMCPULowerVectorToTargetOps() {
  return std::make_unique<LLVMCPULowerVectorToTargetOps>();
}
