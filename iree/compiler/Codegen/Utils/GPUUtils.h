// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_UTILS_GPUUTILS_H_
#define IREE_COMPILER_CODEGEN_UTILS_GPUUTILS_H_

#include "iree/compiler/Codegen/Utils/Utils.h"

namespace mlir {
namespace iree_compiler {

static constexpr int32_t kNumGPUDims = 3;
static constexpr int32_t kWarpSize = 32;

llvm::SmallVector<linalg::ProcInfo, 2> getGPUThreadIdsAndCounts(
    OpBuilder &builder, Location loc, unsigned numDims,
    llvm::ArrayRef<int64_t> workgroupSize);

/// Compute subgroup ID. CUDA doesn't have a subgroupId equivalent so we are are
/// computing the subgroup ID based on the threadID.
/// When tiling to warp we assume each warp is full and we pick a workgroup
/// size so that `workgroupSize.x % warpSize == 0`. This is why we can have
/// warpId = { threadId.x / warpSize, threadId.y, threadId.z }.
llvm::SmallVector<linalg::ProcInfo, 2> getSubgroupIdsAndCounts(
    OpBuilder &builder, Location loc, unsigned numDims,
    llvm::ArrayRef<int64_t> numSubgroups);

/// Return the workgroup size associated to the funcOp entry point.
std::array<int64_t, 3> getWorkgroupSize(func::FuncOp funcOp);

/// Return true if we can use all threads to perform vectorized load/store of
/// the given `shape`.
bool canPerformVectorAccessUsingAllThreads(ArrayRef<int64_t> shape,
                                           int64_t threadCount,
                                           int64_t vectorSize);

/// Allocate GPU workgroup memory matching the given `subview`. If there are
/// dynamic dimensions, the bounds are in `sizeBounds`.
Optional<Value> allocateWorkgroupMemory(OpBuilder &builder,
                                        memref::SubViewOp subview,
                                        ArrayRef<Value> sizeBounds,
                                        DataLayout &);

/// Deallocate GPU workgroup memory behind `buffer`.
LogicalResult deallocateWorkgroupMemory(OpBuilder &, Value buffer);

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CODEGEN_UTILS_GPUUTILS_H_
