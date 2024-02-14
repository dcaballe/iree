// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-global-opt-loop-invariant-code-motion,cse))" --split-input-file %s | FileCheck %s

func.func @hoist_pack_op_with_zero_trip_check(%bound : i32, %src : tensor<100x100xf32>) -> tensor<13x13x8x8xf32> {
  %cst0 = arith.constant 0 : i32
  %cst1 = arith.constant 1 : i32
  %pad0 = arith.constant 0.0 : f32
  %init = arith.constant dense<0.0> : tensor<13x13x8x8xf32>
  %res:2 = scf.while (%iter = %cst0, %val = %init) : (i32, tensor<13x13x8x8xf32>) -> (i32, tensor<13x13x8x8xf32>) {
    %cond = arith.cmpi slt, %iter, %bound : i32
    scf.condition(%cond) %iter, %val : i32, tensor<13x13x8x8xf32>
  } do {
  ^bb0(%arg1: i32, %arg2: tensor<13x13x8x8xf32>):
    %dest = tensor.empty() : tensor<13x13x8x8xf32>
    %pack = tensor.pack %src padding_value(%pad0 : f32) inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %dest : tensor<100x100xf32> -> tensor<13x13x8x8xf32>
    %add = arith.addf %arg2, %pack : tensor<13x13x8x8xf32>
    %next = arith.addi %arg1, %cst1 : i32
    scf.yield %next, %add : i32, tensor<13x13x8x8xf32>
  }
  return %res#1 : tensor<13x13x8x8xf32>
}

// CHECK-LABEL: func.func @hoist_pack_op_with_zero_trip_check
// CHECK-SAME:      (%[[BOUND:.+]]: i32, %[[SRC:.+]]: tensor<100x100xf32>)
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : i32
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : i32
// CHECK-DAG:     %[[PAD:.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:     %[[INIT:.+]] = arith.constant dense<0.000000e+00> : tensor<13x13x8x8xf32>
// CHECK:         %[[PRECOND:.+]] = arith.cmpi slt, %[[C0]], %[[BOUND]] : i32
// CHECK:         %[[RES:.+]]:2 = scf.if %[[PRECOND]] -> (i32, tensor<13x13x8x8xf32>) {
// CHECK:           %[[DEST:.+]] = tensor.empty() : tensor<13x13x8x8xf32>
// CHECK:           %[[PACK:.+]] = tensor.pack %[[SRC]] padding_value(%[[PAD]] : f32)
// CHECK-SAME:          inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %[[DEST]]
// CHECK-SAME:          : tensor<100x100xf32> -> tensor<13x13x8x8xf32>
// CHECK:           %[[LOOP:.+]]:2 = scf.while (%[[ARG2:.+]] = %[[C0]], %[[ARG3:.+]] = %[[INIT]])
// CHECK-SAME:          : (i32, tensor<13x13x8x8xf32>) -> (i32, tensor<13x13x8x8xf32>) {
// CHECK:             %[[ADD:.+]] = arith.addf %[[ARG3]], %[[PACK]] : tensor<13x13x8x8xf32>
// CHECK:             %[[NEXT:.+]] = arith.addi %[[ARG2]], %[[C1]] : i32
// CHECK:             %[[COND:.+]] = arith.cmpi slt, %[[NEXT]], %[[BOUND]] : i32
// CHECK:             scf.condition(%[[COND]]) %[[NEXT]], %[[ADD]] : i32, tensor<13x13x8x8xf32>
// CHECK:           } do {
// CHECK:           ^bb0(%[[ARG4:.+]]: i32, %[[ARG5:.+]]: tensor<13x13x8x8xf32>):
// CHECK:             scf.yield %[[ARG4]], %[[ARG5]] : i32, tensor<13x13x8x8xf32>
// CHECK:           }
// CHECK:           scf.yield %[[LOOP]]#0, %[[LOOP]]#1 : i32, tensor<13x13x8x8xf32>
// CHECK:         } else {
// CHECK:           scf.yield %[[C0]], %[[INIT]] : i32, tensor<13x13x8x8xf32>
// CHECK:         }
// CHECK:         return %[[RES]]#1 : tensor<13x13x8x8xf32>

// -----

func.func @hoist_pack_op_from_do_while(%bound : i32, %src : tensor<100x100xf32>) -> tensor<13x13x8x8xf32> {
  %cst0 = arith.constant 0 : i32
  %cst1 = arith.constant 1 : i32
  %pad0 = arith.constant 0.0 : f32
  %init = arith.constant dense<0.0> : tensor<13x13x8x8xf32>
  %res:2 = scf.while (%iter = %cst0, %val = %init) : (i32, tensor<13x13x8x8xf32>) -> (i32, tensor<13x13x8x8xf32>) {
    %dest = tensor.empty() : tensor<13x13x8x8xf32>
    %pack = tensor.pack %src padding_value(%pad0 : f32) inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %dest : tensor<100x100xf32> -> tensor<13x13x8x8xf32>
    %add = arith.addf %val, %pack : tensor<13x13x8x8xf32>
    %next = arith.addi %iter, %cst1 : i32
    %cond = arith.cmpi slt, %next, %bound : i32
    scf.condition(%cond) %next, %add : i32, tensor<13x13x8x8xf32>
  } do {
  ^bb0(%arg1: i32, %arg2: tensor<13x13x8x8xf32>):
    scf.yield %arg1, %arg2 : i32, tensor<13x13x8x8xf32>
  }
  return %res#1 : tensor<13x13x8x8xf32>
}

// CHECK-LABEL: func.func @hoist_pack_op_from_do_while
// CHECK-SAME:      (%[[BOUND:.+]]: i32, %[[SRC:.+]]: tensor<100x100xf32>)
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : i32
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : i32
// CHECK-DAG:     %[[PAD:.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:     %[[INIT:.+]] = arith.constant dense<0.000000e+00> : tensor<13x13x8x8xf32>
// CHECK-NOT:     scf.if
// CHECK:         %[[DEST:.+]] = tensor.empty() : tensor<13x13x8x8xf32>
// CHECK:         %[[PACK:.+]] = tensor.pack %[[SRC]] padding_value(%[[PAD]] : f32)
// CHECK:             inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %[[DEST]] : tensor<100x100xf32> -> tensor<13x13x8x8xf32>
// CHECK:         %[[LOOP:.+]]:2 = scf.while (%[[ARG2:.+]] = %[[C0]], %[[ARG3:.+]] = %[[INIT]])
// CHECK-SAME:        (i32, tensor<13x13x8x8xf32>) -> (i32, tensor<13x13x8x8xf32>) {
// CHECK:           %[[ADD:.+]] = arith.addf %[[ARG3]], %[[PACK]] : tensor<13x13x8x8xf32>
// CHECK:           %[[NEXT:.+]] = arith.addi %[[ARG2]], %[[C1]] : i32
// CHECK:           %[[COND:.+]] = arith.cmpi slt, %[[NEXT]], %[[BOUND]] : i32
// CHECK:           scf.condition(%[[COND]]) %[[NEXT]], %[[ADD]] : i32, tensor<13x13x8x8xf32>
// CHECK:         } do {
// CHECK:         ^bb0(%[[ARG4:.+]]: i32, %[[ARG5:.+]]: tensor<13x13x8x8xf32>):
// CHECK:           scf.yield %[[ARG4]], %[[ARG5]] : i32, tensor<13x13x8x8xf32>
// CHECK:         }
// CHECK:         return %[[LOOP]]#1 : tensor<13x13x8x8xf32>

// -----

func.func @not_hoist_loop_variant_and_non_leaf_alone(%bound : i32, %src : tensor<100x100xf32>) -> tensor<100x100xf32> {
  %cst0 = arith.constant 0 : i32
  %cst1 = arith.constant 1 : i32
  %pad0 = arith.constant 0.0 : f32
  %bias = arith.constant dense<1.0> : tensor<13x13x8x8xf32>
  %res:2 = scf.while (%iter = %cst0, %val = %src) : (i32, tensor<100x100xf32>) -> (i32, tensor<100x100xf32>) {
    %pack_dest = tensor.empty() : tensor<13x13x8x8xf32>
    %pack = tensor.pack %val padding_value(%pad0 : f32) inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %pack_dest : tensor<100x100xf32> -> tensor<13x13x8x8xf32>
    %add = arith.addf %pack, %bias : tensor<13x13x8x8xf32>
    %unpack_dest = tensor.empty() : tensor<100x100xf32>
    %unpack = tensor.unpack %add inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %unpack_dest : tensor<13x13x8x8xf32> -> tensor<100x100xf32>
    %next = arith.addi %iter, %cst1 : i32
    %cond = arith.cmpi slt, %next, %bound : i32
    scf.condition(%cond) %next, %unpack : i32, tensor<100x100xf32>
  } do {
  ^bb0(%arg1: i32, %arg2: tensor<100x100xf32>):
    scf.yield %arg1, %arg2 : i32, tensor<100x100xf32>
  }
  return %res#1 : tensor<100x100xf32>
}

// CHECK-LABEL: func.func @not_hoist_loop_variant_and_non_leaf_alone
// CHECK-NOT:     tensor.empty
// CHECK-NOT:     tensor.pack
// CHECK-NOT:     tensor.unpack
// CHECK:         scf.while
// CHECK:           %[[PACK_DEST:.+]] = tensor.empty
// CHECK:           tensor.pack {{.*}} into %[[PACK_DEST]]
// CHECK:           %[[UNPACK_DEST:.+]] = tensor.empty
// CHECK:           tensor.unpack {{.*}} into %[[UNPACK_DEST]]
// CHECK:           scf.condition
// CHECK:         } do {
// CHECK:           scf.yield
// CHECK:         }
