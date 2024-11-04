# [Chapter 5: Partial Lowering to Lower-Level Dialects for Optimization](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-5/)

上一章节主要是进行Toy Dialect 自身的 Transform 优化，而在这一章节中，主要进行的是将ToyDialect降低到更低的一层Dialect上。

正常来说  Lowering 会存在多层，如Triton 实现是 `TTIR` -> `TTGIR` -> `LLVMIR` 的路径。

`TTGIR` 就是混杂了多种Dialect， 而LLVM IR 则是收束统一。

这也就是MLIR的特点，Multi—Layer，在多层次的Dialect之间的转换称之为Lowering，一般在上一层做好优化后，然后到下一层进行进一步的优化。本章节选择Affine Dialect作为更底层的Dialect，并且将Toy Dialect中部分的Operation转换为 Affine Dialect。

MLIR 中允许Dialect 混合，即可以存在Toy Dialect、Affine Dialect等其他多种Dialect并存，不过最终到LLVM IR Dialect时，必须都有可以转换过去的Conversion。对于内置的Dialect 可能存在内置方法转换过去，所以我们需要更多关注我们自定的Dialect 如何Lowering 到内置的其他Dialect，或者直接Lowering 到LLVM IR Dialect。

MLIR 具有多种Dialects，而为了在多种Dialect间进行转换，则需要一个统一的转换框架即[`DialectConversion`](https://mlir.llvm.org/docs/DialectConversion/) TODO， 框架将`illegal`标记的Operation为转换成一组`legal` 的Operation。而为李世勇这个框架需要2个必要项和一个可选项

- `Conversion Target`: 对Operation标记是否`legal`, 使用`Rewrite Patterns` 将`illegal` 转换为`legal`
- `Rewrite Patterns`: 如何将`illegal`转换为`legal`。
- `Type Converter`: 可选 将block arguments进行types 转换

这里和上一章节Shape 推导类似，也是要实现一个Pass，而这个Pass则是实现了 Lowering的功能，在InferShape 之后调用。添加进`OpPassManager` 中的Pass，会按照添加进去的顺序，进行执行调用。

目录结构如下：

```
|-- CMakeLists.txt
|-- README.md
|-- affine-lowering.mlir
|-- include
|   |-- AST.h
|   |-- Dialect.h
|   |-- Lexer.h
|   |-- MLIRGen.h
|   |-- Parser.h
|   |-- Passes.h
|   `-- ShapeInferenceInterface.h
|-- mlir
|   |-- Dialect.cpp
|   `-- MLIRGen.cpp
|-- parser
|   `-- AST.cpp
|-- pass
|   |-- CMakeLists.txt
|   |-- ShapeInferenceInterface.td
|   |-- ShapeInferencePass.cpp
|   |-- ToyCombine.cpp
|   |-- LowerToAffineLoops.cpp
|   `-- ToyCombine.td
|-- td
|   |-- CMakeLists.txt
|   `-- Ops.td
`-- toy.cpp
```

## Conversion Target

需要将`Toy` Dialect 转换为 `Affine`、`Arith`、`Fun`c 和 `MemRef` Dialect。

在Lowering的时候 首先需要定义`ConversionTarget`, `ConversionTarget` 可以粗粒度的标记哪些Dialect 是否`legal`，也可以细粒度的标记哪些 Operation 是否`legal`

然后在这个`Target`上应用`RewritePatternSet`, 对于ToyDialect中除了`PrintOp`都添加了Lowering。

Lowering 到更低层触发 则是调用 `applyPartialConversion` 在 `ConversionTarget` 应用 `RewritePatternSet` 的转换Rules。


```c++
void ToyToAffineLoweringPass::runOnOperation() {
  // The first thing to define is the conversion target. This will define the
  // final target for this lowering.
  ConversionTarget target(getContext());

  // We define the specific operations, or dialects, that are legal targets for
  // this lowering. In our case, we are lowering to a combination of the
  // `Affine`, `Arith`, `Func`, and `MemRef` dialects.
  target.addLegalDialect<affine::AffineDialect, BuiltinDialect,
                         arith::ArithDialect, func::FuncDialect,
                         memref::MemRefDialect>();

  // We also define the Toy dialect as Illegal so that the conversion will fail
  // if any of these operations are *not* converted. Given that we actually want
  // a partial lowering, we explicitly mark the Toy operations that don't want
  // to lower, `toy.print`, as `legal`. `toy.print` will still need its operands
  // to be updated though (as we convert from TensorType to MemRefType), so we
  // only treat it as `legal` if its operands are legal.
  target.addIllegalDialect<toy::ToyDialect>();
  target.addDynamicallyLegalOp<toy::PrintOp>([](toy::PrintOp op) {
    return llvm::none_of(op->getOperandTypes(),
                         [](Type type) { return llvm::isa<TensorType>(type); });
  });

  // Now that the conversion target has been defined, we just need to provide
  // the set of patterns that will lower the Toy operations.
  RewritePatternSet patterns(&getContext());
  patterns.add<AddOpLowering, ConstantOpLowering, FuncOpLowering, MulOpLowering,
               PrintOpLowering, ReturnOpLowering, TransposeOpLowering>(
      &getContext());

  // With the target and rewrite patterns defined, we can now attempt the
  // conversion. The conversion will signal failure if any of our `illegal`
  // operations were not converted successfully.
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}
```

## Conversion Patterns

`ConversionTarget` 被定义之后，我们需要定义如何将`illegal` Operation 转换为 `legal`，这里就以`Transpose` 如何lowering为例。

Lowering 从实现上看就是 将 上层的`illegal` 单个Operation, 转换为 0个、1个或者多个较为底层的`legal` 的 `Operation`。

从而上层更加专注于 做上层的优化，下层更加专注于做下层的优化。



```c++

static void lowerOpToLoops(Operation *op, ValueRange operands,
                           PatternRewriter &rewriter,
                           LoopIterationFn processIteration) {
  auto tensorType = llvm::cast<RankedTensorType>((*op->result_type_begin()));
  auto loc = op->getLoc();

  // Insert an allocation and deallocation for the result of this operation.
  auto memRefType = convertTensorToMemRef(tensorType);
  auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

  // Create a nest of affine loops, with one loop per dimension of the shape.
  // The buildAffineLoopNest function takes a callback that is used to construct
  // the body of the innermost loop given a builder, a location and a range of
  // loop induction variables.
  SmallVector<int64_t, 4> lowerBounds(tensorType.getRank(), /*Value=*/0);
  SmallVector<int64_t, 4> steps(tensorType.getRank(), /*Value=*/1);
  affine::buildAffineLoopNest(
      rewriter, loc, lowerBounds, tensorType.getShape(), steps,
      [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
        // Call the processing function with the rewriter, the memref operands,
        // and the loop induction variables. This function will return the value
        // to store at the current index.
        Value valueToStore = processIteration(nestedBuilder, operands, ivs);
        nestedBuilder.create<affine::AffineStoreOp>(loc, valueToStore, alloc,
                                                    ivs);
      });

  // Replace this operation with the generated alloc.
  rewriter.replaceOp(op, alloc);
}

struct TransposeOpLowering : public ConversionPattern {
  TransposeOpLowering(MLIRContext *ctx)
      : ConversionPattern(toy::TransposeOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    lowerOpToLoops(op, operands, rewriter, [loc](OpBuilder &builder, ValueRange memRefOperands, ValueRange loopIvs) {
        // Generate an adaptor for the remapped operands of the
        // TransposeOp. This allows for using the nice named
        // accessors that are generated by the ODS.
        toy::TransposeOpAdaptor transposeAdaptor(memRefOperands);
        Value input = transposeAdaptor.getInput();

        // Transpose the elements by generating a load from the
        // reverse indices.
        SmallVector<Value, 2> reverseIvs(llvm::reverse(loopIvs));
        return builder.create<affine::AffineLoadOp>(loc, input, reverseIvs);
    });
    return success();
  }
};

```

最终Lowering 之后的 IR 如下所示, 除了toy.print以外，其他的toy Dialect全部被转换为其他的Dialect了

```
module {
  func.func @main() {
    %cst = arith.constant 6.000000e+00 : f64
    %cst_0 = arith.constant 5.000000e+00 : f64
    %cst_1 = arith.constant 4.000000e+00 : f64
    %cst_2 = arith.constant 3.000000e+00 : f64
    %cst_3 = arith.constant 2.000000e+00 : f64
    %cst_4 = arith.constant 1.000000e+00 : f64
    %alloc = memref.alloc() : memref<3x2xf64>
    %alloc_5 = memref.alloc() : memref<3x2xf64>
    %alloc_6 = memref.alloc() : memref<2x3xf64>
    affine.store %cst_4, %alloc_6[0, 0] : memref<2x3xf64>
    affine.store %cst_3, %alloc_6[0, 1] : memref<2x3xf64>
    affine.store %cst_2, %alloc_6[0, 2] : memref<2x3xf64>
    affine.store %cst_1, %alloc_6[1, 0] : memref<2x3xf64>
    affine.store %cst_0, %alloc_6[1, 1] : memref<2x3xf64>
    affine.store %cst, %alloc_6[1, 2] : memref<2x3xf64>
    affine.for %arg0 = 0 to 3 {
      affine.for %arg1 = 0 to 2 {
        %0 = affine.load %alloc_6[%arg1, %arg0] : memref<2x3xf64>
        affine.store %0, %alloc_5[%arg0, %arg1] : memref<3x2xf64>
      }
    }
    affine.for %arg0 = 0 to 3 {
      affine.for %arg1 = 0 to 2 {
        %0 = affine.load %alloc_5[%arg0, %arg1] : memref<3x2xf64>
        %1 = arith.mulf %0, %0 : f64
        affine.store %1, %alloc[%arg0, %arg1] : memref<3x2xf64>
      }
    }
    toy.print %alloc : memref<3x2xf64>
    memref.dealloc %alloc_6 : memref<2x3xf64>
    memref.dealloc %alloc_5 : memref<3x2xf64>
    memref.dealloc %alloc : memref<3x2xf64>
    return
  }
}
```
