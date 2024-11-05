# [Chapter 3: High-level Language-Specific Analysis and Transformation](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-3/)


本章节主要讲了 如何进行对MLIR 进行优化，在Lowering 到更底层的LLVM 时，LLVM更加底层，抛弃了shape等上层信息，所以要在底层进行IR 优化更难，所以会在上层的一些IR进行优化。

这些反映了MLIR的设计思路，逐层抽象，逐层优化，越到底层越细节，而上层更抽象。

上层抽象对于全局具有更广的维度，进行一些粗粒度的抽象，而底层则是对细节的优化。

本文主要对"Operation" `transpose` 和 `reshape` 进行优化。

在MLIR 框架中提供两种方式来实现 `pattern-match` 的变换: 1. C++ 模式匹配和改写，2. 声明式  rule-base 模式匹配和改写([Declarative Rewrite Rules, DDR](https://mlir.llvm.org/docs/DeclarativeRewrites/)) TODO, 使用该种方式需要"Operation" 通过ODS 方式定义，反正也没教过怎么手写一个Op，所以就这样吧。


代码结构，主要是新增了`pass` 目录和修改了 `Ops.td` 中Op的一些定义，还有修改了toy.cpp 调用了Pass。


```
|-- CMakeLists.txt
|-- README.md
|-- include
|   |-- AST.h
|   |-- Dialect.h
|   |-- Lexer.h
|   |-- MLIRGen.h
|   `-- Parser.h
|-- mlir
|   |-- Dialect.cpp
|   `-- MLIRGen.cpp
|-- parser
|   `-- AST.cpp
|-- pass
|   |-- CMakeLists.txt
|   |-- ToyCombine.cpp
|   `-- ToyCombine.td
|-- td
|   |-- CMakeLists.txt
|   `-- Ops.td
|-- toy.cpp
|-- transpose_transpose.toy
`-- trivial_reshape.toy
```

## 调用Pass的方式

在本文中调用触发Pass的方式为，通过触发内置的`CanonicalizerPass` 来调用新增的Pass

```c++
    if (enableOpt) {
        mlir::PassManager pm(module.get()->getName());

        // Apply any generic pass manager command line options and run the pipeline.
        if (mlir::failed(mlir::applyPassManagerCLOptions(pm)))
            return 4;

        // Add a run of the canonicalizer to optimize the mlir module.
        pm.addNestedPass<mlir::toy::FuncOp>(mlir::createCanonicalizerPass());

        if (mlir::failed(pm.run(*module)))
            return 4;
    }
```


## 改写 Op 属性

增加了一些Operation的属性，参考[DefiningDialects](https://mlir.llvm.org/docs/DefiningDialects)

```
def ReshapeOp : Toy_Op<"reshape", [Pure]> {
  let summary = "tensor reshape operation";
  let description = [{
    Reshape operation is transforming its input tensor into a new tensor with
    the same number of elements but different shapes. For example:

    ```mlir
       %0 = toy.reshape (%arg1 : tensor<10xf64>) to tensor<5x2xf64>
    ```
  }];

  let arguments = (ins F64Tensor:$input);

  // We expect that the reshape operation returns a statically shaped tensor.
  let results = (outs StaticShapeTensorOf<[F64]>);

  let assemblyFormat = [{
    `(` $input `:` type($input) `)` attr-dict `to` type(results)
  }];

  // Enable registering canonicalization patterns with this operation.
  let hasCanonicalizer = 1;
}
```

`[Pure]` 是一种 操作接口（Operation Interface），用于标记操作的特性。这里的 Pure 指定了 ConstantOp 操作是一个 纯操作（即不产生副作用的操作）。

纯操作通常满足以下条件：

- 操作是确定性的：对于相同的输入总会生成相同的输出。
- 操作不依赖外部状态，也不会改变任何状态（没有副作用）。

在 MLIR 中，为操作添加 Pure 接口可以帮助优化器更好地分析和优化代码，因为优化器可以安全地重新排序、合并或删除这些操作，而不影响程序的语义

`hasCanonicalizer`用于标记操作的属性，表示该操作具有 规范化规则（canonicalization rules）。这些规则定义了如何通过模式（patterns）将操作转换成更简化的等价形式，以优化中间表示（IR）

- 启用规范化模式：标记 hasCanonicalizer 的操作会关联一组模式匹配规则，这些规则会在 MLIR 优化过程中自动应用。
- 优化 IR：规范化规则会将冗余、无用或复杂的操作替换成等价且更简单的操作。例如，将常量加法运算直接折叠成单一常量。
- 改善编译效率：通过消除无效或冗余操作，hasCanonicalizer 帮助简化 IR，提高编译器优化和代码生成的效率。


## C++ 模式匹配和改写

参考case `transpose(transpose(X)) -> X`， 两次`transpose`等价于没有进行变换的思想。

一句话说，如果`hasCanonicalizer=1` 的话，在c++就要实现getCanonicalizationPatterns，返回对应实现的Pattern Pass。

逻辑很简单，就是Transpose Op A的输入B是Transpose Op，如果为Transpose A将替换为B的输入，等同于删除两个Op。

```c++
struct SimplifyRedundantTranspose : public mlir::OpRewritePattern<TransposeOp> {
  /// We register this pattern to match every toy.transpose in the IR.
  /// The "benefit" is used by the framework to order the patterns and process
  /// them in order of profitability.
  SimplifyRedundantTranspose(mlir::MLIRContext *context)
      : OpRewritePattern<TransposeOp>(context, /*benefit=*/1) {}

  /// This method attempts to match a pattern and rewrite it. The rewriter
  /// argument is the orchestrator of the sequence of rewrites. The pattern is
  /// expected to interact with it to perform any changes to the IR from here.
  llvm::LogicalResult
  matchAndRewrite(TransposeOp op,
                  mlir::PatternRewriter &rewriter) const override {
    // Look through the input of the current transpose.
    mlir::Value transposeInput = op.getOperand();
    TransposeOp transposeInputOp = transposeInput.getDefiningOp<TransposeOp>();

    // Input defined by another transpose? If not, no match.
    if (!transposeInputOp)
      return failure();

    // Otherwise, we have a redundant transpose. Use the rewriter.
    rewriter.replaceOp(op, {transposeInputOp.getOperand()});
    return success();
  }
};

/// Register our patterns as "canonicalization" patterns on the TransposeOp so
/// that they can be picked up by the Canonicalization framework.
void TransposeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.add<SimplifyRedundantTranspose>(context);
}
```


## DDR 模式匹配和改写

参考case:

- `reshape(reshape(X)) -> X`， 两次`reshape`操作最终只有一次效果。(DDR框架内置)
- `reshape(constant(X, shape1), shape2) -> constant(X, shape2)`，这个`reshape`操作也是冗余的。(通过DDR中嵌入c++代码)
- `reshape(X) -> X`, 其中`reshape` 的shape和X的原始shape一致，属于冗余`reshape`。(带约束的模式匹配)

我们可以在上面的例子中可以看到C++实现方式的模式匹配Pass实际上有很多代码是模板代码，所以在这个基础上，主要给出 匹配的模板和改写目标，借助DDR 可以实现帮助生成模板代码，而不用手写。

对于上面三种形式的

```
//===----------------------------------------------------------------------===//
// Basic Pattern-Match and Rewrite
//===----------------------------------------------------------------------===//
// Reshape(Reshape(x)) = Reshape(x)
def ReshapeReshapeOptPattern : Pat<(ReshapeOp(ReshapeOp $arg)), (ReshapeOp $arg)>;

//===----------------------------------------------------------------------===//
// Pattern-Match and Rewrite using Native Code Call
//===----------------------------------------------------------------------===//

// Native Code Calls may be used for more complex transformations using inline
// C++ and C++ helper functions.

// Reshape(Constant(x)) = x'
def ReshapeConstant :
  NativeCodeCall<"$0.reshape(::llvm::cast<ShapedType>($1.getType()))">;
def FoldConstantReshapeOptPattern : Pat<
  (ReshapeOp:$res (ConstantOp $arg)),
  (ConstantOp (ReshapeConstant $arg, $res))>;

//===----------------------------------------------------------------------===//
// Pattern-Match and Rewrite with Constraints
//===----------------------------------------------------------------------===//

// DRR allows for constraint checking when the transformation is conditional
// on operand properties.

// Reshape(x) = x, where input and output shapes are identical
def TypesAreIdentical : Constraint<CPred<"$0.getType() == $1.getType()">>;
def RedundantReshapeOptPattern : Pat<
  (ReshapeOp:$res $arg), (replaceWithValue $arg),
  [(TypesAreIdentical $res, $arg)]>;
```
