# [Chapter 4: Enabling Generic Transformation with Interfaces](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-4/)

在上一章节中，主要就实现Operation的变换在Op设置`hasCanonicalizer` 从而对Operation进行变换。

而上面的这个方式并通用，本章节讨论的是通用的Operation变换方式，通过[Interfaces](https://mlir.llvm.org/docs/Interfaces/) TODO的方式。

在之前实现的版本中，存在一个问题就是没有shape 推导，这就造成了在Lowering的时候 没法进行，或者说不知道如何生成更具体的底层IR代码。

MLIR 提供了上述机制，能够进行shape 推导， 本章节会将函数进行内联(Inline)， 然后对个Operation 调用进行shape 推导，并且传递。


目录结构

```
|-- CMakeLists.txt
|-- README.md
|-- codegen.toy
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
|   `-- ToyCombine.td
|-- td
|   |-- CMakeLists.txt
|   `-- Ops.td
`-- toy.cpp
```

## Inline

MLIR 以接口 Interfaces的方式 提供语言的某些能力，此处为Inline。使用到的Interface为 `DialectInlinerInterface`。该Interface 虚类，提供了一组`virtual` 函数 hook，从而能够扩展现有的Dialect。

参考文件 `Dialect.cpp`， 方法`isLegalToInline`对于所有的Operation 和Region 都返回true，即都可以进行内联。`handleTerminator` 则是针对`Return` 返回值，直接复制处理，从能内联。`materializeCallConversion` 当进行内联的时候，函数的入参和实参存在类别不匹配的情况下如何处理，这里是直接进行Cast 到目标类型上去，保证了类型一致。

下面的代码主要解决：内联什么Operation，返回值处理和类型不兼容问题的代码。


```
struct ToyInlinerInterface : public DialectInlinerInterface {
    using DialectInlinerInterface::DialectInlinerInterface;

    //===--------------------------------------------------------------------===//
    // Analysis Hooks
    //===--------------------------------------------------------------------===//

    /// All call operations within toy can be inlined.
    bool isLegalToInline(Operation* call, Operation* callable, bool wouldBeCloned) const final
    {
        return true;
    }

    /// All operations within toy can be inlined.
    bool isLegalToInline(Operation*, Region*, bool, IRMapping&) const final
    {
        return true;
    }

    // All functions within toy can be inlined.
    bool isLegalToInline(Region*, Region*, bool, IRMapping&) const final
    {
        return true;
    }

    //===--------------------------------------------------------------------===//
    // Transformation Hooks
    //===--------------------------------------------------------------------===//

    /// Handle the given inlined terminator(toy.return) by replacing it with a new
    /// operation as necessary.
    void handleTerminator(Operation* op, ValueRange valuesToRepl) const final
    {
        // Only "toy.return" needs to be handled here.
        auto returnOp = cast<ReturnOp>(op);
        // Replace the values directly with the return operands.
        assert(returnOp.getNumOperands() == valuesToRepl.size());

        for (const auto& it : llvm::enumerate(returnOp.getOperands()))
            valuesToRepl[it.index()].replaceAllUsesWith(it.value());
    }

    /// Attempts to materialize a conversion for a type mismatch between a call
    /// from this dialect, and a callable region. This method should generate an
    /// operation that takes 'input' as the only operand, and produces a single
    /// result of 'resultType'. If a conversion can not be generated, nullptr
    /// should be returned.
    Operation* materializeCallConversion(OpBuilder& builder, Value input,
        Type resultType,
        Location conversionLoc) const final
    {
        return builder.create<CastOp>(conversionLoc, resultType, input);
    }
};
```

有一点需要注意的是: Inline 只会丢弃 `private-visible` 未被使用的函数定义，即除了`main`函数外，其他函数都应该设置成`private`。

```
mlir::toy::FuncOp mlirGen(FunctionAST& funcAST) {
    ...
    // If this function isn't main, then set the visibility to private.
    if (funcAST.getProto()->getName() != "main")
        function.setPrivate();

    return function;
}
```

然后则是将该Interface绑定到Dialect。

```
void ToyDialect::initialize()
{
    addOperations <
#define GET_OP_LIST
#include "Ops.cpp.inc"
    > ();
    addInterfaces<ToyInlinerInterface>();
}
```

这个时候需要考虑一下函数内联发生的时机，内联发生在函数调用时，即在Toy Dialect 对应于 `generic_call` 时，将`private-visible` 的function 内联进来，则Call需要具备扩展`CallOpInterface`，而function则需要扩展`CallableOpInterface`。 这里将`CallableOpInterface` 替换为`FunctionOpInterface`，具体可以查看`FunctionInterfaces.td`文件，本质上也是扩展了`CallableOpInterface`。

```
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"

def GenericCallOp : Toy_Op<"generic_call",
    [DeclareOpInterfaceMethods<CallOpInterface>]> {
...
}

def FuncOp : Toy_Op<"func",
    [DeclareOpInterfaceMethods<CallableOpInterface>]> {
...
}
-> 可以变更为如下方式
def FuncOp : Toy_Op<"func", [
    FunctionOpInterface, IsolatedFromAbove
  ]> {
...
}
```

当扩展某一interface时，建议查看一下interface定义文件，是否存在需要实现的方法，本接口`CallInterfaces.td` 中定义的`CallOpInterface` 中定义 `let methods = [...]`存在4个方法需要实现。

```c++
/// Return the callee of the generic call operation, this is required by the
/// call interface.
CallInterfaceCallable GenericCallOp::getCallableForCallee()
{
    return (*this)->getAttrOfType<SymbolRefAttr>("callee");
}

/// Set the callee for the generic call operation, this is required by the call
/// interface.
void GenericCallOp::setCalleeFromCallable(CallInterfaceCallable callee)
{
    (*this)->setAttr("callee", callee.get<SymbolRefAttr>());
}

/// Get the argument operands to the called function, this is required by the
/// call interface.
Operation::operand_range GenericCallOp::getArgOperands() { return getInputs(); }

/// Get the argument operands to the called function as a mutable range, this is
/// required by the call interface.
MutableOperandRange GenericCallOp::getArgOperandsMutable()
{
    return getInputsMutable();
}
```

至此，已经完成了使用内置Inline Interface 一个Operation 需要做的步骤，这个时候只要调用Pass就行

```c++
mlir::PassManager pm(module.get()->getName());
pm.addPass(mlir::createInlinerPass());
```

当完成以上步骤之后，就可以发现`multiply_transpose`函数被内嵌到`main`函数之中，从而没有了函数调用的开销。

这个时候存在一个问题就是 `materializeCallConversion` 需要使用到'CastOp', 补充一下这个CastOp实现。

CastOp 实现了`CastOpInterface`接口， 并且输入输出结果shape一致。

```
def CastOp : Toy_Op<"cast", [
     DeclareOpInterfaceMethods<CastOpInterface>,
     DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
     Pure,
     SameOperandsAndResultShape
  ]> {
  let summary = "shape cast operation";
  let description = [{
    The "cast" operation converts a tensor from one type to an equivalent type
    without changing any data elements. The source and destination types must
    both be tensor types with the same element type. If both are ranked, then
    shape is required to match. The operation is invalid if converting to a
    mismatching constant dimension.
  }];

  let arguments = (ins F64Tensor:$input);
  let results = (outs F64Tensor:$output);

  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($output)";
}

```

同样的查看`CastInterfaces.td` 可以知道存在方法`areCastCompatible`需要被实现。该方法主要判断的是输入和输出之间是否能够Cast过去，典型的比如校验输入和输出的个数。

```c++
bool CastOp::areCastCompatible(TypeRange inputs, TypeRange outputs)
{
    if (inputs.size() != 1 || outputs.size() != 1)
        return false;

    // The inputs must be Tensors with the same element type.
    TensorType input = llvm::dyn_cast<TensorType>(inputs.front());
    TensorType output = llvm::dyn_cast<TensorType>(outputs.front());

    if (!input || !output || input.getElementType() != output.getElementType())
        return false;

    // The shape is required to match if both types are ranked.
    return !input.hasRank() || !output.hasRank() || input == output;
}
```

如果当前执行 `./toy ../codegen.toy -emit=mlir -opt`, 则调用前后的差异如下所示：

```
module {
  toy.func private @multiply_transpose(%arg0: tensor<*xf64>, %arg1: tensor<*xf64>) -> tensor<*xf64> {
    %0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
    %1 = toy.transpose(%arg1 : tensor<*xf64>) to tensor<*xf64>
    %2 = toy.mul %0, %1 : tensor<*xf64>
    toy.return %2 : tensor<*xf64>
  }
  toy.func @main() {
    %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
    %1 = toy.reshape(%0 : tensor<2x3xf64>) to tensor<2x3xf64>
    %2 = toy.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<6xf64>
    %3 = toy.reshape(%2 : tensor<6xf64>) to tensor<2x3xf64>
    %4 = toy.generic_call @multiply_transpose(%1, %3) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
    %5 = toy.generic_call @multiply_transpose(%3, %1) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
    toy.print %5 : tensor<*xf64>
    toy.return
  }
}
```

转换为，可以看到`multiply_transpose` 消失了，同时增加了连个CastOp

```
module {
  toy.func @main() {
    %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
    %1 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
    %2 = toy.cast %1 : tensor<2x3xf64> to tensor<*xf64>
    %3 = toy.cast %0 : tensor<2x3xf64> to tensor<*xf64>
    %4 = toy.transpose(%2 : tensor<*xf64>) to tensor<*xf64>
    %5 = toy.transpose(%3 : tensor<*xf64>) to tensor<*xf64>
    %6 = toy.mul %4, %5 : tensor<*xf64>
    toy.print %6 : tensor<*xf64>
    toy.return
  }
}
```

以上就是开发Inline 功能所需要实现的全部代码。但是此时Operation的shape 并没有被推理到，需要增加`Operation`的shape 推理功能实现。


## Shape Inference


### Custom Interface
当函数内联进来之后，就可以进行`Operation`的shape 推理功能的实现。

这个时候要自己实现一个新的自定义Interface `ShapeInferenceOpInterface`, 即Toy Dialect Shape Inference推理的接口。 这个接口有一个存在一个方法`inferShapes`，这个也是`Ops.td`中定义的`Opeariton`需要实现的方法。

```
def ShapeInferenceOpInterface : OpInterface<"ShapeInference"> {
  let description = [{
    Interface to access a registered method to infer the return types for an
    operation that can be used during type inference.
  }];

  let methods = [
    InterfaceMethod<"Infer and set the output shape for the current operation.",
                    "void", "inferShapes">
  ];
}
```

参考Ops.td 中Mul 的重新定义


```
def MulOp : Toy_Op<"mul",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]>
```

而在Dialect.cpp 中增加Interface 的method 实现

```c++
void MulOp::inferShapes() { getResult().setType(getLhs().getType()); }
```

Operation 能够推导shape了，此时需要增加Pass 能够对每个Operation调用`inferShapes`方法。

### Custom Pass

实现`ShapeInferencePass` 则需要继承`mlir::PassWrapper`即可，同时实现其`runOperation`方法，该方法主要是将存在`UnrankTensorType` 通过调用Op的`inferShapes`方法推导出shape。

```c++
struct ShapeInferencePass
    : public mlir::PassWrapper<ShapeInferencePass, OperationPass<toy::FuncOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ShapeInferencePass)

    void runOnOperation() override
    {
        auto f = getOperation();
        llvm::SmallPtrSet<mlir::Operation*, 16> opWorklist;
        f.walk([&](mlir::Operation * op) {
            if (returnsDynamicShape(op))
                opWorklist.insert(op);
        });

        while (!opWorklist.empty()) {
            ...
            if (auto shapeOp = dyn_cast<ShapeInference>(op))
                shapeOp.inferShapes();
            ...
        }
    }
    ...
};
```

然后创建Pass的的方法`createShapeInferencePass`

```c++
std::unique_ptr<mlir::Pass> mlir::toy::createShapeInferencePass() {
  return std::make_unique<ShapeInferencePass>();
}
```

最后通过将Pass添加到`PassManager`中完成调用`Pass`的触发。

```c++
mlir::OpPassManager& optPM = pm.nest<mlir::toy::FuncOp>();
        optPM.addPass(mlir::toy::createShapeInferencePass());
```

最终优化后的ir可以看到所有的shape 都被推导了。


```
module {
  toy.func @main() {
    %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
    %1 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
    %2 = toy.mul %1, %1 : tensor<3x2xf64>
    toy.print %2 : tensor<3x2xf64>
    toy.return
  }
}
```




























在 CMake 中，以下变量用于表示不同的目录路径，它们的含义和用途如下：

1. CMAKE_SOURCE_DIR
定义：表示顶层 CMakeLists.txt 所在的目录（项目的根目录）。
用途：通常用于访问项目中任何位置的文件或目录，尤其是在需要引用项目根目录下的文件时。
2. CMAKE_CURRENT_SOURCE_DIR
定义：表示当前正在处理的 CMakeLists.txt 文件所在的目录。
用途：当 CMake 处理多层目录时，可以用这个变量获取当前目录的路径。这在定义库、可执行文件或查找文件时非常有用，尤其是在子目录中。
3. CMAKE_CURRENT_BINARY_DIR
定义：表示当前正在处理的 CMakeLists.txt 文件的构建目录。
用途：通常用于生成构建相关的文件，如目标输出、临时文件等。每个子目录可以有自己的构建目录，便于管理。
总结
CMAKE_SOURCE_DIR：总是指向项目的根目录，适用于全局引用。
CMAKE_CURRENT_SOURCE_DIR：指向当前正在处理的 CMakeLists.txt 的所在目录，适用于子目录中的特定引用。
CMAKE_CURRENT_BINARY_DIR：指向当前正在处理的 CMakeLists.txt 的构建目录，适用于生成构建文件。
示例
假设项目结构如下：

css
复制代码
project_root/
├── CMakeLists.txt
├── src/
│   ├── CMakeLists.txt
│   └── main.cpp
└── include/
    └── myheader.h
在 project_root/CMakeLists.txt 中：

CMAKE_SOURCE_DIR 为 project_root/
CMAKE_CURRENT_SOURCE_DIR 在根目录时也是 project_root/
CMAKE_CURRENT_BINARY_DIR 为构建目录（例如 project_root/build/）。
在 project_root/src/CMakeLists.txt 中：

CMAKE_SOURCE_DIR 仍然是 project_root/
CMAKE_CURRENT_SOURCE_DIR 为 project_root/src/
CMAKE_CURRENT_BINARY_DIR 为 project_root/build/src/（假设构建目录与源目录一致）。
这种区分可以帮助在多层目录的项目中有效管理文件和路径。