# [Chapter 2: Emit Basic MLIR](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-2/)

在完成Chapter 1生成了AST，而这次目标是将AST转换为MLIR Dialect，以方便后续的优化。

目录结构

```
|-- CMakeLists.txt
|-- README.md
|-- build
|   |-- include
|   |   |-- Dialect.cpp.inc
|   |   |-- Dialect.h.inc
|   |   |-- Ops.cpp.inc
|   |   |-- Ops.h.inc
|   `-- toy
|-- codegen.toy
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
|-- td
|   `-- Ops.td
`-- toy.cpp
```

## MLIR(Multi-Level Intermediate Representation)

[MLIR Language Reference](https://mlir.llvm.org/docs/LangRef/)

[LLVM Language Reference](https://llvm.org/docs/LangRef.html)

MLIR多级中间表示，就是存在多层次的IR表达，通常在不同层次之间转换，逐层递进的进行优化，提供了灵活的中间表示层次结构。

典型应用例如Triton中就是：AST->TTIR->TTGIR->LLVM IR 这样的表达方式。

AST -> TTIR: 层次就是计算的表达式的正确性转换，优化
TTIR -> TTGIR： 硬件绑定层次的IR优化
TTGIR -> LLVM IR: 也是到达最底层了, 最后一步就是生成字节码了。

一般来说越接近上层，优化也就越好做。OK，废话结束，开始正题了。

MLIR 提供了Dialect 扩展能力，支持扩展attributes、operations、types。

以如下 operation 作为参考例子：

```
%t_tensor = "toy.transpose"(%tensor) {inplace = true} : (tensor<2x3xf64>) -> tensor<3x2xf64> loc("example/file/path":12:1)
```

上述语句 可以分解为:

- `"toy.transpose"`: MLIR 中的Opration, Dialect namespace为"toy"，在"toy" dialect中所定义的"transpose" Operation。
- `%t_tensor`: MLIR 中的Value, Operation 可以返回0个或多个结果。
- `(%tensor)`: Oprand，操作数。来自于其他Block Arguments 或者是 其他Operation 的Value。
- `{ inplace = true }`: Attributes，0个或者多个属性构成的Dict，明星属性值总是为常量。
- `(tensor<2x3xf64>) -> tensor<3x2xf64>`: 函数形式的类型，括号内的为Oprand类型，-> 之后的为Value类型。
- `loc("example/file/path":12:1)`: 源码位置

MLIR 经过上述分解后可以被抽象属性为：

- Operation
- Oprand Values
- Attributes
- Types
- Source Location
- Blocks
- Regions


对于未注册的 Attributes、Operation、Types，MLIR会谨慎的对待而不是限制使用。而在进行IR转换的时候，自己对于实现应该审慎。

换句话说，就是MLIR 支持 Attributes、Types 不详时，仍然能够运行，只是结果是否符合预期由不保证。

在本章中，主要实现两个必须要实现功能支持 toy language能够跑起来，产生MLIR。

- Dialect Define
- Operation Define

这里要介绍一下[MLIR TableGen](https://llvm.org/docs/TableGen/ProgRef.html) TODO，在MLIR 中很多代码的生成都是模板代码，对于模板代码的生成不用手写C++，容易出错，还麻烦。

```shell
mlir-tblgen --help
General options:
  Generator to run
      --gen-dialect-decls                               - Generate dialect declarations
      --gen-dialect-defs                                - Generate dialect definitions
      --gen-op-decls                                    - Generate op declarations
      --gen-op-defs                                     - Generate op definitions

```

在CMAKE 如何实现参考CMakeLists.txt 文件，h和cpp都是分别生成的。

```
mlir_tablegen(include/Dialect.h.inc -gen-dialect-decls)
mlir_tablegen(include/Dialect.cpp.inc -gen-dialect-defs)
mlir_tablegen(include/Ops.h.inc -gen-op-decls)
mlir_tablegen(include/Ops.cpp.inc -gen-op-defs)
```

本文继承chapter 01 解析出AST，在上文的基础上将AST 转换为 toy dialect。而在此基础上主要关注如下新增文件。

table gen 生成的h和cpp文件的使用，可以认为是h文件使用到就引入h文件，其他cpp文件使用到就引入cpp文件

```

Dialect.h           # 引入必要的mlir 头文件，和table gen h文件
	Ops.h.inc         # table-gen 产生
	Dialect.h.inc     # table-gen 产生
Dialect.cpp         # 引入table gen cpp文件，以及Operation 一些属性需要补充实现的c++代码
	Ops.cpp.inc       # table-gen 产生
	Dialect.cpp.inc   # table-gen 产生
MLIRGen.h
MLIRGen.cpp         # AST 解析成Toy MLIR Dialect
```


本文不会深入模板生成的文件，详情可以查看 `build/include` 目录下。

## Define a Toy Dialect

最简化实现为

```
def Toy_Dialect : Dialect {
  let name = "toy";
  let cppNamespace = "::mlir::toy";
}
```

在cpp 中使用主要是注意其 "cppNamespace", 通过如下方式加载Dialects, 默认情况下 "MLIRContext" 只加载 [Builtin Dialect](https://mlir.llvm.org/docs/Dialects/Builtin/)。其他Dialect 必须显示的加载进来才能使用。

```c++
mlir::MLIRContext context;
context.getOrLoadDialect<mlir::toy::ToyDialect>();
```


## Define Toy Operations

MLIR 中 新增Operation 继承自 "Op" 类, 而在MLIR中存在一种抽象通用 "Operation" 类。


"Op" 派生类充当 "Operation*" 的智能指针包装器，提供特定于操作的访问器方法和操作的类型安全属性。

如何理解上述说明，通过如下一段代码:


```c++
void processConstantOp(mlir::Operation *operation) {
  ConstantOp op = llvm::dyn_cast<ConstantOp>(operation);

  // This operation is not an instance of `ConstantOp`.
  if (!op)
    return;

  // Get the internal operation instance wrapped by the smart pointer.
  mlir::Operation *internalOperation = op.getOperation();
  assert(internalOperation == operation && "these operation instances are the same");
}

```

上面的 "Operation" 可以被封装 "ConstantOp", "ConstantOp" 通过 "getOperation" 方法获取到原始的 "Operation"。


而在MLIR 中新增Operation实现，依赖于[ODS](https://mlir.llvm.org/docs/DefiningDialects/Operations/)(Operation Definition Specification) TODO框架。

新增所有Op的基类

```
class Toy_Op<string mnemonic, list<Trait> traits = []> :
    Op<Toy_Dialect, mnemonic, traits>;
```

"Toy_Op" 继承来自于 "OpBase.td" 中的类 "Op", 主要参数为: dialect, mnemonic, props.

"Toy_Op" 定义直接绑定了 "Toy_Dialect", 将新增op的名mnemonic， 属性props继续传递。

这样后续新增的任何Op主要继承自 "Toy_Op" 则就绑定在 "Toy_Dialect" 中。

```
class Op<Dialect dialect, string mnemonic, list<Trait> props = []>
```

现在新增一个"ConstantOp" 则为如下定义：

```
def ConstantOp : Toy_Op<"constant", [Pure]> {
  let summary = "constant";
  let description = [{
    Constant operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

    %0 = toy.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64>
  }];

  let arguments = (ins F64ElementsAttr:$value);

  let results = (outs F64Tensor);

  let hasCustomAssemblyFormat = 1;

  let builders = [
    OpBuilder<(ins "DenseElementsAttr":$value), [{
      build($_builder, $_state, value.getType(), value);
    }]>,

    OpBuilder<(ins "double":$value)>
  ];

  let hasVerifier = 1;
}
```

上面定义的属性可以分解为:

- 基础定义： `def ConstantOp : Toy_Op<"constant">`, 继承自 "Toy_Op", 并且新增Op名为 "constant"
- documentation: summary 和 description，写不写无所谓，看你要不要文档
- 入参和输出: arguments 和 results， 该新增op 存在几个输入和几个输出, 以及对应的类型和参数名字。
- OpVerifier: hasVerifier, 如果设置为1，则需要通过c++代码实现verify逻辑。
- 构造方法: builders, builder中的方法说明如何构造ConstantOp类， ODS框架可以生成第一个简单的构造函数，而后续新增的构造函数通过 "double" 入参构造则要用户自己在c++代码中实现逻辑。
- Assembly Format: 就是props， 这里设置上了 `[Pure]`


OK 现在让我们看看C++ 端Dialect.cpp需要新增什么样的代码。

ConstantOp 主要实现了 build、parse、print、verify 方法

其中build 为在 builders 新增的 构造实现方法。

parse、print 为序列化输入和输出方法，用来从文本中恢复Op和序列化Op成文本的方法。

verify 方法 hasVerifier = 1时候需要补充，校验Op是否 Valid。

## AST -> Toy Dialect

AST 转换为 Toy Dialect主要逻辑在 MLIRGen.cpp 文件中。

Toy Dialect 顶层容器可以认为是 "mlir::ModuleOp",

总的拉说 AST -> Toy Dialect 分为3个步骤 对Function的解析:

- 函数原型 proto: 函数名，入参mlirGen
- 函数体 body： 函数表达式mlirGen
- 函数返回 return: 返回值mlirGen

函数body 解析 expr 以DFS的方式去解析，这里只有一点函数body里面都是 VarDecl(除了Print)，里面的各种细节的Call、Literal都是被VarDecl所持有的。

总结，这里基本的toy language能够解析出来，但是少了一些东西，就是Type Infer。每个操作如果要保持准确性，就必须知道 datatype、shape信息。 这些信息将在后续章节中被补全。

## Reference

### ScopedHashTable 和 ScopedHashTableScope 的用法介绍

ScopedHashTableScope 为 ScopedHashTable 辅助类，当在 ScopedHashTableScope 作用域内添加到ScopedHashTable 的值，会在离开作用域之后被清除。

参考例子如下

```
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/raw_ostream.h"

void test_ScopedHashTable()
{
    llvm::ScopedHashTable<llvm::StringRef, int> symbolTable;
    {
        llvm::ScopedHashTableScope<llvm::StringRef, int> scope(symbolTable);
        symbolTable.insert("var1", 1);
        symbolTable.insert("var2", 2);

        if (auto value = symbolTable.lookup("var1"))
            llvm::outs() << "var1: " << value << "\n";
    }

    if (!symbolTable.lookup("var1"))
        llvm::outs() << "var1 not found outside scope.\n";
}
```