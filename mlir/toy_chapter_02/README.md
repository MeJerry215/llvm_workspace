# [Chapter 2: Emit Basic MLIR](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-2/)

在完成Chapter 1生成了AST，而这次目标是将AST转换为MLIR Dialect，以方便后续的优化。

目录结构

```
.
|-- CMakeLists.txt
|-- README.md
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
