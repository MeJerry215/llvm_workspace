# Chapter 2: Emit Basic MLIR

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