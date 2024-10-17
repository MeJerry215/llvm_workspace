# MLIR Toy Language

基于llvm project tag llvmorg-19.1.1

编译命令在llvm-project 创建build 目录后执行

```shell

mkdir build

cd build

cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON -DLLVM_INCLUDE_TOOLS=ON -DLLVM_INCLUDE_UTILS=ON -DLLVM_BUILD_EXAMPLES=ON -DLLVM_BUILD_TESTS=ON -DLLVM_ENABLE_RTTI=ON ../llvm -DLLVM_ENABLE_PROJECTS="mlir;llvm;clang" -DLLVM_TARGETS_TO_BUILD="host;NVPTX"

ninja -j32

ninja install
```

## [Chapter 1: Toy Language and AST](toy_chapter_01/README.md)

```shell
cd toy_chapter_01
mkdir build
cmake .. -G ninja
ninja

./toy ../ast.toy -emit=ast

```




## Reference

https://llvm.org/doxygen/classllvm_1_1StringRef.html

### LLVM Class Reference

