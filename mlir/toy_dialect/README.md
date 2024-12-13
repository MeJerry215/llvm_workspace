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
mkdir build && cd build
cmake -G Ninja ..
ninja

./toy ../cases/ast.toy -emit=ast

```

## [Chapter 2: Emit Basic MLIR](toy_chapter_02/README.md)

```shell
cd toy_chapter_02
mkdir build && cd build
cmake -G Ninja ..
ninja

./toy ../cases/codegen.toy -emit=mlir -mlir-print-debuginfo 2>codegen.mlir

```

## [Chapter 3: High-level Language-Specific Analysis and Transformation](toy_chapter_03/README.md)

```shell
cd toy_chapter_03
mkdir build && cd build
cmake -G Ninja ..
ninja

./toy ../cases/trivial_reshape.toy -emit=mlir -opt
./toy ../cases/transpose_transpose.toy -emit=mlir -opt
```

## [Chapter 4: Enabling Generic Transformation with Interfaces](toy_chapter_04/README.md)

```shell
cd toy_chapter_04
mkdir build && cd build
cmake -G Ninja ..
ninja

./toy ../cases/codegen.toy -emit=mlir -opt

```

## [Chapter 5: Partial Lowering to Lower-Level Dialects for Optimization](toy_chapter_05/README.md)


```shell
cd toy_chapter_05
mkdir build && cd build
cmake -G Ninja ..
ninja

./toy ../cases/affine-lowering.mlir -emit=mlir-affine
```


## [Chapter 6: Lowering to LLVM and CodeGeneration](toy_chapter_06/README.md)

```shell
cd toy_chapter_06
mkdir build && cd build
cmake -G Ninja ..
ninja

echo 'def main() { print([[1, 2], [3, 4]]); }' | ./toy -emit=ast
echo 'def main() { print([[1, 2], [3, 4]]); }' | ./toy -emit=mlir-affine
echo 'def main() { print([[1, 2], [3, 4]]); }' | ./toy -emit=mlir-llvm
echo 'def main() { print([[1, 2], [3, 4]]); }' | ./toy -emit=llvm
echo 'def main() { print([[1, 2], [3, 4]]); }' | ./toy -emit=jit


./toy ../cases/affine-lowering.mlir -emit=ast
./toy ../cases/affine-lowering.mlir -emit=mlir-affine
./toy ../cases/affine-lowering.mlir -emit=mlir-llvm
./toy ../cases/affine-lowering.mlir -emit=jit

# 可以使用 --mlir-print-ir-after-all 每个pass 之后的IR 变化
./toy ../cases/affine-lowering.mlir -emit=jit --mlir-print-ir-after-all
```


## [Chapter 7: Adding a Composite Type to Toy](toy_chapter_07/README.md)

```shell
cd toy_chapter_07
mkdir build && cd build
cmake -G Ninja ..
ninja

./toy ../cases/struct-codegen.toy -emit=jit

```


## Reference

https://llvm.org/doxygen/classllvm_1_1StringRef.html

### LLVM Class Reference

