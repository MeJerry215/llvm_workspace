# LLVM

tag: llvmorg-19.1.1

```shell
cmake -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DLLVM_ENABLE_ASSERTIONS=ON -DLLVM_INCLUDE_TOOLS=ON -DLLVM_INCLUDE_UTILS=ON -DLLVM_BUILD_EXAMPLES=ON -DLLVM_BUILD_TESTS=ON ../llvm -DLLVM_ENABLE_PROJECTS="mlir;llvm;clang" -DLLVM_TARGETS_TO_BUILD="host;NVPTX"

cmake -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DLLVM_ENABLE_ASSERTIONS=ON -DLLVM_INCLUDE_TOOLS=ON -DLLVM_INCLUDE_UTILS=ON -DLLVM_BUILD_EXAMPLES=ON -DLLVM_BUILD_TESTS=ON -DLLVM_BUILD_DOCS=ON -DLLVM_ENABLE_DOXYGEN=ON ../llvm -DLLVM_ENABLE_PROJECTS="mlir;llvm;clang" -DLLVM_TARGETS_TO_BUILD="host;NVPTX"
$ ninja
```

ssh lei.zhang1@lei.zhang@10.113.1.63@10.101.1.125 -p 2222

/home/lei.zhang/src_code/triton_opt/tune_kernels/token_attention/tune.py
/home/lei.zhang/triton_opt/llvm-project/llvm_workspace/README.md 10.113.1.15


怎样去学习mlir这套框架？
https://www.zhihu.com/question/435109274/answer/3585914452


mlir
https://mlir.llvm.org/talks/


https://llvm.org/devmtg/2020-09/slides/MLIR_Tutorial.pdf

