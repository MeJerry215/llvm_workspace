# Getting Started with LLVM System

`LLVM` project包含多个组件，而核心组件成为`LLVM`。

C-like languages using [Clang](https://clang.llvm.org/)。

本文开头了如何构建llvm prerequisites、build toolchain、environment、compile。

然后为llvm 的项目目录结构、llvm 编译相关工具的使用、以及FAQ等。

## 目录结构

子项目llvm 和其他的子项目拥有类似的目录结构，可以大致分为

``` tree
tree -L 1 llvm

llvm
|-- CMakeLists.txt
|-- cmake
|-- examples
|-- include
|-- lib
|-- bindings
|-- projects
|-- test
|-- tools
`-- utils

```

**`llvm/cmake`**

- `llvm/cmake/modules`: llvm用户定义选项build配置。
- `llvm/cmake/platforms`: Android NDK、IOS 等其他系统toolchain 配置文件

**`llvm/exmaples`**

llvm 官方sample 例子

**`llvm/include`**

- `llvm/include/llvm`: llvm 头文件，各个头子目录下是不同的组件
- `llvm/include/llvm/Support`: llvm 通用支持库，非必要。包含c++ stl、command line 处理库
- `llvm/include/llvm/Config`: 由cmake 配置处理生效头文件。

**`llvm/lib`**

- `llvm/lib/IR`: LLVM 核心类实现，实现如Instruction、BasicBlock等
- `llvm/lib/AsmParser`: LLVM 汇编语言解析库
- `llvm/lib/Bitcode`: LLVM 读写字节码库
- `llvm/lib/Analysis`: LLVM 各种程序分析库，支持如Call Graph、 Induction variables、Natural Loop Identification等
- `llvm/lib/Transforms`: LLVM IR 转换库，支持如 Dead Code Elimination、Sparse Conditional Constant Propagation、 Inline、 Loop Invariant Code Motion等变换
- `llvm/lib/Target`: LLVM 描述Target架构库
- `llvm/lib/CodeGen`: LLVM 代码生成库，如Instruction Selector、Instruction Scheduling 以及 Register Allocation
- `llvm/lib/MC`: LLVM Machine Code库，主要用来处理汇编和object文件
- `llvm/lib/ExecutionEngine`: LLVM 解释执行和JIT 场景下的runtime 库
- `llvm/lib/Support`: 对应头文件`llvm/include/ADT` 和 `llvm/include/Support/`下的源码实现。

**`llvm/bindings`**

非c/c++的其他语言实现bindings

**`llvm/projects`**

非严格意义上 llvm 的工程，用来给你自定义实现的。

**`llvm/test`**

llvm 测试项代码

**`llvm/tools`**

llvm 编译相关的各种工具

- `bugpoint`: llvm debug 工具，用来确定pass导致core dump或者编译错误，用法可以参考。
- `llvm-ar`: llvm 创建bitcode archive的工具
- `llvm-as`: 汇编码转机器码工具 `.ll -> .bc`
- `llvm-dis`: 机器码转汇编码工具 `.bc -> .ll`
- `llvm-link`: llvm 链接工具
- `lli`: llvm 解释器，可以直接执行llvm 机器码
- `llc`: llvm 后端编译工具 `.bc/.ll -> .s/.o`
- `opt`: llvm 优化工具，`.bc -> .bc`

**`llvm/utils`**

- `codegen-diff`: 比较llc/lli 生成的code， 用法参考`perldoc codegen-dif`
- `TableGen/`: llvm TableGen tools 相关工具

## LLVM Tool Chain Examples

```c
#include <stdio.h>

int main() {
  printf("hello world\n");
  return 0;
}
```


```shell
# 1. 编译可执行二进制文件
clang hello.c -o hello
# 生成object 文件
clang -c hello.c -o hello.o

# 2. 生成llvm bitcode 代码文件
clang -c hello.c -emit-llvm  -o hello.bc

# 3. JIT 运行
lli hello.bc

# 4. 反汇编查看llvm ir
llvm-dis < hello.bc | less
# 也可以通过 clang 得到
clang -S hello.c -emit-llvm  -o hello.ll

# 5. bitcode to assembly
llc hello.bc -o hello.s
# 也可以通过clang得到
clang -S hello.c -o hello.s

```


/home/lei.zhang/triton_opt/llvm-project/llvm_workspace/llvm/00_Getting_Started_with_the_LLVM_System.md