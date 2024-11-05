# cmake 最小版本要求
cmake_minimum_required(VERSION 3.13.4)

# 定义项目名称
project(toy_chapter_01)

# 查找并引入 LLVM 和 MLIR 的配置
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

# 将 LLVM 和 MLIR 的 CMake 工具导入到当前作用域
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")

# 使用 LLVM 和 MLIR 所需的工具
include(TableGen) # 用于生成 MLIR 所需的表格生成代码（如果使用 TableGen）
include(AddLLVM) # 添加 LLVM 项目相关工具
include(AddMLIR) # 添加 MLIR 项目相关工具

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17 CACHE STRING "")


# 设置包含头文件的目录
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include)

# 链接 LLVM 和 MLIR
link_directories(${LLVM_LIBRARY_DIRS})

# 源文件
set(SOURCE_FILES
    toy.cpp
    parser/AST.cpp
)
# 生成 toy 可执行文件，并添加依赖项
add_executable(toy ${SOURCE_FILES})

# 链接 LLVM 和 MLIR 库
target_link_libraries(toy
    PRIVATE
    MLIRSupport # 支持库
    MLIRIR # IR 相关库
    MLIRParser # IR 解析库
    MLIRPass # Pass 相关库
    MLIRTransforms # 变换库
    MLIRDialect # 具体语言相关的方言库（如果有方言）
)

# 确保 CMake 生成文件后输出成功消息
message(STATUS "Toy Compiler project configured successfully.")
