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

# 开启更多的编译警告
# include(CheckCXXCompilerFlag)
# check_cxx_compiler_flag("-Werror" HAS_WERROR_FLAG)

# if(HAS_WERROR_FLAG)
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
# endif()

# 设置包含头文件的目录
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include)

# 链接所需的库
link_directories(${LLVM_LIBRARY_DIRS})

# 添加子目录 (如有)
# add_subdirectory(lib)
# add_subdirectory(tools)

# 定义目标：一个简单的可执行文件，包含 main.cpp
add_executable(toy toy.cpp parser/AST.cpp)

target_link_libraries(toy
    PRIVATE
    MLIRSupport # 支持库
    MLIRIR # IR 相关库
    MLIRParser # IR 解析库
    MLIRPass # Pass 相关库
    MLIRTransforms # 变换库
    MLIRDialect # 具体语言相关的方言库（如果有方言）
)

# table gen
# mlir_tablegen(MLIRExample.td MLIRExampleGen)

# table gen
# add_public_tablegen_target(MLIRExampleGen)