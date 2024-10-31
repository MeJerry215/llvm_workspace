# cmake 最小版本要求
cmake_minimum_required(VERSION 3.13.4)

cmake_policy(SET CMP0116 NEW)

# 定义项目名称
project(toy_chapter_02)

# 查找并引入 LLVM 和 MLIR 的配置
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

# 将 LLVM 和 MLIR 的 CMake 工具导入到当前作用域
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17 CACHE STRING "")

# 设置包含头文件的目录
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include)

# 链接 LLVM 和 MLIR
link_directories(${LLVM_LIBRARY_DIRS})
link_directories(${MLIR_LIBRARY_DIRS})

# 源文件
set(SOURCE_FILES
    toy.cpp
    parser/AST.cpp
    mlir/Dialect.cpp
    mlir/MLIRGen.cpp
    mlir/LowerToAffineLoops.cpp
    pass/ToyCombine.cpp
    pass/ShapeInferencePass.cpp
)

# 生成 toy 可执行文件，并添加依赖项
add_executable(toy ${SOURCE_FILES})

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

# 链接 LLVM 和 MLIR 库
target_link_libraries(toy
    PRIVATE
    ${dialect_libs}
    ${extension_libs}
    MLIRSupport
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRTransforms
    MLIRDialect
    MLIRCastInterfaces
    MLIRCallInterfaces
    MLIRFunctionInterfaces
    MLIRSideEffectInterfaces
)

# 添加 td 目录的 CMakeLists.txt
add_subdirectory(td)
add_subdirectory(pass)

# 确保 CMake 生成文件后输出成功消息
message(STATUS "Toy Compiler project configured successfully.")
