include(TableGen) # 用于生成 MLIR 所需的表格生成代码（如果使用 TableGen）
include(AddLLVM) # 添加 LLVM 项目相关工具
include(AddMLIR) # 添加 MLIR 项目相关工具

include_directories(${CMAKE_SOURCE_DIR}/td)

# 设置 ToyCombine.td 文件路径
set(LLVM_TARGET_DEFINITIONS ${CMAKE_CURRENT_SOURCE_DIR}/ToyCombine.td)

# 使用 mlir_tablegen 生成 .inc 文件
mlir_tablegen(include/ToyCombine.inc -gen-rewriters)

# 添加生成目标
add_public_tablegen_target(CombineIncGen)

# 确保 toy 依赖于生成的 Ops 和 Dialect 文件
add_dependencies(toy CombineIncGen)

# 设置 ShapeInferenceInterface.td 文件路径
set(LLVM_TARGET_DEFINITIONS ${CMAKE_CURRENT_SOURCE_DIR}/ShapeInferenceInterface.td)

# 使用 mlir_tablegen 生成 .inc 文件
mlir_tablegen(include/ShapeInferenceOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(include/ShapeInferenceOpInterfaces.cpp.inc -gen-op-interface-defs)

# 添加生成目标
add_public_tablegen_target(ShapeInferenceInterfaceIncGen)

# 确保 toy 依赖于生成的 Ops 和 Dialect 文件
add_dependencies(toy ShapeInferenceInterfaceIncGen)

# 添加 include 目录以包含生成的 .inc 文件
target_include_directories(toy PRIVATE include ${CMAKE_CURRENT_BINARY_DIR}/include)


