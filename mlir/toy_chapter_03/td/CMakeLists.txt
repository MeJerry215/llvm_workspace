include(TableGen) # 用于生成 MLIR 所需的表格生成代码（如果使用 TableGen）
include(AddLLVM) # 添加 LLVM 项目相关工具
include(AddMLIR) # 添加 MLIR 项目相关工具

# 设置 Ops.td 文件路径
set(LLVM_TARGET_DEFINITIONS ${CMAKE_CURRENT_SOURCE_DIR}/Ops.td)

# 使用 mlir_tablegen 生成 .inc 文件
mlir_tablegen(include/Ops.h.inc -gen-op-decls)
mlir_tablegen(include/Ops.cpp.inc -gen-op-defs)
mlir_tablegen(include/Dialect.h.inc -gen-dialect-decls)
mlir_tablegen(include/Dialect.cpp.inc -gen-dialect-defs)

# 添加生成目标
add_public_tablegen_target(GenerateOps)

# 确保 toy 依赖于生成的 Ops 和 Dialect 文件
add_dependencies(toy GenerateOps)

# 添加 include 目录以包含生成的 .inc 文件
target_include_directories(toy PRIVATE include ${CMAKE_CURRENT_BINARY_DIR}/include)