cat <<'EOF' > README.md
# PTOAS (PTO Assembler & Optimizer)

## 项目简介 (Introduction)

**PTOAS** (`ptoas`) 是一个基于 **LLVM/MLIR** 框架构建的专用编译器工具链，专为 **PTO IR** (Programming Tiling Operator IR) 设计。

作为连接上层 AI 框架与底层华为昇腾（Ascend）硬件的关键组件，`ptoas` 提供了完整的 C++ 与 Python 接口，主要职责包括：

1.  **IR 解析与验证**：解析 `.mlir` 输入文件，验证 PTO Dialect 操作（Ops）的语义正确性。
2.  **编译优化 (Passes)**：执行针对达芬奇架构（Da Vinci Architecture）的特定优化 Pass，如算子融合、Tile 自动切分策略等。
3.  **代码生成 (Lowering)**：支持将 PTO IR 下降（Lowering）到 `EmitC` Dialect，最终生成可调用 `pto-isa` C++ 库的代码。
4.  **Python 绑定 (Python Bindings)**：提供名为 `pto` 的 Python 模块，支持与 **PyPTO**、**TileLang**、**CuTile** 等上层 Python 语言/框架的无缝对接，允许用户在 Python 端直接构建、操作和编译 PTO IR。

## 目录结构 (Directory Structure)

```text
pto-project/
├── include/
│   └── pto/            # PTO Dialect 的头文件与 TableGen 定义 (.td)
├── lib/
│   └── pto/            # Dialect 核心实现、Pass 逻辑与 C++ 源码
├── python/             # [新增] Python Binding 源码与模块定义
│   └── pto/
├── test/
│   └── pto/            # 基于 lit 和 FileCheck 的回归测试用例
├── tools/
│   └── ptoas/          # ptoas 命令行工具入口
└── CMakeLists.txt      # 顶级构建配置
```

## 构建指南 (Build Instructions)
本项目采用**Out-of-Tree**方式构建，依赖外部已编译好的 LLVM 和 MLIR 库。

### 前置依赖 (Prerequisites)
 - C++ 编译器: 支持 C++17 标准 (GCC >= 7.5 或 Clang).
 - CMake: >= 3.20.
 - Ninja: 推荐使用的构建系统.
 - Python: 3.6+ (如果需要构建 Python 绑定).
 - LLVM/MLIR: 需预先编译并安装 (建议 LLVM 16+).

### 编译步骤 (Compiling)
假设您的 LLVM/MLIR 安装/构建路径位于 /path/to/llvm-project/build。

1. 创建构建目录
```Bash
mkdir build && cd build
```

2. 配置 CMake (启用 Python 绑定) 要启用 Python 支持，请务必添加 -DMLIR_ENABLE_BINDINGS_PYTHON=ON。 请根据您的实际环境修改 MLIR_DIR 和 LLVM_DIR。

```Bash
cmake -G Ninja .. \
    -DMLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir \
    -DLLVM_DIR=/path/to/llvm-project/build/lib/cmake/llvm \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DPython3_EXECUTABLE=$(which python3)
```

3. 执行编译 编译核心工具 ptoas 和 Python 模块：

```Bash
ninja
```

4. 环境配置 (Python Path) 编译完成后，需要将生成的 Python 包路径添加到环境变量中，以便 PyPTO/TileLang 调用：

```Bash
export PYTHONPATH=$PWD/tools/pto/python_packages/pto_core:$PYTHONPATH
```

## 使用方法 (Usage)

### 1. 命令行工具 (CLI)
ptoas 可直接处理 MLIR 文本文件：

```Bash

# 解析并打印 PTO IR
./bin/ptoas input.mlir

# 运行 Tile 优化 Pass
./bin/ptoas --pto-tile-optimization input.mlir
```

### 2. Python 接口 (For PyPTO/TileLang)
上层框架可以通过 pto 模块直接构建 IR。

#### 示例：在 PyPTO 中使用 PTO Dialect

```Python
import pto
from pto.ir import Context, Module, Location
from pto.dialects import pto as pto_dialect

with Context() as ctx, Location.unknown():
    pto_dialect.register_dialect(ctx)
    module = Module.create()
    
    with module.body:
        # 构建 PTO 算子
        arg0 = ...
        arg1 = ...
        op = pto_dialect.AddOp(arg0, arg1)
        
    print(module)
```

## 贡献 (Contributing)
 - 添加新算子: 在 include/pto/PTOOps.td 中定义。
 - 扩展 Python 接口: 在 python/pto/pto_ops.td 或 python/pto/PTOModule.cpp 中添加绑定逻辑。
