# PTOAS (PTO Assembler & Optimizer)

## 未发布

### PTO IR 接口调整

- `pto.tfillpad_inplace` 和 `pto.tfillpad_expand` 已整合到
  `pto.tfillpad`；通过 `mode` 选择行为，默认值为 `normal`。
- `pto.textract_fp`、`pto.tinsert_fp`、`pto.tmov.fp` 和
  `pto.tstore_fp` 已移除。请改用 `pto.textract`、`pto.tinsert`、
  `pto.tmov` 和 `pto.tstore` 的可选 `fp` operand。旧文本 PTO IR
  不会自动迁移，需要在升级时替换 op 名称并按新语法传入 `fp`。
- PTO-BC v0 保留上述 FP op 的历史 opcode 作为 wire 兼容别名；已有
  `.ptobc` 文件可由新版 decoder 读取并转换为统一后的 PTO IR。

## 版本
- 版本号：v0.51
- 发布日期：2026-02-14

## 变更摘要
- PTOAS 首次发布

## 概述
PTOAS（PTO Assembler & Optimizer）是面向 PTO Bytecode 的编译器工具链，基于 LLVM/MLIR LLVM21 VPTO 分支 `vpto-dev/llvm-project:feature-vpto-llvm21` 构建。它提供 PTO Dialect 的定义、解析、验证、优化与代码生成能力，并输出可调用 `pto-isa` 的 C++ 代码。

PTOAS很快将集成到以下框架中，敬请期待
- PyPTO
- TileLang

## 本仓库的目标用户
PTOAS 主要面向：
- 编译器与框架后端开发者
- 高性能算子/内核开发者
- 需要进行 PTO Bytecode 生成、调试与落地的工程团队

## 主要能力
- PTO Dialect 全流程（定义、解析、验证、打印）
- 与 Tile 抽象/地址空间/同步模型配套的 IR 支撑
- PTO Bytecode → C++ 生成
- Python 端的 Dialect 构建与测试样例

## 平台与依赖最低配置
- **操作系统**：macOS (Darwin) 或 Linux (Ubuntu 20.04+)
- **编译器**：Clang >= 12 或 GCC >= 9（支持 C++17）
- **构建工具**：CMake >= 3.20，Ninja
- **Python**：Python 3.8+

## 如何使用PTOAS以及PTO IR的详细描述
- 构建与环境配置：`README.md`
- PTO Bytecode 定义：`docs/PTO_IR_manual.md`
