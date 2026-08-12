# PTOAS (PTO Assembler & Optimizer)

## 未发布

- 为 A5 `pto.tquant.mx` 增加 `grpAxis`（默认 `axis1`）和 `interleave`（默认关闭），按分组轴校验 canonical/legacy-flat shape，并统一生成 grouped `TQUANT<grp_axis, MxQuantAlg[, interleave]>`。
- 增加 axis0/DN 与 axis1/ND 的 MXFP8/MXFP4 物理 stride、packed destination、compact-prefix 和 full-VL scaling capacity 校验；axis1 canonical B16、flat B16 NV、DN FP16 MXFP4 非对齐 stride、DN FP32 interleave 和不完整 B16 padding VL 等当前 pin 不安全组合明确拒绝。
- 扩展 `pto.tmov` 的非 scaling 第三 tile 为 exponent X-to-ZZ 形态，保留 `$fp`、Python `fp=` 和 C++ `getFp()` API；ND/DN 分别生成 `TMOV(...)` / `TMOV<0>(...)`，并准确声明 source mutation 与临时区内存效应。scaling FP 和普通 TMOV 行为保持兼容。
- MX TQUANT 的 f16/bf16 padded source 按原地清零声明 `Read + Write`；TQuantMx 与 X-to-ZZ 在 legacy/modern PlanMemory 中执行 byte-range no-alias，防止 source 与任一输出或目标重叠。
- PTO-BC 保持 scaling-FP legacy wire；TQuantMx 与 X-to-ZZ 通过显式 generic v0 compatibility record 编解码，不新增专用 opcode。`exp_zz + storeMode` fused 形态保留但标记 deprecated，并限制为 `axis1`。
- 既有 tight MXFP8 legacy-flat case 迁移 scaling physical capacity（128B -> 256B，展开路径至少 512B）；既有 BF16 MXFP4 packed destination physical 列和 scaling capacity 按新物理契约迁移。
- 新增 MX axis/Exponent X-to-ZZ 的 lit、MemoryEffects、PlanMemory no-alias、PTO-BC、Python binding、样例和 A5 TileLang ST 覆盖。

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
