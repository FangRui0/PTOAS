<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# A5 MX TQUANT `grp_axis` 与 exponent X-to-ZZ TMOV 设计

## 1. 状态与关联事项

- 状态：设计提案，待评审。本 PR 当前只含设计文档；实现基线直接对齐两个 pto-isa
  远端当前已发布的最新分支头，不等待额外的 pto-isa 修复 revision。
- 关联 issue：[#1185](https://github.com/hw-native-sys/PTOAS/issues/1185)
- 设计复核基线：
  - PTOAS `988d50e24`
  - PTOAS pin 盘点基线 `main@f8912bc781f57b66b9d41007ec730507a8c40096`（rev11）
  - pto-isa 本地快照 `7af803bc4056af8b39a55751ac2f4b75cdb47fbd`（下称"快照"）
  - pto-isa 接口祖先 `f03c2454e4211bdfc5fe9d3e859bc9239443514c`（GitHub 上已验证）
  - pto-isa GitCode `master` `69a81f3b2d145fe4f9925cfd65a083f78ad1f804`（rev11 pin）
  - pto-isa GitHub `main` `40e741bf1cfce99da3b1caa514e08c2f72894922`（rev11 pin；merge PR #239，
    commit message 明确同步 GitCode `master@69a81f3b2d14`）
- 目标接口：`pto.tquant.mx`、`pto.tmov` 的 exponent 布局转换形态

### 1.1 修订记录

**rev11（本次）**：把 pin 策略收敛为“直接对齐最新 pto-isa”。不再要求先向 pto-isa
合入本设计派生的隐藏 scratch / padding-tail 修复，也不再设计未来占位 revision。

| 事项 | rev11 结论 | 位置 |
|---|---|---|
| 当前 pin | GitCode 三处是 `27386d9`，GitHub CPU-sim 是 `e948507`，`Dockerfile.dev` 是 `662d7f2`；rev4/5 记录的 `ce3262/a8168` 已过时 | §13-2 |
| latest 目标 | GitCode 目标为 `master@69a81f3`；GitHub PR #239 已合并，GitHub 目标为 `main@40e741b`。二者是同一轮同步在不同 remote 的实际落点 | §10、§12 提交 0、§13-2 |
| 实现边界 | latest 的 `TQuant.hpp` 仍有 §3.1.1 所列隐藏写；它们不再参与 pin 选 SHA，而是在 PTOAS verifier 中形成明确的不支持/拒绝边界，不能继续列作正向验收 case | §7.1.1、§11.3、§13-10 |
| updater 现状 | `.github/scripts/update_pto_isa_pin.py` 已在 `e488f9e3d` 删除；本 PR 不是“扩展现有 updater”，而是重新引入 repo-aware pin manifest/updater，或以等价的可审计 target map 落地 | §1.2、§10、§12、§13-2 |

rev8–rev10 中“必须先产出 pto-isa 安全 revision 才能 pin/实现”的结论由 rev11 替换；
这些修订记录保留为评审历史，不再是当前实施门槛。

**rev10**：把 pin 验证从会被 PTOAS 256B slot 对齐掩盖的普通 canary
改为 PTO-ISA 原生精确 redzone harness，修正 PTO-BC generic 分派、DN `tmp` 静态契约、
TQuant-only 用例边界与提交顺序。

| 事项 | rev10 结论 | 位置 |
|---|---|---|
| pin redzone | 当时要求提交 0 以原生 C++ harness 准入候选 pin；rev11 保留精确 redzone 方法作为 latest 行为审计，但不再作为 pin gate | §11.6.1、§13-10 |
| `pto.tquant.mx` PTO-BC | PTOAS `988d50e24` 的 v0 known-op 表只有 `pto.tquant`，没有 `pto.tquant.mx`；给 `TQuantMxOp` 增加显式 generic compatibility shim，无需 `PTOBC_ALLOW_GENERIC`，并解析二进制断言 opcode 为 `kOpcodeGeneric` | §10、§11.7、§12 |
| DN `tmp` dynamic | ISA 不访问不等于 PTOAS 不分配/不类型化；DN `tmp` physical shape 同样必须静态，PlanMemory 才能计算 local slot，EmitC 才不会生成含 `-1` 的 physical template 维 | §7.2、§11.4、§13-8 |
| `n=1` case 分层 | 当时只用于 TQUANT pin/mirror；rev11 进一步降为原生行为诊断，不进入 latest 支持范围内的 PTOAS 正向 case。其 exponent `1x1` / `2x1` 均不能进入 DN-to-ZZ 完整链路 | §11.5、§11.6、§11.6.1 |
| 提交 0 可执行性 | 原则仍是提交 0 不依赖 PTOAS 新属性/lowering；rev11 将动作改为 exact latest SHA 的 remote/接口/既有构建验证，不再等待 harness 全过 | §11.6.1、§12 |

**rev9**：补齐 DN FP32 interleave 与 B16 VL-tail 两处 pin blocker，并修正
canonical B16 测试 dtype 与既有 MXFP4 lit 迁移范围。

| 事项 | rev9 结论 | 位置 |
|---|---|---|
| DN interleave FP32 `dst` scratch | 当前实现按 `align32(srcPhysicalCols)` 把线性 exponent 暂存到 `dst`；rev9 要求最终 pin 修复，rev11 改为 latest 下由 verifier 拒绝该组合 | §3.1.1、§7.1.1、§11.3、§13-10 |
| B16 source VL-tail | 不能要求 IR 为实现越界伪造额外 source capacity；rev9 要求上游修复，rev11 改为 latest 下拒绝不完整 VL 组合 | §3.1.1、§7.1.1、§11.3、§13-10 |
| canonical B16 test dtype | max/scaling 的存储 dtype 跟随 source（f32/f16/bf16），读取后再提升到 f32 比较；MXFP4 dst 按 packed byte 位精确比较 | §11.5 |
| 既有 MXFP4 lit 迁移 | `tquant_mx_a5_emitc.pto` 的 packed dst physical cols 从 32 收紧为 16；ReleaseNotes 明示。兼容承诺只覆盖两个 tight MXFP8 legacy-flat case | §11.1、§12、§13-11、§14 |

**rev8**：补齐 TQuantMx 的 source-derived destination 契约，并把 PTO-ISA
隐藏 scratch 修复提升为 pin 的硬前置条件。

| 事项 | rev8 结论 | 位置 |
|---|---|---|
| axis1 flat + padded source | flat exp 分支额外要求 `srcPhysicalCols == srcValidCols`；padded source 只能走 canonical 2D exp | §3.1.1、§7.1.1、§11.3 |
| `dst` stride / capacity | MXFP8 axis1 的 destination stride 等于 source physical cols；MXFP4 的 packed destination stride 等于 source physical cols / 2；DN MXFP4 同样按 packed stride 约束 | §7.1.1、§11.1/11.3 |
| 隐藏 scratch / `max` 语义 | 不把越界 scratch 或覆盖 `max` 定义成合法语义；rev8 选择 pin-first 修复，rev11 改为 latest 下 verifier 拒绝对应组合 | §3.1.1、§7.1.1、§11.3、§13-10 |
| PR 元数据 | PR body 与 rev8、ready-for-review 状态、复用 `pto.tmov`、动态/API/PTO-BC 结论同步 | PR #1197 |

**rev7**：关闭 rev6 评审提出的 3 个 P1，并保留 `TMovOp` 的现有公开 API。

| 事项 | rev7 结论 | 位置 |
|---|---|---|
| `TQuantMxOp` physical 契约 | 按 axis0/axis1 和 exp flat/2D 分支约束 stride、compact prefix 与 capacity；参与证明的 valid/physical shape 动态时拒绝 | §7.1、§11.1/11.3 |
| `TQuantMxOp` source effects | f16/bf16 且 `validCols < physicalCols` 时 `src` 为 `Read + Write`，其余为 `Read`；补 padding 后再次读取 source 的内存规划/同步回归 | §9、§11.2 |
| `TMovOp` API 兼容 | ODS operand 名继续为 `$fp`，保留 Python `fp=`、C++ `getFp()` 与 builder；helper 只在内部把该值分类为 FP 或 X-to-ZZ tmp | §5.3、§6.2、§10 |
| PTO-BC 分派 | scaling `$fp` 保持 #1122 的 legacy FP wire 编码；非-scaling `$fp` 走 generic v0 兼容记录，不能仅因 `getFp()` 非空而选择 FP wire opcode | §10、§11.7 |

**rev6**：不再新增独立 X-to-ZZ op，改为复用并扩展现有 `pto.tmov`。

| 事项 | rev6 结论 | 位置 |
|---|---|---|
| X-to-ZZ 的 PTO IR 表达 | 复用 `pto.tmov` 现有第三个 tile operand 槽位；`loc=scaling` 表示既有 `fp`，非 scaling 表示 X-to-ZZ `tmp` | §1.2、§5.3、§6.2 |
| 分派依据 | 地址空间只负责区分 overload 家族；非-scaling operand 还必须通过 X-to-ZZ 的 dtype/layout/shape/stride/capacity verifier | §7.2 |
| 兼容与编码 | 不新增 op、opcode 或独立 binding；旧二参数/FP/preQuantScalar 形态不变，`grpAxis` 通过现有 `pto.tmov` 的属性字典编码。rev6 当时提出的 `$fp`→`$aux` ODS 改名已由 rev7 撤销 | §8.2、§10、§11.7 |

**rev5**：将 3 项原待决事项固化为确定决策。当时选择独立 X-to-ZZ op，
该 op 拆分结论已由 rev6 替换，其余三项继续有效。

| 事项 | rev5 结论 | 位置 |
|---|---|---|
| ND 存量扁平 shape | axis1 永久同时接受 canonical `[M, N/32]` 和 legacy flat `[1, M*N/32]`，不迁移现有 IR | §1.2、§7.1、§13-3 |
| `pto.tquant.mx` EmitC 文本 | 所有不带 `exp_zz` 的 grouped 形态统一生成 `<grp_axis, MxQuantAlg[, interleave]>`；不为默认 ND 保留旧类型列表文本 | §1.2、§8.1 |
| pin 三元组 ownership | PR #1122 已合入且仍是单 repo/SHA updater；本 PR 负责落地 repo-aware 三元组与本特性 pin bump，不再声明依赖 #1122 | §1.2、§12 提交 0、§13-2 |

**rev4**：按第三轮评审关闭 4 个 P1。

| 编号 | rev3 的问题 | rev4 的结论 | 位置 |
|---|---|---|---|
| P1-A | 断言 GitCode/GitHub "SHA 空间不通"，并要求先搜索复杂的三元组共同后继 | `f03c2454` 在 GitHub 上确实存在，且相对 `a8168c6` ahead 8 / behind 0，已包含 CPU-Sim duplicate-stub 修复。每个目标仍须按自己的 remote 验证 SHA；rev11 直接选择各 remote 的 latest，`f03c2454` 仅保留为 ancestry 证据 | §10、§12 提交 0、§13-2 |
| P1-B | axis1 从"只比较总元素数"一步收紧为唯一的自然二维 shape，会拒绝已存在的 `[1, M*N/32]` IR | axis1 同时接受 canonical `[M, N/32]` 与 legacy flat `[1, M*N/32]`；axis0 仍只接受自然二维 `[M/32, N]` | §7.1、§11.1/11.3、§13-3、§14 |
| P1-C | `pto.tmov` 的 X-to-ZZ 形态只查 valid shape/元素数/tmp，没有约束 ISA 实际假定的紧密 source、physical stride 和 padded capacity | 增加 compact-prefix、ND `align16(rows) * cols` 容量、DN source stride 及静态 physical shape 契约；无法静态证明时拒绝 | §3.2、§7.2、§11.4 |
| P1-D | ND X-to-ZZ 的 `src` 只声明 `Read`，但 `ZeroSourcePaddingB16` 会写 source padding | axis1/ND 的 `src` 改为 `Read + Write`；axis0/DN 仍为 `Read` | §9、§11.2 |

**rev3（历史记录，pin 结论已被 rev4 覆盖）**：按第二轮评审修正 2 个 P1 + 2 个 P2。

| 编号 | rev2 的错误 | rev3 当时结论 | 位置 |
|---|---|---|---|
| P1-A | 说 updater "漏更" `ci_sim.yml` / `Dockerfile.dev`，应扩展覆盖 | 正确识别了 repo-aware 建模和 CANN 9.0 独立兼容目标，但进一步错误断言两个 remote 必然不共享 SHA；该断言已由 rev4 的 `f03c2454` / ancestry 验证替换 | §10、§12 提交 0、§13-2 |
| P1-B | 四条 interleave 约束只有一条笼统负向用例 | 按 valid/physical × rows/cols 拆成 4 条，另加顺序哨兵与 max 形状哨兵 | §11.3 |
| P2-A | physical 检查只判 exp 侧动态，未判 src 侧 → `srcRowsPhys` 动态时对合法 IR 误报 | interleave 下**显式拒绝动态 physical shape**，之后全在静态值间比较；valid 仍可动态（跳过） | §7.1 |
| P2-B | 先按 interleave 分派 exp 形状、后查 `!isDn`，导致 axis1+interleave 先报误导性 shape 错 | `interleave && !isDn` 提到所有 shape 推导之前，作为步骤 (0) | §7.1 |

**rev2**：按第一轮评审修正 5 个 P1 + 2 个 P2。这些都是会直接写进 verifier 的
硬约束，rev1 的版本会产出**能通过 verifier 但在设备上越界、不写输出或编译失败**的 IR：

| 编号 | rev1 的错误 | rev2 的结论 | 位置 |
|---|---|---|---|
| P1-1 | ND `tmp` 下界写成 `32 + …` | 常数项是 `2 * BLOCK_SIZE = 64`，两端各一个 32B block；少 32B 会被尾部 `vstus` 越界写 | §3.2、§7.2 规则 12 |
| P1-2 | `interleave` 校验自相矛盾：exp/max/scaling 先按普通分组形状统查，随后又要求 exp 是 `[M/64, 2N]` | max/scaling 恒为分组形状，**只有 exp 随 interleave 二选一**；另补 64 对齐与 physical `align32(2N)` | §7.1 |
| P1-3 | ND 规则用错坐标系：对 exponent 列数写 `% 64 == 0` | exponent 列数已是 `N/32`，真实约束是**列数为偶数**；行对齐要求撤销（ISA 用 ceil + 零填充显式支持非 16 对齐） | §7.2 规则 9 |
| P1-4 | 把 `srcRows == 1`（`M = 32`）当退化 identity 放行，还列了 ST 正向用例 | `numPairs = hatM / 2 = 0`，主循环零次迭代，**`dst` 根本不被写**；改为拒绝，用例转负向 | §7.2 规则 10、§11.6 |
| P1-5 | 元素类型集合写成 `{ui8, i8, f8E8M0}` | ISA 接受 `uint8_t` / `hifloat8_t` / `float8_e8m0_t`；`i8` 降成 `int8_t` 会在 C++ 期炸 `static_assert`，`!pto.hif8` 被漏 | §7.2 规则 5 |
| P2-1 | 称 DN `tmp` 下界"未知"，并计划做大小 `tmp` 对照实验 | DN 实现是 `(void)tmp;`，**根本不用**；对照实验无意义，内存效应改为按轴区分 | §3.2、§9、§13-1 |
| P2-2 | 称 pin 有"五处、由 updater 统一管理"，并把 pin 覆盖列为开放问题 | 树内有 **3 个不同 SHA**、updater 只覆盖 **3 处**（漏 `ci_sim.yml`、`Dockerfile.dev`）；`interleave` 必须 bump，已非开放问题 | §10、§12 提交 0、§13-2 |

另新增开放问题 7（动态维度处理）与对应的 `dn_dynamic` lit 用例；该 rev3 方案已被
rev7 的“无法静态证明即拒绝”结论取代，见 §7.1、§11.3 与 §13-7。

**核对状态**：rev2 的 P1-1/3/4/5 与 P2-1/2 均已在基线快照与本仓库源码上
逐条自证。rev4 又直接核对了 GitHub `f03c2454`：该 commit 存在，与
`a8168c6...f03c2454` 的比较结果为 ahead 8 / behind 0；`TMov.hpp` 中 ND 路径确实以
`validRow` / `validCol` 生成索引、零写 padding，而不读取 tile row stride。这些结论分别固化在
§13-2 与 §3.2/§7.2。任意 pin 在写入具体目标前，仍必须对该目标自己的 remote 做存在性验证。

### 1.2 已拍板事项

- **ND shape 兼容**：axis1 的 canonical `[M, N/32]` 与 legacy flat
  `[1, M*N/32]` 都是长期合法形态；legacy flat 要求 source tight，padded source 改用
  canonical 2D。`69a81f3/40e741b` 下 canonical 2D 的 B16 实现会破坏可观察输出，故本版
  verifier 只放行 f32 canonical；B16 canonical 先明确拒绝。仓库内两个既有 MXFP8
  legacy-flat case 都满足 tight 条件。
  axis0 只接受 `[M/32, N]`。
- **EmitC 生成形态**：除 deprecated `exp_zz` fused 形态外，所有
  `pto.tquant.mx` 统一生成 `TQUANT<grp_axis, MxQuantAlg[, interleave]>(...)`。
  默认 ND 不保留另一套类型列表 EmitC 文本。
- **pin 建模 ownership**：本 PR 落地 `(repo, revision, 兼容性约束)` 三元组、
  repo-aware target map/updater 和本特性 pin bump。[#1122](https://github.com/hw-native-sys/PTOAS/pull/1122)
  已于 2026-08-10 合入，只作为 PTO-BC 兼容处理的基线；其后
  `.github/scripts/update_pto_isa_pin.py` 已被删除，本 PR 需要重新引入 repo-aware 管理，
  不是扩展一个仍存在的脚本。GitCode 的 CI/主 Dockerfile/remote-validation 对齐
  `69a81f3`，GitHub `ci_sim` 对齐已合入同步 PR #239 的 `40e741b`；不等待新的
  pto-isa commit。`Dockerfile.dev` 继续作为 CANN 9.0 独立兼容目标。
- **X-to-ZZ 的 IR 形态**：复用现有 `pto.tmov`。第三个 tile 位于 scaling 地址空间时
  保持既有 FP 语义；第三个 tile 位于非-scaling 地址空间时表示 X-to-ZZ `tmp`。
  `grpAxis` 只允许用于后一形态，缺省为 axis1。
- **`TMovOp` API 兼容**：ODS operand 名保留为 `$fp`，Python 的 `fp=`、C++ 的
  `getFp()` 和既有 builder 均不改名；实现内部通过共享 helper 把 `getFp()` 返回值按
  地址空间分类为 FP 或 X-to-ZZ tmp。
- **`TQuantMxOp` 内存契约**：valid shape 合法只是第一层；还必须满足 §7.1 的
  physical stride、compact-prefix、branch-selection 与 capacity 矩阵。f16/bf16 source
  存在列 padding 时，source memory effect 为 `Read + Write`。
- **latest-pin 支持边界**：`max` 是可观察输出，`exp` / `dst` 只能在各自声明的
  physical allocation 内写入。`69a81f3/40e741b` 的 `TQuant.hpp` 仍有 canonical B16
  借用 `exp/max`、DN FP16 MXFP4 借用 `dst`、DN FP32 interleave 借用 `dst` 以及 B16
  VL-tail 越界写。rev11 不再要求 pto-isa 先合修复；§7.1.1 直接把无法在当前实现上安全
  证明的组合标为 unsupported，并由 verifier 拒绝。以后 pto-isa 修复这些路径时，再单独
  放宽 verifier 和增加正向精度用例，不能仅 bump SHA 就静默扩大合法 IR。

本文只定义 A5 MX 量化的分组轴（`grp_axis`）表达，以及 E8M0 exponent 从 ND/DN
到 ZZ 的布局转换在 PTO IR 与 EmitC 中的表达方式。不改变 INT8 量化；扩展
`pto.tmov` 的第三个 tile operand 语义，但不改变其已有 ND-to-NZ / acc-to-vec / FP 形态。

## 2. 结论摘要

1. **不新增 `mx_alg` 属性。** PTOAS 已有的 `quant_type` × `quantScaleAlg` 与
   pto-isa `MxQuantAlg` 是精确的一一映射，`MxQuantAlg` 由 EmitC 合成。
2. **`pto.tquant.mx` 新增 `grp_axis` 与 `interleave` 属性**，默认
   `grp_axis = 1`、`interleave = false`，保持两个 tight MXFP8 legacy-flat case 的文本/语义兼容；
   过去被放行的 flat+padded source 属于可证明越界的非法形态，rev8 明确拒绝。
3. **verifier 的分组 shape 校验按轴分派**：axis0 只接受自然二维形状；
   axis1 的 shape 分类同时识别 canonical `[M, N/32]` 与 legacy flat
   `[1, M*N/32]`。随后应用 latest 支持边界，canonical B16 暂不放行。`max`/`scaling`
   恒为分组形状，只有 `exp` 随 `interleave` 改形（§7.1）。
4. **不带 `exp_zz` 的 `pto.tquant.mx` 统一生成 grouped EmitC 形态**，不根据
   `grpAxis` 是否显式写出而保留双文本形态（§8.1）。
5. **exponent X-to-ZZ 复用现有 `pto.tmov`**。三参数形态按第三个 tile 的地址空间
   分派：scaling 继续生成既有 FP TMOV；非-scaling 进入 X-to-ZZ，生成
   `TMOV(...)` / `TMOV<0>(...)`。后者还要按独立语义分支校验 compact source、
   physical stride、padded capacity，并声明按轴不同的 memory effects（§5.3、§7.2、§9）。
6. **ND X-to-ZZ 是 source-mutating op**：为支持非 16 对齐行，ISA 会将 source padding
   清零，因此 `src` 的内存效应是 `Read + Write`（§9）。
7. **`pto.tquant.mx` 同样可能修改 source**：f16/bf16 且 valid 列数小于 physical
   列数时，ND/DN 都会原地清零 source 的行尾 padding；该形态的 `src` 必须建模为
   `Read + Write`。同时按 §7.1 静态证明所有输出 stride/capacity；axis1 flat 只允许
   tight source，padded source 只有 f32 可使用 canonical 2D exp；B16 canonical 在 latest
   pin 下拒绝（§7.1、§9）。
8. **`dst` 不是只检查自身 footprint**：axis1 的 MXFP8/MXFP4 与 axis0 的 MXFP4
   使用 source-derived output stride，必须按 packed factor 校验 destination physical
   cols/capacity（§7.1.1）。
9. **pin 三元组与 bump 由本 PR 落地**；GitCode 目标统一到 `69a81f3`，GitHub
   `ci_sim` 使用 `40e741b`；#1122 只作为 PTO-BC 兼容实现基线。原 updater 已删除，本 PR
   重新落地 repo-aware target map/updater（§1.2、§12）。
10. **pin 对齐 latest，不承担修复 TQuantMx 的职责**：`69a81f3/40e741b` 的隐藏
    scratch 与 padding tail 被记录为当前实现限制。无法保证 `src/exp/max/dst` 语义的组合
    在 verifier 中拒绝，相关 redzone case 只作 known-behavior 诊断；不等待额外 pto-isa
    合入（§3.1.1、§7.1.1、§11.3、§13-10）。
11. **现有 `exp_zz` + `storeMode` fused 路径标记为 deprecated**，理由见 §5.4：
   该 overload 在快照里只有 CPU-sim 实现，A5 设备头文件没有对应实现。

## 3. PTO-ISA 侧现状（已对快照与 rev11 两个 latest pin 逐条核对）

### 3.1 grouped MX TQUANT

`include/pto/npu/a5/TQuant.hpp:3077`：

```cpp
template <int grp_axis, MxQuantAlg mx_alg, typename TileDataOut, typename TileDataSrc,
          typename TileDataExp, typename TileDataMax, typename TileDataScaling>
PTO_INTERNAL void TQUANT_IMPL(TileDataOut& dst, TileDataSrc& src,
                              TileDataExp* exp, TileDataMax* max, TileDataScaling* scaling)
```

实现内部把 `mx_alg` **就地解码成 `(QuantType, QuantScaleAlg)` 二元组**：

```cpp
constexpr bool isFp8 = (mx_alg == MxQuantAlg::OcpMxFp8E4M3 || mx_alg == MxQuantAlg::NvMxFp8E4M3);
constexpr bool isNv  = (mx_alg == MxQuantAlg::NvMxFp8E4M3  || mx_alg == MxQuantAlg::NvMxFp4E2M1);
constexpr QuantType     quant_type = isFp8 ? QuantType::MXFP8 : QuantType::MXFP4_E2M1;
constexpr QuantScaleAlg scale_alg  = isNv  ? QuantScaleAlg::NV : QuantScaleAlg::OCP;
```

`grp_axis == 0` 走 `TQuant_MXFP8_Impl_DN` / `TQuant_MXFP4_E2M1_Impl_DN`；
`grp_axis == 1` 回落到既有的 `TQUANT_IMPL<quant_type, scale_alg, ...>` ND 流水线，
其余取值由 `static_assert(grp_axis == 1, ...)` 拒绝。

### 3.1.1 当前目标实现的额外 stride、隐藏 scratch 与 padding-tail 行为

rev11 复核 GitCode `cann/pto-isa` `master@69a81f3`。随后
[GitHub sync PR #239](https://github.com/hw-native-sys/pto-isa/pull/239) 已合并为 GitHub
`hw-native-sys/pto-isa` `main@40e741b`；merge commit message 明确记录
`sync: GitCode master @ 69a81f3b2d14`。PR #239 不修改 `TQuant.hpp`，所以以下行为在两个
latest pin 中都存在。rev11 的 pin 决策仍是直接使用这两个已发布分支头；本节用于界定
PTOAS 在该基线下可以安全放行的 IR，不再用于搜索另一个未来 SHA。

**axis1 flat 分支按 source physical cols 计算 group 数。**
`TQuant_MXFP8_Impl` / `TQuant_MXFP4_E2M1_Impl` 都先计算：

```text
totalElems = srcValidRows * TileDataSrc::Cols
numGroups  = totalElems / 32
```

当 `TileDataExp::Rows == 1` 时，exp/max/scaling 都被 reshape 成 1D flat prefix，后续按
`numGroups` 连续写。因而 legacy flat 的语义 group 数 `M*N/32` 只有在
`srcPhysicalCols == srcValidCols` 时成立。反例
`src valid=16x32, physical=16x64` + 三个辅助输出 `valid/physical=1x16` 会按 32 个 group
写入 16 元素 allocation。结论不是扩大 legacy 输出到 physical group 数，而是：

- flat exp 分支强制 source tight：`srcPhysicalCols == srcValidCols`；
- padded source 必须选择 canonical 2D exp，让实现按每行 `validCols/32` 个 group 处理；
  若 source 是 B16，再由 §7.1.1 的 latest 支持规则拒绝。

**destination 也使用 source-derived stride。** axis1 MXFP8 的 F32/B16 2D 实现都以
`dstPtr + row * srcPhysicalCols` 定位输出行；axis1 MXFP4 以
`dstPtr + row * (srcPhysicalCols/2)` 定位 packed byte 行。axis0 MXFP4 的 BF16/FP16
路径同样使用 source physical cols / 2，而不是 `TileDataOut::Cols`。因此仅验证 dst
自身 `physical >= valid` 不足以证明寻址安全，§7.1.1 必须按 quant type 校验 destination
physical row stride 与 capacity。PTO 的 packed FP4 tile shape 以**packed pair / byte**计数，
所以这里的 `/2` 已经是 destination tile 的列单位，不是逻辑 FP4 标量数。

**当前实现存在五类不可接受的隐藏 scratch / 越界 padding。** 前三类由 rev8 固定：

1. axis1 MXFP8 canonical 2D 且 `totalGroups < 8` 时，把临时 scale 写到
   `exp + row*expStride + 16`。例如 `exp valid/physical=2x1` 从偏移 16 开始写，普通
   canonical allocation 无法容纳。
2. axis1 MXFP8 `totalGroups >= 8` 时改用 `max` 开头作为 scale scratch，覆盖已经写出的
   per-group max；axis1 MXFP4 canonical 2D 始终把 `max` 用作同类 scratch。
3. axis0/DN FP16 MXFP4 在 packed destination 行 stride 非 32B 对齐时，从
   `dst + TileDataOut::Rows*srcPhysicalCols/2` 开始使用 scratch，位置已经越过按 tile
   physical shape 声明的 destination footprint。

rev10 再确认两类：

4. axis0/DN MXFP8 的 `interleave=true && src=f32` 路径先把每个 32-row group 的线性
   exponent 写进尚未产出量化结果的 `dst`，scratch row stride 为
   `align32(srcPhysicalCols)`，scratch rows 为 `srcValidRows/32`，最后才用正常 quantized
   output 覆盖 destination。反例 `src valid=64x1, physical=64x128`、
   `dst valid/physical=64x1`、`exp physical=1x256`、max/scaling physical `2x128`
   满足 rev8 的全部 IR shape 规则，但第二个 scratch row 从 `dst+128` 开始，而 dst
   allocation 只有 64 bytes。普通 destination footprint 是正确的 IR 契约；借用 dst
   是 latest 实现的限制，不能由 tight-dst IR 安全承诺覆盖。
5. B16 source 使用 VL-aligned padding 分支时，周期 predicate 覆盖一个完整 VL 的
   `rowsPerVL = elemPerVL/srcPhysicalCols` 行；`vlCount = ceil(validRows*srcPhysicalCols/elemPerVL)`，
   但最后一次 `vsts` 没有按剩余 valid rows 收紧 predicate。反例 bf16
   `src valid=32x1, physical=32x2` 中一个 VL 可容纳 64 行，唯一一次 store 会同时写
   后 32 个不存在行的 padding 列。在 latest pin 上，凡进入 VL-aligned 分支且最后一个
   VL 不完整的 padded B16 source 必须由 verifier 拒绝；本设计不要求用户为越界 store
   额外分配虚假行。

这些行为不能通过“再给 src/exp/max/dst 多分一点容量”统一合法化：`max` 在 PTO IR 和手册中是
可观察输出，覆盖它会改变 op 语义；B16 tail 也是实现缺少 predicate，而不是 IR 需要额外 rows。
rev11 因此采用 **latest pin + 显式支持边界**：commit 0 只把 PTOAS 对齐到
`69a81f3/40e741b`，不要求 pto-isa 再合任何提交；§7.1.1 对命中上述行为且无法静态证明安全的
组合直接拒绝。将来上游实现变化时，放宽 verifier 必须与新的原生 redzone/max golden 和
PTOAS 正向精度用例同提交完成，不能靠 pin bump 隐式改变合法集合。

### 3.2 exponent X-to-ZZ

`include/pto/npu/a5/TMov.hpp:704` 附近：

```cpp
template <int grp_axis = 1, typename DstTileData, typename SrcTileData, typename TmpTileData,
          std::enable_if_t<(TmpTileData::Loc != TileType::Scaling), int> = 0>
PTO_INTERNAL void TMOV_IMPL(DstTileData& dst, SrcTileData& src, TmpTileData& tmp)
{
    CommonCheckZZ<DstTileData, SrcTileData, TmpTileData>();
    if constexpr (grp_axis == 0) {
        TMovDnTo2Zz<...>(dst.data(), src.data(), tmp.data(), src.GetValidRow(), src.GetValidCol());
    } else {
        TMovNdTo2Zz<...>(dst.data(), src.data(), tmp.data(), dst.GetValidRow(), dst.GetValidCol());
    }
}
```

四条必须写进设计的事实：

1. **SFINAE 用 `Tmp::Loc != TileType::Scaling` 把 X-to-ZZ 与 FP 三参数 TMOV 区分开。**
   若 `tmp` 落在 scaling 地址空间，会静默命中 FP overload。
2. **ND 用 `dst` 的 valid shape 驱动，DN 用 `src` 的 valid shape 驱动。** 这是不对称的，
   verifier 与文档都必须显式说明，否则用户会把 valid 写在错误的一侧。
3. 默认模板参数 `grp_axis = 1`，所以不带模板参数的三参数 `TMOV` 即 ND-to-ZZ。
4. **两条路径都不使用 tile 类型的 physical row stride。** ND 把 source 当成
   `validRow * validCol` 个紧密连续元素，DN 直接用 `srcValidCols` 作为行步长。
   因此 physical capacity 足够不等于布局就合法；带 row padding 的自然二维 source
   会被错误解释。

`CommonCheckZZ` 与两个实现的静态约束：

| 约束 | 内容 |
|---|---|
| 元素类型 | `uint8_t` / `hifloat8_t` / `float8_e8m0_t` 三者之一 |
| 类型一致 | `dst`、`src`、`tmp` 元素类型必须完全相同 |
| `src` 布局 | `isRowMajor && SFractal == SLayout::NoneBox` |
| `dst` 布局 | `isRowMajor && SFractal == SLayout::RowMajor` |

ND-to-ZZ 的 `tmp` 字节数在实现里是闭式公式：

```text
rowBlockCount = ceil(validRow / 16)
P             = validCol / 2
tmpBytes      = (BLOCK_SIZE / sizeof(uint16_t) + rowBlockCount * P
                 + BLOCK_SIZE / sizeof(uint16_t)) * sizeof(uint16_t)
              = 2 * BLOCK_SIZE + rowBlockCount * validCol      // 化简，BLOCK_SIZE = 32
              = 64 + ceil(validRow / 16) * validCol
```

**两端各有一个 32 字节 block**，不是共用一个：`basePtr` 占前 `blkElem = 16` 个 `uint16_t`
（`GenerateB8IndicesZZToUB` 开头的 `vsts(vb16_base, basePtr, ...)`），`offsetBuf` 从
`basePtr + blkElem` 起占 `rowBlockCount * P` 个 `uint16_t`，循环之后还有一次
`vstus(ureg_align, blkElem, ...)` 写出**尾部又一个 16 元素块**。因此常数项是 `2 * 32 = 64`。
`rowBlockCount * P` 个 `uint16_t` 即 `rowBlockCount * validCol` 字节（`P = validCol / 2`）。

> 早期版本的本文档把常数项误写为 32。按那个下界放行的 `tmp` 会**少 32 字节**，
> 尾部 `vstus` 直接越界写 UB，属未定义行为且极难定位。规则 12 必须用 64。

其中 `validRow` / `validCol` 取自 **`dst`**（见上面第 2 条）。

ND 还有两个容易被忽略的 observable memory 行为：

```text
paddedRows = alignTo(validRow, 16)
paddedElems = paddedRows * validCol
```

- `ZeroSourcePaddingB16` 把 source 中 `[validRow * validCol, paddedElems)` 这段 padding
  **原地写零**；因此 source 不是只读的，且 allocation 必须容纳 `paddedElems`。
- gather/store 总共向 destination 的连续前缀写入 `paddedElems` 字节，不是只写
  `validRow * validCol`。destination 也必须容纳该 padded 前缀。

这里的"紧密 source"定义为：所有 valid 元素在 row-major allocation 起始处构成
无空洞的连续前缀。对静态 rank-2 tile，可判定为
`srcValidRows == 1 || srcPhysicalCols == srcValidCols`。第一个分支保留 axis1 的
legacy flat `[1, M*N/32]` source；自然二维 source 则必须以 valid 列数为物理行步长。
例如 `valid=20x4, physical=20x32` 虽然总容量足够，但 valid 行之间有 28 字节
padding，ISA 却会按 stride 4 读取，必须由 verifier 拒绝。

`valid=20x4, physical=20x4` 虽然紧密，但只有 80 字节；ND 需要
`align16(20) * 4 = 128` 字节，因此 source 的 padding 零写与 destination 的 padded
写出都会越界。这个例子要分别作为 source/destination capacity 负向用例。

DN-to-ZZ **完全不使用 `tmp`**：`GenerateB8IndicesDN2ZZToUB` 函数体第一行就是 `(void)tmp;`，
上方注释也写明 "tmp is unused (kept in the signature to match the ND->ZZ TMOV interface)"。
它只做 `vlds` + `vintlv` + `vsstb`，不需要索引表。

DN 的 source 步长同样不是从 `SrcTileData::Cols` / `RowStride` 取值：实现使用
`row1Base = row0Base + srcValidCols`。由于 DN 已要求 `srcValidRows >= 2`，这里没有
legacy 单行特例，必须直接要求 `srcPhysicalCols == srcValidCols`。destination 写出
`srcValidRows * srcValidCols` 个元素，也需要不小于该值的 physical capacity。

所以 DN 的 `tmp` ISA 使用量**不是"未知"，而是 0**；`tmp` 在 DN 形态下纯粹是为了对齐
ND 的三参数签名而保留的占位。本设计不对 DN 的 `tmp` 施加 ISA 最小容量公式，其内存效应
按轴区分，见 §9；但它仍须有静态 physical shape，供 PlanMemory 分配并供 EmitC 实例化
`Tile<...>`，见 §7.2 规则 14。

### 3.3 fused NZ store overload 的真实位置

`pto.tquant.mx` 现有的 `exp_zz` + `storeMode` 形态对应 pto-isa 的六 tile overload。
快照里：

- 公共 wrapper 在 `include/pto/common/pto_instr.hpp:2460`；
- **实现只在 `include/pto/cpu/TQuant.hpp:545`（CPU-sim）**；
- `include/pto/npu/a5/TQuant.hpp` 里 grep `VecStoreMode` / `store_mode` **无任何命中**。

即：该形态在 `__CPU_SIM` 下可用，A5 设备路径没有对应实现。这是 §5.4 决策的直接依据。
合入前需按仓库当前 pin 的 pto-isa 复核该结论（见 §13-2）。

## 4. PTOAS 侧现状核查

issue 对现状的描述与代码有出入，先对齐事实。

### 4.1 `pto.tquant.mx` 已有的能力

[`include/PTO/IR/PTOOps.td` `def TQuantMxOp`](../../include/PTO/IR/PTOOps.td)：

```tablegen
let arguments = (ins
  PTODpsType:$src, PTODpsType:$dst, PTODpsType:$exp,
  PTODpsType:$max, PTODpsType:$scaling,
  Optional<PTODpsType>:$exp_zz,
  PTO_QuantTypeAttr:$quant_type,
  DefaultValuedAttr<PTO_QuantScaleAlgAttr, "::mlir::pto::QuantScaleAlg::OCP">:$quantScaleAlg,
  OptionalAttr<PTO_VecStoreModeAttr>:$storeMode
);
```

对照 issue 的四条诉求：

| issue 诉求 | 现状 |
|---|---|
| 输出 `dst/exp/max/scaling` 四个结果 | **已具备** |
| 携带 `MxQuantAlg` | **已可表达**：`quant_type`(MXFP8/MXFP4_E2M1) × `quantScaleAlg`(OCP/NV) = 4 组合 |
| 携带 `grp_axis` | 缺失 |
| 携带 `interleave` | 缺失 |
| exponent → ZZ | 存在 `exp_zz` + `storeMode` fused 形态（issue 未提及） |

### 4.2 现有 verifier 的分组校验是轴无关的

[`lib/PTO/IR/PTO.cpp` `TQuantMxOp::verify()`](../../lib/PTO/IR/PTO.cpp) 关键片段：

```cpp
if (srcCols % 32 != 0)
  return emitOpError("expects src valid_shape[1] to be a multiple of 32 for tquant.mx");
int64_t groups = (srcRows * srcCols) / 32;
if (expElems != groups) emitOpError("expects exp valid element count to equal src valid elements / 32");
// max / scaling / exp_zz 同样只比较总元素数
```

后果有两条，都要修：

1. **`srcCols % 32` 是 ND 专属约束**，DN 要求的是 `srcRows % 32 == 0`。
2. 只比较总元素数 ⇒ `[M, N/32]` 与 `[M/32, N]` 对 verifier 等价，**写错轴不会被拦**，
   而生成的 C++ 会调到 ND 实现。这正是 issue 第 2 条诉求的实际价值。

### 4.3 现有 EmitC 的模板参数形状与新接口不同

[`PTOToEmitC.cpp` `PTOQuantMxToEmitC`](../../lib/PTO/Transforms/PTOToEmitC.cpp) 当前生成：

```cpp
TQUANT<QuantType, [VecStoreMode,] DstT, SrcT, ExpT, MaxT, ScalingT [, QuantScaleAlg]>(
    dst, src, &exp, &max, &scaling [, &exp_zz]);
```

即**显式写出全部 tile 类型**。而目标接口是 `TQUANT<grp_axis, mx_alg>(...)`，tile 类型
由实参推导。两者不是"追加一个模板参数"的关系，是**两族不同的 overload**，EmitC 必须
按形态分支，不能在现有参数列表上做增量。

`pto.tmov` 的 lowering 也必须按第三个 tile 的地址空间分派：既有普通/FP 形态继续生成
原调用；非-scaling 的第三个 tile 选择 X-to-ZZ，生成 `TMOV(dst, src, tmp)` 或
`TMOV<0>(dst, src, tmp)`。

### 4.4 `pto.tmov` 现状

`def TMovOp` 当前包含 `src`、`dst`、`Optional fp`、`Optional preQuantScalar`、
`accToVecMode`、`reluPreMode`。第三个 tile operand 在 ODS 中固定叫 `fp`，verifier
强制它位于 scaling 地址空间，lowering 也据此进入 FP 分支；因此当前 PTO IR 尚不能
写非-scaling `tmp`。

pto-isa 已经提供了无歧义的编译期分派：`FpTileData::Loc == TileType::Scaling` 命中
FP overload，`TmpTileData::Loc != TileType::Scaling` 命中 X-to-ZZ overload。PTO IR
可以复用同一规则，无需为底层同一 `TMOV` overload 家族再增加独立 op。

### 4.5 与 `Layout::MX_A_ZZ` / `MX_B_NN` 无关

`InferPTOLayout.cpp` 里的 `Layout::MX_A_ZZ` / `MX_B_NN` 是 **GM `tensor_view` 的布局枚举**，
由 `tload` 目标 tile 的 `(blayout, slayout)` 推断而来，和本文讨论的 **tile 级 ZZ 布局**
是两个不同层次的概念。本设计不触碰它们，与 issue 中"不要求 PTOAS 识别这两个名称"一致。

## 5. 设计决策

### 5.1 决策一：不新增 `mx_alg` 属性

**做法**：保留 `quant_type` + `quantScaleAlg`，EmitC 合成 `MxQuantAlg` token。

| `quant_type` | `quantScaleAlg` | 生成的 `MxQuantAlg` |
|---|---|---|
| `MXFP8` | `OCP` | `MxQuantAlg::OcpMxFp8E4M3` |
| `MXFP8` | `NV` | `MxQuantAlg::NvMxFp8E4M3` |
| `MXFP4_E2M1` | `OCP` | `MxQuantAlg::OcpMxFp4E2M1` |
| `MXFP4_E2M1` | `NV` | `MxQuantAlg::NvMxFp4E2M1` |

**理由**：
- pto-isa 自己就是把 `mx_alg` 解码成这个二元组（§3.1），映射是机械的、无信息损失；
- 新增 `mx_alg` 会与两个既有属性构成双真值源，必须再写一致性校验，而且 INT8 路径仍要
  用 `quant_type`，属性语义会分裂；
- 避免一次破坏性 schema 变更（`quant_type` 已进入 PTO-BC v0 的属性字典与下游 IR）。

**代价**：IR 与 pto-isa 的字面形态不同构。通过 §7 的诊断文案和用户手册消化。

### 5.2 决策二：`grp_axis` / `interleave` 作为属性加在 `pto.tquant.mx`

`grp_axis` 是编译期模板参数，不是运行期值，用属性而非 operand。默认 `1` 使现有 IR
逐字不变（不写属性 = ND）。

### 5.3 决策三：X-to-ZZ 复用并扩展现有 `pto.tmov`

`pto.tmov` 的可选第三个 tile operand **继续使用 ODS 名 `$fp`**，从而保留生成的
Python `fp=` keyword、C++ `getFp()` accessor 和 builder 签名。只有它的运行语义由
静态地址空间分类；helper 内可使用局部变量名 `aux`，但不能改 ODS/public API 名：

```text
getFp() 不存在              -> 既有二参数 TMOV
getFp().loc == scaling      -> 既有 FP TMOV，第三个 tile 解释为 fp
getFp().loc != scaling      -> X-to-ZZ TMOV，第三个 tile 解释为 tmp
```

这个分派与 pto-isa 的 SFINAE 条件逐字同构：FP overload 要求
`FpTileData::Loc == TileType::Scaling`，X-to-ZZ overload 要求
`TmpTileData::Loc != TileType::Scaling`。因此 PTO IR 不需要额外 `mode=x2zz` 属性；
`grpAxis` 只在第三个 tile 为非-scaling 时合法，缺省表示 axis1。

以 DN exponent `2x16` 为例，普通二参数 `TMOV(dst, src)` 只复制连续数据：

```text
src row 0: a0 a1 a2 ... a15
src row 1: b0 b1 b2 ... b15
plain dst: a0 a1 a2 ... a15 b0 b1 b2 ... b15
```

非-scaling `tmp` + `grpAxis=axis0` 选择三参数 `TMOV<0>(dst, src, tmp)`，结果是
Cube 所需的 ZZ box 交织顺序：

```text
ZZ dst: a0 b0 a1 b1 a2 b2 ... a15 b15
```

因此复用的是同一个 `pto.tmov` IR op 和 pto-isa `TMOV` overload 家族，不是把 X-to-ZZ
误降成普通二参数 copy。

| `pto.tmov` 形态 | 第三个 tile | 合法属性 | verifier / effects | EmitC |
|---|---|---|---|---|
| 普通 move | 无 | 既有 `accToVecMode` / `reluPreMode` 组合 | 完全保持现状 | 既有二参数或 scalar 形态 |
| FP move | `loc=scaling`，角色为 `fp` | 既有 FP / pre-quant 约束；不允许 `grpAxis` | 完全保持现状 | 既有 `TMOV_FP` / FP 参数化 `TMOV` |
| X-to-ZZ | 非-scaling，角色为 `tmp` | 可选 `grpAxis`；禁止 `preQuantScalar`、`accToVecMode` 与非默认 `reluPreMode` | 走 §7.2 与 §9 的 X-to-ZZ 分支 | axis1: `TMOV(dst, src, tmp)`；axis0: `TMOV<0>(dst, src, tmp)` |

**地址空间只完成 overload 分类，不等于完成合法性证明。** “非-scaling”只能说明第三个
tile 不是 `fp`；它还必须满足 X-to-ZZ 的 vec 地址空间、E8M0-compatible dtype、layout、
shape、stride、capacity 和按轴效应契约。所有消费者通过一个共享的
`classifyTMovForm(getFp())` helper 得到 `NoTileAux` / `Fp` / `XToZz`，禁止各自复制一套判断。

复用现有 op 的兼容性依据：

1. 旧 verifier 从未接受非-scaling 的第三个 tile，因此没有既有合法 IR 会被重新解释成
   X-to-ZZ；旧二参数、scaling `fp` 和 `preQuantScalar` 形态行为不变。
2. ODS 名、operand 顺序、segment size、文本位置以及 Python/C++ builder/accessor
   全部保持不变；`grpAxis` 走属性字典。仓库已有 `pto.TMovOp(..., fp=fp_scaling)`
   调用无需迁移。
3. `PTORemoveIdentityTMov` 必须继续把任何带第三个 tile 的形态视为非 plain；
   `PTOA5NormalizeTMov` 必须显式跳过 `XToZz`，不能把布局重排改写成普通 copy。
4. `TMovOp::verify()`、`getEffects()` 与 EmitC lowering 都按同一个分类结果分支，避免
   verifier 认为是 tmp、lowering 却认为是 fp 的跨层分歧。

**代价**：`pto.tmov` 不再只有一套 shape/effect 契约，所有读取第三个 tile 或假设普通
move 语义的消费者都必须完成审计。相比新增 op，该方案不增加 IR surface、opcode 和
binding，并与 pto-isa 的现有 overload 选择保持一致。

### 5.4 决策四：`exp_zz` + `storeMode` 标记 deprecated

**做法**：本次不删除，行为不变，但：

- verifier 增加一条：`exp_zz` 形态与 `grp_axis = 0` **互斥**（该 fused 路径只有 ND 语义）；
- ODS `description` 与用户手册标注 deprecated，推荐改用 `pto.tquant.mx` +
  `pto.tmov` 的非-scaling `tmp` 形态；
- ReleaseNotes 记录该状态。

**理由**：§3.3 已确认该 overload 在快照里只有 CPU-sim 实现。把 `grp_axis` 也接进这个
fused 形态，等于在一条设备侧不可用的路径上继续加语义。删除则属于独立的破坏性变更，
应单独走一个 PR（含 PTO-BC 兼容处理），不与本特性混合。

## 6. IR 设计

### 6.1 `pto.tquant.mx` 新增属性

```tablegen
// include/PTO/IR/PTOAttrs.td
def PTO_MxGroupAxisEnum : PTO_I32Enum<
  "MxGroupAxis", "MX quantization grouping axis", [
    I32EnumAttrCase<"Axis0", 0, "axis0">,   // DN: 沿 axis 0 / 行方向每 32 元素分组
    I32EnumAttrCase<"Axis1", 1, "axis1">    // ND: 沿 axis 1 / 列方向每 32 元素分组
  ]>;
def PTO_MxGroupAxisAttr : EnumAttr<PTO_Dialect, PTO_MxGroupAxisEnum, "mx_group_axis">;
```

用 enum 而非裸 `I64Attr`，使非法取值在解析期就被拒绝，无需 verifier 兜底。

```tablegen
// include/PTO/IR/PTOOps.td, def TQuantMxOp 的 arguments 追加
DefaultValuedAttr<PTO_MxGroupAxisAttr, "::mlir::pto::MxGroupAxis::Axis1">:$grpAxis,
DefaultValuedAttr<BoolAttr, "false">:$interleave
```

IR 形态：

```mlir
// ND（默认，与现有 IR 逐字兼容）
pto.tquant.mx ins(%src : !pto.tile_buf<vec, 16x64xf32>)
              outs(%dst, %exp, %max, %scaling : ...)
              {quant_type = #pto<quant_type MXFP8>}

// DN
pto.tquant.mx ins(%src : !pto.tile_buf<vec, 64x16xf32>)
              outs(%dst, %exp, %max, %scaling : ...)
              {quant_type = #pto<quant_type MXFP8>,
               grpAxis = #pto<mx_group_axis axis0>}

// DN + interleave
pto.tquant.mx ins(%src : ...) outs(...)
              {quant_type = #pto<quant_type MXFP8>,
               grpAxis = #pto<mx_group_axis axis0>, interleave = true}
```

`TQuantMxOp` 使用 `hasCustomAssemblyFormat`，新属性走 `attr-dict` 打印路径，
parser/printer 需同步（§12 步骤 1）。

### 6.2 `pto.tmov` 保留 `$fp` 槽位并扩展其合法语义

不新增 op，也不重命名 operand。`TMovOp` 只追加 optional `grpAxis`：

```tablegen
// include/PTO/IR/PTOOps.td, def TMovOp
let arguments = (ins
  PTODpsType:$src,
  PTODpsType:$dst,
  Optional<PTODpsType>:$fp,   // API 名保持；loc=scaling: fp，其他 loc: X-to-ZZ tmp
  Optional<I64>:$preQuantScalar,
  OptionalAttr<PTO_AccToVecModeAttr>:$accToVecMode,
  DefaultValuedAttr<PTO_ReluPreModeAttr,
                    "::mlir::pto::ReluPreMode::NoRelu">:$reluPreMode,
  OptionalAttr<PTO_MxGroupAxisAttr>:$grpAxis
);
```

assembly format、`$fp` operand 的位置、分隔符和 segment size 均保持不变。
`grpAxis` 是 optional attribute：

- `$fp` operand 非-scaling 时，未写 `grpAxis` 表示 axis1；显式 `axis0` / `axis1` 都合法；
- `$fp` operand 不存在或位于 scaling 时，`grpAxis` 必须不存在；
- `preQuantScalar` 与 `$fp` tile operand 继续互斥。

IR 形态：

```mlir
// 既有 FP 形态：$fp.loc = scaling，语义不变
pto.tmov ins(%src : !pto.tile_buf<...>,
             %fp  : !pto.tile_buf<loc=scaling, ...>)
         outs(%dst : !pto.tile_buf<...>)

// ND-to-ZZ：第三个 tile（ODS 名仍为 $fp）的 loc != scaling，grpAxis 缺省为 axis1
pto.tmov ins(%exp : !pto.tile_buf<loc=vec, ...>,
             %tmp : !pto.tile_buf<loc=vec, ...>)
         outs(%exp_zz : !pto.tile_buf<loc=vec, ...>)

// DN-to-ZZ
pto.tmov ins(%exp : !pto.tile_buf<loc=vec, ...>,
             %tmp : !pto.tile_buf<loc=vec, ...>)
         outs(%exp_zz : !pto.tile_buf<loc=vec, ...>)
         {grpAxis = #pto<mx_group_axis axis0>}
```

`tmp` 仍在 `ins`：它会被覆写，但不承载结果语义。`TMovOp` 既有 DPS init 仍只有 `dst`；
`tmp` 与 ND source 的写行为由 §9 的分支 memory effects 表达。

## 7. Verifier 设计

### 7.1 `TQuantMxOp::verify()` 改造

替换 §4.2 引用的那段轴无关校验，改为：

```cpp
const bool isDn = getGrpAxis() == MxGroupAxis::Axis0;

// (0) 语义合法性必须先于任何 shape 推导。
//
// interleave 的 exp 期望形状是按 DN 推出来的；若先做 shape 分派再查这条，
// axis1 + interleave 这种非法 IR 会先撞上"exp shape 不匹配"，报出一条与真正
// 病因无关的诊断，用户还得自己反推。rev2 就是这个顺序，导致 §11.3 的
// interleave_on_nd 用例根本匹配不到预期文案。
if (getInterleave() && !isDn)
  return emitOpError("expects interleave to be used only with grpAxis=axis0");

// MX TQUANT 的 stride/capacity 都由编译期 tile shape 驱动。
// 任一相关 valid/physical 维动态时无法证明实际寻址安全，直接拒绝。
for (auto [name, ty] : {std::pair{"src", srcTy}, std::pair{"dst", dstTy},
                         std::pair{"exp", expTy}, std::pair{"max", maxTy},
                         std::pair{"scaling", scalingTy}}) {
  if (hasDynamicValidOrPhysicalShape(ty))
    return emitOpError() << "expects static valid and physical shapes for " << name
                         << " in MX quantization";
}
if (getExpZz() && hasDynamicValidOrPhysicalShape(expZzTy))
  return emitOpError("expects static valid and physical shapes for exp_zz "
                     "in the deprecated fused MX quantization form");

// (1) 分组轴上的整除约束
if (isDn) {
  if (srcRows % 32 != 0)
    return emitOpError("expects src valid_shape[0] to be a multiple of 32 when grpAxis is axis0");
} else {
  if (srcCols % 32 != 0)
    return emitOpError("expects src valid_shape[1] to be a multiple of 32 when grpAxis is axis1");
}

// (2) 分组 tile 的按轴 shape 契约
//
// 关键：max/scaling 永远是"每组一个标量"的普通分组形状；只有 exp 会被 interleave 改形。
// 因此不能把三者塞进同一个循环用同一份期望值（早期版本的本文档就是这么写的，
// 导致合法的 interleave 输入先在通用检查里被拒 —— 见下方注记）。
//
// axis0 只接受 canonical [M/32, N]。axis1 同时接受：
//   canonical   [M, N/32]
//   legacy flat [1, M*N/32]
// 后者是已存 IR 实际使用的形态，不是待摸底的假设。
const int64_t grpRows = isDn ? srcRows / 32 : srcRows;
const int64_t grpCols = isDn ? srcCols      : srcCols / 32;

auto checkGroupedShape = [&](StringRef name, Type ty) -> LogicalResult {
  SmallVector<int64_t, 2> actual = getValidShapeVec(ty);
  SmallVector<int64_t, 2> canonical = {grpRows, grpCols};
  if (actual == canonical)
    return success();

  if (!isDn) {
    // 乘法使用 checked arithmetic，溢出直接拒绝。
    SmallVector<int64_t, 2> legacyFlat = {1, checkedMul(srcRows, grpCols)};
    if (actual == legacyFlat)
      return success();
    return emitOpError() << "expects " << name
                         << " valid_shape to match canonical [M, N/32] or "
                            "legacy flat [1, M*N/32] for grpAxis=axis1";
  }

  return emitOpError() << "expects " << name
                       << " valid_shape to match canonical [M/32, N] for grpAxis=axis0";
};

// max / scaling：与 interleave 无关，始终走普通分组 shape 契约。
for (auto [name, ty] : {std::pair{"max", maxTy}, std::pair{"scaling", scalingTy}}) {
  if (failed(checkGroupedShape(name, ty))) return failure();
}

// exp：随 interleave 二选一
if (!getInterleave()) {
  if (failed(checkGroupedShape("exp", expTy))) return failure();
} else {
  // 走到这里必然是 DN（(0) 已拦掉 axis1 + interleave），仅接受 [M/64, 2N]。
  SmallVector<int64_t, 2> actual = getValidShapeVec(expTy);
  SmallVector<int64_t, 2> expected = {srcRows / 64, checkedMul(srcCols, 2)};
  if (actual != expected)
    return emitOpError("expects exp valid_shape to match [M/64, 2N] "
                       "for grpAxis=axis0 with interleave=true");
}
```

**兼容性结论已定**：[`test/lit/pto/tquant_mx_a5_emitc.pto`](../../test/lit/pto/tquant_mx_a5_emitc.pto)
与 [`test/lit/pto/quant_mx_tile_native.pto`](../../test/lit/pto/quant_mx_tile_native.pto)
都已使用 `src=16x32`、
`exp/max/scaling valid=1x16`，这正是 legacy flat，而 canonical 是 `16x1`。
因此不再把兼容性留作"实现前摸底"；axis1 双形态是 verifier 的正式契约，
axis0 则始终使用自然二维形状。rev7 进一步固定：这些候选只在所有参与寻址的
valid/physical 维均为静态时比较；无法静态证明 stride/capacity 时拒绝。

#### 7.1.1 physical stride / capacity 矩阵

下表中的 capacity 单位均为**元素数**，所有乘法与 `alignTo` 使用 checked arithmetic。
设 `M = srcValidRows`、`N = srcValidCols`、`Ps = srcPhysicalCols`，并定义
`pack = 1`（MXFP8）或 `2`（MXFP4），则 destination 的列单位分别是一个 FP8 byte 或一个
packed FP4 pair，`dstValidCols = N`（MXFP8）或 `ceilDiv(N,2)`（MXFP4），source-derived
destination stride 为 `Pd = Ps/pack`。MXFP4 还要求 `Ps % 2 == 0`。`physicalRows >= validRows`、
`physicalCols >= validCols`、`dstValidRows == M` 以及 src/dst 自身的二维 footprint
只是所有分支的共同前置条件，不能替代下面的 source-derived stride 检查。

| 分支 | tile | physical 契约 | 原因 |
|---|---|---|---|
| axis1，MXFP8，任意 exp 分支 | `dst` | `dstPhysicalCols == Ps`；capacity ≥ `(M-1)*Ps + N` | F32/B16 2D 实现都按 `row * TileDataSrc::Cols` 写 destination |
| axis1，MXFP4，任意 exp 分支 | `dst` | `Ps % 2 == 0`；`dstPhysicalCols == Ps/2`；capacity ≥ `(M-1)*(Ps/2) + ceilDiv(N,2)` | packed FP4 每字节存两个标量，实际行基址按 `row * (srcPhysicalCols/2)` 计算；axis1 的 N 为 32 倍数，ceil 在此等于 N/2 |
| axis0，MXFP4，任意 interleave | `dst` | `Ps % 2 == 0`；`dstPhysicalCols == Ps/2`；capacity ≥ `(M-1)*(Ps/2) + ceilDiv(N,2)` | DN BF16/FP16 MXFP4 同样使用 source-derived packed byte stride；latest 的 FP16 路径在 `(Ps/2) % 32 != 0` 时越过 dst 借 scratch，该组合另由 unsupported 规则拒绝 |
| axis0，MXFP8，任意 interleave | `dst` | **语义契约仍是自身 `TileDataOut::Cols` + 普通二维 footprint** | 正常 quantized output 使用 destination static cols；latest 的 `interleave && f32` 会借用 dst 暂存 exponent，因此该组合整体拒绝，不把额外 scratch 伪装成 output capacity |
| axis0，任意 interleave | `max` / `scaling` | `physicalCols == srcPhysicalCols`；capacity ≥ `(srcValidRows/32) * srcPhysicalCols` | DN 实现以 `TileDataSrc::Cols` 作为每个 group row 的行步长 |
| axis0，`interleave=false` | `exp` | `physicalCols == srcPhysicalCols`；capacity ≥ `(srcValidRows/32) * srcPhysicalCols` | linear exp 与 max/scaling 使用同一 source-derived stride |
| axis0，`interleave=true` | `exp` | physical 必须精确为 `[srcPhysicalRows/64, align32(2*srcPhysicalCols)]`；valid 为 `[srcValidRows/64, 2*srcValidCols]` | interleaved exp 使用独立、32 列对齐的 physical box |
| axis1，任意 | `max` / `scaling` | valid 区域必须是 allocation 的 compact prefix：`validRows == 1 \|\| physicalCols == validCols`；capacity ≥ `M*(N/32)` | PTO-ISA 无条件把两者 reshape 为 1D，并连续写入前缀 |
| axis1，exp flat branch | `src` + `exp` | **source tight：`Ps == N`**；exp `physicalRows == 1`，valid 必须是 legacy flat `[1, M*N/32]`，capacity ≥ `M*(N/32)` | flat path 用 `M*Ps/32` 决定 exp/max/scaling 的连续写入数；只有 `Ps == N` 才等于语义 group 数 |
| axis1，exp 2D branch | `exp` | `physicalRows > 1`，valid 必须是 canonical `[M, N/32]`；`physicalRows >= M`、`physicalCols >= N/32`；capacity ≥ `(M-1)*physicalCols + N/32` | `TileDataExp::Rows > 1` 选择 2D path，并使用 `TileDataExp::Cols` 作为真实行步长；padded source 必须先匹配此分支，B16 再由下一行拒绝 |
| 任意 f16/bf16 且 `srcValidCols < srcPhysicalCols` | `src` | 普通 source footprint 仍是合法上界；若 `128 % Ps == 0` 且 `(M*Ps) % 128 != 0`，latest 会进入不带 row-tail 的 VL-aligned store，**拒绝** | B16 下 `elemPerVL = 128`；不能要求 IR 额外容纳 `alignTo(M*Ps, 128)`，多出的行不是 source 语义 |
| axis1 canonical 2D 且 src 为 f16/bf16 | 所有可观察输出 | **latest 下 unsupported，拒绝** | MXFP8 会按 group 数借用 `exp` 或覆盖 `max`；MXFP4 始终借用 `max`，无法用 capacity 约束保持可观察输出语义 |
| axis0 FP16 MXFP4 且 `(Ps/2) % 32 != 0` | `dst` | **latest 下 unsupported，拒绝** | 当前实现把 `dst + physicalRows*Ps/2` 当临时区，已经越过声明 footprint |
| axis0 FP32 MXFP8 且 `interleave=true` | `dst` | **latest 下 unsupported，拒绝** | 当前实现把 source-derived-stride 线性 exponent 暂存在 dst；普通 destination footprint 无法证明安全 |

当 `M == 1` 时 canonical 与 legacy flat 数值上相同，`physicalRows == 1` 归入 flat branch；
除此之外，legacy-flat valid shape 配上 `physicalRows > 1` 必须拒绝，不能只因元素总数一致
而进入实际的 2D 分支。

对应 verifier 骨架：

```cpp
auto requireCompactPrefix = [&](StringRef name, Type ty) {
  auto valid = getValidShapeVec(ty);
  auto physical = getPhysicalShapeVec(ty);
  if (valid[0] != 1 && physical[1] != valid[1])
    return emitOpError() << "expects " << name
                         << " valid elements to form a compact physical prefix";
  return success();
};

const int64_t pack = isMxFp4 ? 2 : 1;
const int64_t dstValidCols = isMxFp4 ? ceilDiv(srcCols, 2) : srcCols;
requireValidShape(dstTy, {srcRows, dstValidCols});

if (isDn) {
  if (isMxFp4) {
    require(srcPhysicalCols % 2 == 0);
    requirePhysicalCols("dst", dstTy, srcPhysicalCols / 2);
    requireCapacity(dstTy, (srcRows - 1) * (srcPhysicalCols / 2) + dstValidCols);
  } else {
    requireOrdinary2DFootprint("dst", dstTy);
  }

  requirePhysicalCols("max", maxTy, srcPhysicalCols);
  requirePhysicalCols("scaling", scalingTy, srcPhysicalCols);
  requireCapacity(maxTy, (srcRows / 32) * srcPhysicalCols);
  requireCapacity(scalingTy, (srcRows / 32) * srcPhysicalCols);

  if (!getInterleave()) {
    requirePhysicalCols("exp", expTy, srcPhysicalCols);
    requireCapacity(expTy, (srcRows / 32) * srcPhysicalCols);
  } else {
    require(srcRows % 64 == 0 && srcPhysicalRows % 64 == 0);
    requirePhysicalShape(expTy,
                         {srcPhysicalRows / 64, alignTo(2 * srcPhysicalCols, 32)});
  }
} else {
  require(srcPhysicalCols % pack == 0);
  requirePhysicalCols("dst", dstTy, srcPhysicalCols / pack);
  requireCapacity(dstTy,
                  (srcRows - 1) * (srcPhysicalCols / pack) + dstValidCols);

  requireCompactPrefix("max", maxTy);
  requireCompactPrefix("scaling", scalingTy);
  requireCapacity(maxTy, srcRows * (srcCols / 32));
  requireCapacity(scalingTy, srcRows * (srcCols / 32));

  if (expPhysicalRows == 1) {
    require(srcPhysicalCols == srcCols); // flat group count is M*srcPhysicalCols/32
    requireValidShape(expTy, {1, srcRows * (srcCols / 32)});
    requireCapacity(expTy, srcRows * (srcCols / 32));
  } else {
    requireValidShape(expTy, {srcRows, srcCols / 32});
    require(expPhysicalRows >= srcRows && expPhysicalCols >= srcCols / 32);
    requireCapacity(expTy, (srcRows - 1) * expPhysicalCols + srcCols / 32);
  }
}

```

上述 shape/stride 骨架之后必须追加 latest 支持边界检查：axis1 canonical 2D B16、
axis0 FP16 MXFP4 的非 32B packed stride、axis0 FP32 MXFP8 interleave，以及命中不完整
VL-aligned store 的 padded B16 source 均返回稳定诊断。实现不得为隐藏 scratch 增加
“特殊大 allocation 即合法”分支，也不得把 B16 source capacity 向完整 VL 向上取整。

例如 axis0 的 `src valid=64x16, physical=64x32` 会让辅助输出使用 32 元素行步长；
`exp/max/scaling valid=2x16, physical=2x16` 虽然 valid shape 正确，但第二行从 `+32`
开始，已越过 32 元素 allocation，必须由 `physicalCols == srcPhysicalCols` 拒绝。

axis1 的两个反例也被直接拒绝：

- `src valid=16x32, physical=16x64` + flat `exp/max/scaling=1x16` 在 flat 分支先因
  `srcPhysicalCols != srcValidCols` 失败，不能按 32 个 physical group 写坏 16 元素输出；
- MXFP8 `src valid=16x64, physical=16x128` + `dst physical=16x64` 在 destination
  检查失败；实现按 128 byte 行步长写，合法 destination 必须同样使用 128 列 physical stride。

> **来源标注**：上面这组 interleave 约束（64 对齐、`align32(2N)` physical shape）来自评审
> 指出的 PTO-ISA `include/pto/npu/a5/TQuant.hpp` L3394-L3406。**本文档基线快照
> `7af803bc`（`TQuant.hpp` 共 3106 行）里根本没有 `bool interleave` 这个 overload**，
> `grep "bool interleave"` 零命中，故本节无法在该快照上自证。rev11 已在目标 latest pin
> `69a81f3/40e741b` 上复核；这也直接决定了 interleave 实现前必须先完成 commit 0 的 pin bump。

`exp_zz` 互斥（§5.4）：

```cpp
if (getExpZz() && isDn)
  return emitOpError("expects the deprecated exp_zz form to use grpAxis=axis1; "
                     "use pto.tmov with a non-scaling tmp for axis0 exponents");
```

### 7.2 `TMovOp::verify()` 的 X-to-ZZ 分支

先以第三个 tile 的地址空间执行唯一一次形态分类，再分派到既有 verifier 或 X-to-ZZ
verifier。分类结果必须由 verifier、memory effects、EmitC 和相关 pass 共享：

```cpp
TMovForm form = classifyTMovForm(getFp());
// getFp() absent        -> NoTileAux（既有普通/preQuantScalar 形态）
// getFp().loc=scaling  -> Fp
// getFp().loc!=scaling -> XToZz

if (form != TMovForm::XToZz) {
  if (getGrpAxisAttr())
    return emitOpError("expects grpAxis only on the X-to-ZZ form with a non-scaling third tile");
  return verifyExistingTMovForm(form);
}
```

地址空间缺失时不能猜测，直接拒绝。进入 `XToZz` 后，按 §3.2 的 pto-isa 约束逐条
落地；该形态 A5-only，通过 `dispatchVerifierByArch` 在 A2/A3 上直接拒绝。

| # | 规则 | 诊断文案（草案） |
|---|---|---|
| 1 | 第三个 tile 必须带显式地址空间；scaling 归类为 FP，非-scaling 归类为 X-to-ZZ | `expects the third tile to have an explicit address space` |
| 2 | X-to-ZZ 禁止 `preQuantScalar`、`accToVecMode` 与非默认 `reluPreMode`；`grpAxis` 只允许用于 X-to-ZZ | `expects the X-to-ZZ tmov form not to use preQuantScalar, accToVecMode, or reluPreMode` |
| 3 | X-to-ZZ 的 `src`/`dst`/`tmp` 均为 vec tile | `expects X-to-ZZ src/dst/tmp to be vec tiles` |
| 4 | 三者元素类型相同 | `expects src, dst, and tmp to share one element type` |
| 5 | 元素类型 ∈ {`ui8`, `!pto.hif8`, `!pto.f8E8M0`}；**`i8` 必须拒绝** | `expects element type to be one of ui8, !pto.hif8, !pto.f8E8M0 (i8 lowers to int8_t, which PTO-ISA CommonCheckZZ rejects)` |
| 6 | `src` 为 `row_major` + `none_box` | `expects src to use blayout=row_major, slayout=none_box` |
| 7 | `dst` 为 `row_major` + `slayout=row_major` | `expects dst to use blayout=row_major, slayout=row_major (ZZ box)` |
| 8 | rank-2 valid shape | `expects rank-2 valid_shape for src/dst/tmp` |
| 9 | ND：`dstCols % 2 == 0`。**不检查行对齐** | `expects ND-to-ZZ dst valid_shape[1] (the grouped exponent column count) to be even` |
| 10 | DN：`srcRows % 2 == 0 && srcRows >= 2`。**`srcRows == 1` 必须拒绝** | `expects DN-to-ZZ src valid_shape[0] to be an even count >= 2; a single row-group produces no output in PTO-ISA` |
| 11 | DN：`srcCols % 16 == 0` | `expects DN-to-ZZ src valid_shape[1] to be a multiple of 16` |
| 12 | ND：`tmp` 物理容量 ≥ `64 + ceil(dstRows / 16) * dstCols` 字节（§3.2） | `expects tmp to provide at least <N> bytes for ND-to-ZZ (64 + ceil(dst rows / 16) * dst cols)` |
| 13 | src/dst 元素总数相等 | `expects src and dst to hold the same exponent count` |
| 14 | 用于 stride/capacity、local allocation 与 C++ tile 类型化的 shape 必须静态：ND、DN 都要求 `src`/`dst` valid 与 physical、`tmp` physical 静态。DN 的 ISA 虽不访问 `tmp`，PTOAS 仍要分配并以静态 `Tile<...>` 传给 `TMOV<0>` | `expects static valid and physical shapes for src/dst and a static tmp physical shape for X-to-ZZ` |
| 15 | ND：source valid 元素必须是 allocation 的 compact prefix：`srcValidRows == 1 \|\| srcPhysicalCols == srcValidCols` | `expects ND-to-ZZ src valid elements to form a compact prefix (single-row legacy flat or physical row stride equal to valid cols)` |
| 16 | ND：`src` physical capacity ≥ `align16(dstRows) * dstCols` 字节 | `expects ND-to-ZZ src physical capacity to cover align16(dst rows) * dst cols because source padding is zeroed in place` |
| 17 | ND：`dst` physical capacity ≥ `align16(dstRows) * dstCols` 字节 | `expects ND-to-ZZ dst physical capacity to cover align16(dst rows) * dst cols` |
| 18 | DN：source physical row stride 必须等于传入 ISA 的 `srcValidCols`，即 `srcPhysicalCols == srcValidCols` | `expects DN-to-ZZ src physical row stride to equal src valid_shape[1]` |
| 19 | DN：`src` / `dst` physical capacity 均 ≥ `srcRows * srcCols` 字节 | `expects DN-to-ZZ src/dst physical capacity to cover src valid rows * src valid cols` |

规则 9–19 里用来驱动转换的 shape 按轴不同（ND 看 `dst`，DN 看 `src`），
实现时**必须按轴选择被检查的 tile**，这一点在 §3.2 已单独标注。
所有容量乘法和 `align16` 都要使用 checked arithmetic，不得让 `int64_t` 溢出后绕过校验。

**规则 9 的坐标系**（早期版本写错，必须注意）：`src`/`dst` 在这个 op 里已经是 **exponent
tile**，它的列数是原矩阵的 `N / 32`。ISA 侧的真实约束来自 `GenerateB8IndicesZZToUB` 里
`P = groupedCols / 2` 这个整除——要求 **exponent 列数为偶数**，换算回原矩阵才是 `N % 64 == 0`。
早期版本直接对 exponent 列数写 `% 64 == 0`，等于把约束放大了 32 倍，**会拒掉本文档自己
列出的每一个 ND 用例**：§11.1 的 `nd_fp8_ocp`（src `16x64` → exp `16x2`，`2 % 64 != 0`）、
§11.6 的 `nd_fp8_f32_ocp_16x128`（exp `16x4`，`4 % 64 != 0`）无一幸免。

**规则 9 不要求 valid 行对齐**：`rowBlockCount = (rows + 15) / 16` 是显式的 ceil-divide，注释写着
"to support non-16-aligned row counts"，并且 `ZeroSourcePaddingB16` 会把 `validRow` 之外、
补到 16 对齐为止的那段源数据清零，避免 `vgather2` 读到 UB 里的残留。所以非 16 对齐行数是
**ISA 支持的**。但"支持非对齐 valid 行"不等于可以不分配 padding：
规则 12/16/17 必须同时按 `ceil` / `align16` 计入 tmp、source 和 destination 的额外空间。

**规则 10 为什么必须拒绝 `srcRows == 1`**：`GenerateB8IndicesDN2ZZToUB` 里
`numPairs = hatM / 2`，主循环是 `for (p = 0; p < numPairs; ++p)`。`hatM == 1` 时
`numPairs == 0`，循环一次都不执行，**`dst` 完全不被写入**。早期版本把 `M = 32`（即
`hatM = 1`）当作"退化 identity"放行，只有在 `src`/`dst` 地址别名时才碰巧成立；
对独立分配的 `dst` 就是读未初始化内存。若确实需要支持 `M = 32`，应当在 lowering 阶段
显式降成一次 copy（或证明别名），而不是生成 `TMOV<0>`。§11.6 早期原计划的
`m=32` **完整链路** ST 用例相应改为 TMov **负向 lit**；带不完整 B16 VL-tail 的
TQUANT 形态在 latest pin 下也由 §7.1.1 拒绝，只保留 §11.6.1 的原生行为审计。

规则 12 只对 ND 施加。DN 侧不做 `tmp` **最小容量**校验，理由见 §9 与 §13-1：
当前 ISA 实现里 DN 的 `tmp` 是 `(void)tmp;`，**根本不使用**。但这不意味着 DN
`tmp` 可以有动态 physical shape：它仍是一个 local vec tile operand，PlanMemory 需要静态
physical shape 计算 slot bytes，EmitC 也会把 physical rows/cols 直接写进 `Tile<...>` 模板参数。
动态 physical 维会导致无法规划 local allocation，并生成 `Tile<..., -1, ...>`，在调用
`TMOV<0>` 前就已经不可落地。本次选择直接拒绝，不设计额外的静态 dummy-tile lowering。

**dynamic shape 策略**：本设计选择"无法静态证明就拒绝"。不仅 physical shape
必须静态，驱动转换和元素数的 `src`/`dst` valid shape 也必须静态；否则对齐、
奇偶性、compact stride 与 capacity 中至少一项无法证明。本次不在 lowering 中
隐式插入 compact copy/padded scratch。后者会引入新 allocation、copy 和同步语义，也会
改变本 op 直接对应一次 ISA TMOV 的成本模型。未来如需支持动态 shape，应单独
设计显式 compact/pad op 或可证明的 lowering，不得在本 verifier 中静默放行。

## 8. EmitC 设计

### 8.1 `pto.tquant.mx`

`PTOQuantMxToEmitC` 按形态二分：

**形态 A（deprecated fused，`exp_zz` 存在）** — 完全保持现状，不受本次改动影响：

```cpp
TQUANT<QuantType, VecStoreMode, DstT, SrcT, ExpT, MaxT, ScalingT>(
    dst, src, &exp, &max, &scaling, &exp_zz);
```

**形态 B（grouped，本次新增；`exp_zz` 不存在）**：

```cpp
TQUANT<grp_axis, MxQuantAlg[, interleave]>(dst, src, &exp, &max, &scaling);
```

- 第一个模板参数是整数字面量 `0` / `1`；
- 第二个是 §5.1 表格合成的 `MxQuantAlg` token；
- 第三个仅在 `interleave = true` 时出现（对应 pto-isa 的 bool overload）；
- **tile 类型不再写进模板参数**，由实参推导。

生成样例：

```cpp
TQUANT<1, MxQuantAlg::OcpMxFp8E4M3>(v_dst, v_src, &v_exp, &v_max, &v_scaling);
TQUANT<0, MxQuantAlg::OcpMxFp8E4M3>(v_dst, v_src, &v_exp, &v_max, &v_scaling);
TQUANT<0, MxQuantAlg::OcpMxFp8E4M3, true>(v_dst, v_src, &v_exp, &v_max, &v_scaling);
```

**已拍板：统一到形态 B。** 所有不带 `exp_zz` 的 `pto.tquant.mx`，包括不写
`grpAxis` 的现有 ND IR，都生成 `<grp_axis, MxQuantAlg[, interleave]>`。不根据
`grpAxis` / `interleave` 是否在文本中显式写出而保留另一条旧类型列表路径。

现有 ND IR 的 C++ 文本会从类型列表模板参数变为
`<1, MxQuantAlg::...>`，但 PTO IR 语义不变。受影响的 lit golden 在 §11.1 与
§12 提交 3 统一更新。这个决策避免为同一 grouped ND 语义长期维护两种 EmitC
文本。deprecated `exp_zz` fused 形态仍暂时走形态 A，因为它的 operand 与 pto-isa
overload 本来就与 grouped 形态不同；这不是为同一语义保留双形态。

### 8.2 `pto.tmov` 的 X-to-ZZ 分支

`PTOMovToEmitC` 复用 §7.2 的形态分类。无第三个 tile 与 scaling `fp` 继续走现有
lowering；只有第三个 tile 为非-scaling 时进入以下 X-to-ZZ lowering：

```cpp
// grpAxis = axis1（默认）
TMOV(dst, src, tmp);
// grpAxis = axis0
TMOV<0>(dst, src, tmp);
```

`axis1` **不写模板参数**（依赖 pto-isa 的 `grp_axis = 1` 默认值），与 issue 期望一致；
`axis0` 写 `<0>`。tile 类型全部由实参推导，不进模板参数列表。

lowering 不得仅以“存在第三个 tile”判断 FP，也不得仅以 `grpAxis` 是否显式出现判断
X-to-ZZ；唯一依据是 `classifyTMovForm(getFp())`。这样默认 axis1 的 X-to-ZZ 即使不打印
`grpAxis`，仍会稳定生成三参数 `TMOV(dst, src, tmp)`。

## 9. 内存效应、内存规划与 liveness

`TMovOp::getEffects()` 先调用 `classifyTMovForm(getFp())`。`NoTileAux` 与 `Fp` 完全保持既有效应；
`XToZz` 再按 `grpAxis` 分派：

| operand | `grpAxis = axis1`（ND） | `grpAxis = axis0`（DN） |
|---|---|---|
| `src` | `Read` + `Write` | `Read` |
| `dst` | `Write` | `Write` |
| `tmp` | `Read` + `Write` | **无效应** |

**ND 的 `src` 必须声明读写**。`GenerateB8IndicesZZToUB` 会调用
`ZeroSourcePaddingB16`，把 `[validRows, align16(validRows))` 对应的 compact source
padding 原地清零。`m = 20` 这类非 16 对齐用例一定发生写操作，不能把它当成
只是 ISA 内部不可观察的细节。若只标 `Read`：

- PlanMemory / alias 分析会在 source 的后续使用和 buffer 复用判定中漏掉这次修改；
- InsertSync 和依赖分析会丢掉对 source 的 write-after-read / read-after-write 边；
- 后续再读取同一 source allocation 的 op 可能被错误重排。

`src` 语法上仍放在 `ins`，因为它不承载结果语义；是否写内存由
`MemoryEffectsOpInterface` 表达，不能从 DPS `ins`/`outs` 位置推断。

**ND 的 `tmp` 必须同时声明读写**，否则：

- PlanMemory 可能把 `tmp` 与仍然活跃的 buffer 复用；
- CSE / DCE 可能认为写 `tmp` 无副作用而删除或合并；
- InsertSync 的宏模型拿不到该 tile 的依赖边。

**DN 的 `tmp` 按轴区分、不声明效应**（早期版本对两个轴一律声明 Read+Write）。ISA 侧
DN 路径是 `(void)tmp;`，一个字节都不碰（§3.2）。若仍统一声明 Read+Write，会凭空拉长该
buffer 的 liveness、阻止 PlanMemory 复用它，在 UB 吃紧的 kernel 里是实打实的浪费。

`grpAxis` 是编译期常量属性，因此 `getEffects()` 可以直接按它分支，不存在动态不确定性。

> **代价与缓解**：这条建模绑定了"当前 pin 的 DN 实现不使用 `tmp`"这个事实。若未来某个
> pto-isa 版本让 DN 也用上 `tmp`，效应声明就会漏报，症状是 PlanMemory 复用后的静默数据
> 损坏。缓解：在 §12 的提交里附一条注释指向 `GenerateB8IndicesDN2ZZToUB` 的 `(void)tmp;`，
> 并把"复核 DN 是否仍不使用 tmp"写进 pin bump 的检查清单（§13-1）。
> 保守起见也可以只声明 `Write` 不声明 `Read`——但既然一个字节都不碰，那同样是虚构的效应。

scaling `fp` 形态仍是 `src: Read`、`fp: Read`、`dst: Write`；普通二参数和
`preQuantScalar` 形态的效应也不变。

### 9.1 `TQuantMxOp` 的 source mutation

`TQuantMxOp` 不能继续无条件把 `src` 声明为只读。目标 PTO-ISA 的 ND 和 DN 路径都会调用
`ZeroPadSourceTile<T, TileDataSrc::Cols>`；当 source 是 f16/bf16 且
`srcValidCols < srcPhysicalCols` 时，它把每个 valid row 的
`[validCols, physicalCols)` padding **原地写零**。f32 通过 `if constexpr` 跳过该写入。

因此 `TQuantMxOp::getEffects()` 按静态 tile type/shape 建模：

| source 条件 | `src` effect |
|---|---|
| f16/bf16 且 `validCols < physicalCols` | `Read + Write` |
| f32，或 f16/bf16 且 `validCols == physicalCols` | `Read` |

§7.1 已拒绝相关动态 valid/physical shape，所以 `getEffects()` 不需要在“不知道是否有
padding”的情况下猜测。`dst`、`exp`、`max`、`scaling`（以及存在时的 `exp_zz`）继续为
`Write`。

该写效应与 X-to-ZZ source mutation 同等可观察：PlanMemory、InsertSync、alias/liveness
必须在 TQuant 后再次读取同一 source allocation 时保留 RAW 依赖，不能跨越 TQuant 复用
或重排 buffer。§11.2 增加 f16/bf16 padding source 的“量化后再次读取”回归；另保留
f32 与无 padding f16 的对照，防止把所有 source 无条件扩大为 `Read + Write`。

`Read + Write` 只授权修改 source **声明 footprint 内**的行尾 padding，不授权
`ZeroPadColumns_VLAligned` 在最后一个 VL 写不存在的额外 rows。该边界由 §7.1.1 的
latest 支持边界的 verifier 拒绝规则保证，§11.6.1 的 source 精确 redzone 只保存底层行为证据；
不通过扩大 `MemoryEffects` 或 source capacity 规避。

X-to-ZZ 的 src/dst 都是 vec tile，现有 `TMovOp::getPipe()` 会返回 `PIPE_V`；无需新增
pipe 实现，但 InsertSync 必须使用上面的分支 effects。

## 10. 需要同步的层

按 `.claude/rules/cross-layer-sync.md` 逐层列出：

| 层 | 改动 |
|---|---|
| ODS | `PTOAttrs.td` 新增 `MxGroupAxis`；`PTOOps.td` 给 `TQuantMxOp` 加两个属性，给 `TMovOp` 增加 optional `grpAxis`；保留 `$fp` operand 名和生成 API |
| IR / verifier | `PTO.cpp`：`TQuantMxOp::verify()` 按轴及 physical 契约分派；`TMovOp::verify()` 以 `classifyTMovForm(getFp())` 区分 `NoTileAux` / `Fp` / `XToZz`，后者进入 §7.2；`TQuantMxOp` 自定义 parser/printer 同步新属性 |
| EmitC | `PTOToEmitC.cpp`：`PTOQuantMxToEmitC` 二分；扩展既有 `PTOMovToEmitC`，`getFp()` 为非-scaling tile 时生成三参数 X-to-ZZ `TMOV` |
| CAPI | `include/pto-c/Dialect/PTO.h` + `lib/CAPI/Dialect/PTO.cpp`：新枚举的 C 入口（参照 `QuantScaleAlg` 现有写法） |
| Python binding | `lib/Bindings/Python/PTOModule.cpp` 暴露 `MxGroupAxis`；生成的 `TMovOp(..., fp=...)` keyword 保持不变，只追加 optional `grpAxis=`；不新增 op binding |
| PTO-BC | 不新增 opcode。scaling `getFp()` 保持 #1122 的 legacy FP wire opcode；非-scaling `getFp()` 必须走 generic v0 兼容记录并保留 `grpAxis`，不能复用 legacy FP payload。`pto.tquant.mx` 在基线 v0 known-op 表中不存在，必须新增显式 generic compatibility shim，使其不依赖 `PTOBC_ALLOW_GENERIC` 就编码为 `kOpcodeGeneric`；不得伪造 known-op schema 或占用 `pto.tquant` 的 opcode |
| TMov 消费者 | 审计 `PTORemoveIdentityTMov`、`PTOA5NormalizeTMov`、PlanMemory、InsertSync、InferMemScope 和所有 `getFp()` 使用点；保留 `getFp()` API，语义相关消费者统一调用 `classifyTMovForm(getFp())` |
| Memory effects | `TQuantMxOp::getEffects()` 按 f16/bf16 source padding 分支为 `Read + Write`；`TMovOp::getEffects()` 按 FP/X-to-ZZ 与 axis 分支；PlanMemory/InsertSync 增加 source mutation 回归 |
| 文档 | `docs/PTO_IR_manual.md`（两个 op 章节）、`docs/release/PTO-tile-Instruction-SPEC-v0.4.md`、本设计文档 |
| 测试 | 见 §11 |
| **pto-isa pin** | 直接对齐最新发布头：GitCode `master@69a81f3` 用于 CI、主 Dockerfile 与 remote-validation；GitHub `main@40e741b` 用于 `ci_sim`。GitHub commit 是 PR #239 的 merge，内容同步自 GitCode `69a81f3`。每个目标仍建模为 `(repo, revision, 兼容性约束)` 并在自己的 remote 验证。`Dockerfile.dev@662d7f2` 继续作为 CANN 9.0 独立目标，仅在 latest 通过该环境验证后更新。原 updater 已删除，本 PR 重新落地 repo-aware target map/updater；不引入未来占位 revision，也不要求 pto-isa 先合其他修改 |
| ReleaseNotes | 记录 `pto.tmov` 新增非-scaling `tmp` 的 X-to-ZZ 形态，以及 `grpAxis`/`interleave`；标注 `exp_zz`/`storeMode` deprecated；说明 axis1 f32 canonical / tight-source legacy flat 受支持、两个 tight MXFP8 legacy-flat case 无需迁移，B16 canonical 及 §7.1.1 另外三类 latest 限制会被拒绝；明确 MXFP4 destination physical cols 收紧为 source physical cols / 2，既有 `tquant_mx_a5_emitc.pto` 要迁移；另记录 §8.1 的生成文本变化 |

**PTO-BC 注意事项**：这里必须区分“已有 known-op schema”和“generic record”。
PTOAS `988d50e24` 的 `tools/ptobc/generated/ptobc_opcodes_v0.h` 只登记了
`pto.tquant`（`0x1047`），**没有** `pto.tquant.mx`。当前 encoder 对未登记 op 的缺省行为是：
只有设置 `PTOBC_ALLOW_GENERIC` 才允许 generic，否则报
`op is not in v0 opcode table`。所以不能要求保持一个不存在的 TQuantMx schema，也不能让
`pto.tquant.mx` 借用 `pto.tquant` 的 fixed payload。

实现应在 `shouldEncodeViaGenericV0CompatibilityShim()` 中显式识别 `TQuantMxOp` 并返回
`true`，从而无论是否设置 `PTOBC_ALLOW_GENERIC` 都编码为现有 `kOpcodeGeneric (0xFFFF)`；
generic record 完整携带 op 名、五个 operand 与 `grpAxis` / `interleave` 属性。另一方面，
普通 `pto.tmov` 与 scaling-FP legacy wire 的既有 fixed schema 不改；`pto.tmov` 的编码分支
从“`getFp()` 是否存在”收紧为 `classifyTMovForm(getFp())`，只有 `Fp` 可选择
`kTMovFpWireOpcode`，`XToZz` 必须经 generic v0 record 编码完整 operand/attribute。
schema 守卫应钉住真正存在的 `pto.tquant` / 普通 `pto.tmov` / scaling-FP legacy wire，
并明确断言 known-op 表中没有 `pto.tquant.mx`；roundtrip 的二进制解析再分别证明
TQuantMx 和 X-to-ZZ 都选择 generic，而不是依赖环境变量后只 grep 解码文本。

## 11. 测试方案

### 11.1 lit 正向：`test/lit/pto/tquant_mx_grp_axis_emitc.pto`

```
// RUN: ptoas --pto-arch=a5 --pto-level=level3 %s | FileCheck %s --check-prefix=EMITC
// RUN: not ptoas --pto-arch=a3 %s 2>&1 | FileCheck %s --check-prefix=A3-REJECT
```

用例矩阵（同一文件多个 `func.func`）：

| 函数 | `grpAxis` | alg | valid shape | physical 关键契约 | 期望模板参数 |
|---|---|---|---|---|---|
| `nd_fp8_ocp` | 默认 | MXFP8 × OCP | src f32/dst `16x64`；exp/max/scaling `16x2` | src/dst physical cols 均为 `64`；exp/max/scaling 为 `2/2/2`，max/scaling compact | `TQUANT<1, MxQuantAlg::OcpMxFp8E4M3>` |
| `nd_exp_2d_strided` | axis1 | MXFP8 × OCP | src f32/dst `16x64`；exp/max/scaling `16x2` | src/dst physical cols 均为 `128`；exp physical `16x32`，证明 padded source 走 2D exp 且 destination 匹配 source-derived stride；max/scaling 仍 compact | `TQUANT<1, MxQuantAlg::OcpMxFp8E4M3>` |
| `nd_fp8_nv` | axis1 | MXFP8 × NV | src f32 `16x64`；exp/max/scaling `16x2` | 同 `nd_fp8_ocp` | `TQUANT<1, MxQuantAlg::NvMxFp8E4M3>` |
| `nd_fp8_legacy_flat` | 默认 | MXFP8 × OCP | src/dst `16x32`；exp/max/scaling `1x16` | src/dst physical cols 均为 `32`（flat source 必须 tight）；exp physical rows 为 `1`；三个辅助输出都是 compact prefix | `TQUANT<1, MxQuantAlg::OcpMxFp8E4M3>` |
| `dn_fp8_ocp` | axis0 | MXFP8 × OCP | src/dst `64x16`；exp/max/scaling `2x16` | src physical `64x32`；dst 可为 `64x16`；exp/max/scaling 均为 `2x32` | `TQUANT<0, MxQuantAlg::OcpMxFp8E4M3>` |
| `dn_fp4_ocp` | axis0 | MXFP4_E2M1 × OCP | src bf16 `64x16`；dst packed `64x8`；exp/max/scaling `2x16` | src physical cols `32`；dst packed physical cols `16`；三个辅助输出均为 `2x32` | `TQUANT<0, MxQuantAlg::OcpMxFp4E2M1>` |
| `dn_fp8_interleave` | axis0 | MXFP8 × OCP | src bf16/dst `128x16`；exp `2x32`；max/scaling `4x16` | src physical `128x32`，总 physical 元素数整除 B16 VL；dst 可为 `128x16`；exp `2x64`；max/scaling `4x32` | `TQUANT<0, MxQuantAlg::OcpMxFp8E4M3, true>` |

CHECK 要点：
- 断言**完整的模板参数串**，而不只是 `TQUANT(`，否则 axis/alg 写错测不出来；
- 加 `// EMITC-NOT: VecStoreMode`，确认新路径不再走 fused 形态；
- `A3-REJECT` 断言 `tquant.mx is only supported on A5`；
- `dn_fp8_interleave` 这一行同时钉住了 §7.1 的分派：**exp 走 interleave 形状、
  max/scaling 走普通分组形状**。早期版本的 verifier 把三者一起按 `4x16` 检查，
  这个合法用例会先在通用检查里被拒——它是那个 bug 的回归哨兵；
- `nd_fp8_legacy_flat` 必须保留精确的 `1x16` valid shape，它与现有
  `tquant_mx_a5_emitc.pto` / `quant_mx_tile_native.pto` 共同固定 axis1 legacy flat
  兼容契约；同时 physical rows 必须为 `1` 且 source tight，确保实际命中安全的 flat branch；
- canonical B16 2D、FP32 interleave、FP16 MXFP4 非 32B packed stride 与不完整 B16
  VL-tail 不在本正向表中；它们由 §11.3 的 latest 支持边界负向用例拒绝。§11.6.1 仅保留
  原生行为审计，不能拿经过 256B PlanMemory slot 的普通 canary 掩盖实际越界；
- 实施 §7.1.1 的 MXFP4 packed dst 规则时，同提交把现有
  `test/lit/pto/tquant_mx_a5_emitc.pto` 的 `%dst_fp4` physical cols 从 `32` 改为 `16`
  （`v_col=16` 不变），并同步 `outs` type。该用例不是兼容哨兵；两个保持不迁移的
  legacy-flat 兼容哨兵仅指该文件的 MXFP8 case 与 `quant_mx_tile_native.pto` 的 MXFP8 case。

### 11.2 lit 正向：`test/lit/pto/tmov_x2zz_emitc.pto`

所有 X-to-ZZ case 都写成 `pto.tmov ins(%src, %tmp) outs(%dst)`，其中 `%tmp` 为
非-scaling vec tile；不出现独立 op 名。

| 函数 | 第三个 tile / `grpAxis` | shape / physical 要点 | 期望生成 |
|---|---|---|---|
| `nd_to_zz` | 非-scaling tmp / 默认 | canonical source，`valid=16x4, physical=16x4` | `TMOV({{.*}}, {{.*}}, {{.*}});` 且 `CHECK-NOT: TMOV<` |
| `nd_to_zz_explicit_axis1` | 非-scaling tmp / axis1 | 与 `nd_to_zz` 相同 | 与默认 axis1 生成文本完全相同 |
| `nd_to_zz_tail` | 非-scaling tmp / 默认 | `src/dst valid=20x4`，source 紧密，两者 capacity 至少 `32x4` | 同上，覆盖 padded rows |
| `nd_to_zz_legacy_flat` | 非-scaling tmp / 默认 | `src valid=1x80, physical=1x128`，`dst valid=20x4, physical=32x4` | 同上，覆盖 compact-prefix 单行特例 |
| `dn_to_zz` | 非-scaling tmp / axis0 | source physical cols 等于 valid cols；unused tmp 仍给出静态 physical shape | `TMOV<0>({{.*}}, {{.*}}, {{.*}});` |
| `nd_to_zz_hif8` | 非-scaling tmp / 默认 | `!pto.hif8`，满足同样的 compact/capacity 契约 | 无模板参数 `TMOV(...)` |
| `scaling_fp_stays_fp` | scaling fp / 无 `grpAxis` | 沿用既有合法 FP tile 组合 | 与修改前完全相同的 `TMOV_FP` / FP `TMOV` 文本 |
| `plain_tmov_unchanged` | 无第三个 tile | 沿用既有 vec-to-vec copy | `TMOV(dst, src)`，不进入 X-to-ZZ |

内存效应另加可机读的 interface 回归用例，精确断言：

- ND：`src = Read + Write`、`dst = Write`、`tmp = Read + Write`；
- DN：`src = Read`、`dst = Write`、`tmp = none`。
- FP：`src = Read`、`dst = Write`、`fp = Read`，证明 scaling 分类没有被 X-to-ZZ 改写。

同文件再追加 `--mlir-print-ir-after=pto-plan-memory` 和 InsertSync 的 RUN，构造 ND
X-to-ZZ 前后都访问 source 的序列，断言内存规划不做跨越该 source mutation 的
非法复用，且 source write 与后续 read 之间的依赖/同步不会丢失。只检查 `tmp` 不足以覆盖本次的
source-mutating 修正。

另建 `test/lit/pto/tquant_mx_source_effects.pto`，覆盖 §9.1：

- `dn_f16_padded_src`：src valid `64x16`、physical `64x32`，TQuant 后再次读取同一
  source，断言 `src = Read + Write` 且 PlanMemory/InsertSync 保留依赖；
- `dn_bf16_padded_src_vl_complete`：选择 `M*physicalCols` 整除 128 的 axis0 shape，断言
  `src = Read + Write`；不完整 VL 由 §11.3 拒绝；
- `dn_f32_padded_src_control`：同 shape 的 f32 source 只声明 `Read`；
- `dn_f16_tight_src_control`：f16 但 `validCols == physicalCols` 时只声明 `Read`。

这组测试必须检查可机读的 `MemoryEffectsOpInterface`，并构造“padding 后再次读取 source”
的实际 pass 序列；只检查生成 C++ 文本不能证明依赖边正确。

另加 Python binding/API 兼容 smoke test：现有
`pto.TMovOp(..., fp=fp_scaling)` 调用原样构造 FP 形态，C++ 编译用例继续调用 `getFp()`；
新增 X-to-ZZ 也仍通过 `fp=tmp` 传入第三个 tile。测试中禁止出现 `aux=` / `getAux()`，
从生成 API 层钉住“只扩展 `$fp` 的语义，不改公开 operand 名”。

### 11.3 lit 负向：`test/lit/pto/tquant_mx_grp_axis_invalid.pto`

用 `split-file` 拆分，逐条覆盖 §7.1：

| 子用例 | 期望诊断 |
|---|---|
| `dn_rows_not_aligned` | `expects src valid_shape[0] to be a multiple of 32 when grpAxis is axis0` |
| `dn_wrong_exp_shape`（给 `[M, N/32]`） | `expects exp valid_shape to match canonical [M/32, N] for grpAxis=axis0` |
| `nd_shape_matches_neither`（既不是 `[M, N/32]` 也不是 `[1, M*N/32]`） | `expects exp valid_shape to match canonical ... or legacy flat ...` |
| `interleave_on_nd`（axis1 + interleave） | `expects interleave to be used only with grpAxis=axis0`，**且不得先报成 shape 不匹配**（见 §7.1 的检查顺序） |
| `interleave_exp_valid_rows_mismatch` | `expects exp valid_shape to match [M/64, 2N] for grpAxis=axis0 with interleave=true` |
| `interleave_exp_valid_cols_mismatch` | 同上，用独立 case 证明列维错误也被拒绝 |
| `interleave_valid_rows_not_multiple_of_64` | `expects src valid rows to be a multiple of 64 when interleave is true` |
| `interleave_physical_rows_not_multiple_of_64` | `expects src physical rows to be a multiple of 64 when interleave is true` |
| `interleave_exp_physical_rows_mismatch` | `expects interleaved exp physical rows to be src physical rows / 64` |
| `interleave_exp_physical_cols_mismatch` | `expects interleaved exp physical cols to be align32(2 * src physical cols)` |
| `interleave_max_shape_follows_grouping`（max 按 `[M/64, 2N]` 给，应被拒） | `expects max valid_shape to match canonical [M/32, N] for grpAxis=axis0`——钉住 max/scaling **不**随 interleave 改形 |
| `dn_exp_stride_mismatch`（非 interleave） | `expects exp physical cols to equal src physical cols for grpAxis=axis0` |
| `dn_max_stride_mismatch` | `expects max physical cols to equal src physical cols for grpAxis=axis0` |
| `dn_scaling_stride_mismatch` | `expects scaling physical cols to equal src physical cols for grpAxis=axis0` |
| `interleave_max_stride_mismatch` | 同样拒绝 max；证明 interleave 只改变 exp，不改变 max 的 source-derived stride |
| `interleave_scaling_stride_mismatch` | 同样拒绝 scaling |
| `nd_max_not_compact` | `expects max valid elements to form a compact physical prefix` |
| `nd_scaling_not_compact` | `expects scaling valid elements to form a compact physical prefix` |
| `nd_legacy_exp_selects_2d`（valid 是 `[1, M*N/32]`，但 physical rows > 1） | `expects legacy flat exp to use physical rows == 1`；证明分支依据与 `TileDataExp::Rows` 一致 |
| `nd_flat_exp_with_padded_src`（src valid `16x32`、physical `16x64`，aux flat `1x16`） | `expects axis1 flat exp to use a tight source with physical cols equal to valid cols` |
| `nd_fp8_dst_stride_mismatch`（src physical cols `128`、dst physical cols `64`） | `expects MXFP8 axis1 dst physical cols to equal src physical cols` |
| `nd_fp4_dst_packed_stride_mismatch`（src physical cols `128`、dst packed physical cols `32`） | `expects MXFP4 axis1 dst physical cols to equal src physical cols / 2` |
| `dn_fp4_dst_packed_stride_mismatch` | `expects MXFP4 axis0 dst physical cols to equal src physical cols / 2` |
| `mxfp4_odd_src_physical_cols` | `expects MXFP4 src physical cols to be even for packed destination addressing` |
| `nd_canonical_b16_latest_unsupported` | `does not support axis1 canonical 2D B16 quantization with the pinned pto-isa revision` |
| `dn_fp4_fp16_unaligned_packed_stride_latest_unsupported` | `does not support FP16 MXFP4 axis0 when packed source stride is not a multiple of 32 bytes` |
| `dn_fp8_f32_interleave_latest_unsupported` | `does not support FP32 interleave with the pinned pto-isa revision` |
| `b16_padding_incomplete_vl_latest_unsupported` | `does not support padded B16 source whose VL-aligned padding store has an incomplete final VL` |
| `tquant_dynamic_valid` | `expects static valid and physical shapes for <tile> in MX quantization` |
| `tquant_dynamic_physical` | 同上；至少分别覆盖 axis0 source-derived stride 与 axis1 compact-prefix 两条路径 |
| `exp_zz_with_axis0` | `expects the deprecated exp_zz form to use grpAxis=axis1` |

rev2 这里只有一条笼统的 `interleave_wrong_exp_shape`，证明不了 §7.1 的四条 interleave
约束各自生效，与 §14"每条 verifier 规则都有负向用例"的验收标准不符。上表按分支拆开，
`valid` / `physical`、`rows` / `cols` 四个组合各一条，外加顺序哨兵与 max 形状哨兵；
rev7 再加入 axis0 三个辅助输出的 source-derived stride、axis1 compact-prefix/实际 exp
分支选择及全局 dynamic-reject 用例；rev8 加入 flat+padded source 与 MXFP8/MXFP4
destination packed stride。实现不能只验证 valid shape、元素总数或 dst 自身 footprint。

### 11.4 lit 负向：`test/lit/pto/tmov_x2zz_invalid.pto`

逐条覆盖 §7.2 的 19 条规则，重点包含：

- `third_tile_without_address_space`（规则 1）——禁止 verifier/lowering 对第三个 tile 的角色做猜测；
- `grp_axis_on_scaling_fp`（规则 2）——scaling operand 明确属于 FP 分支，不能携带
  X-to-ZZ 的 `grpAxis`；
- `x2zz_with_fp_attributes`（规则 2）——非-scaling tmp 与 `accToVecMode` / `reluPreMode`
  等 FP/ACC 属性互斥；
- `mismatched_elem_type`（规则 4）；
- `elem_type_i8`（规则 5）——**必须有**：`i8` 会被 EmitC 降成 `int8_t`
  （`PTOToEmitC.cpp:445`），而 `CommonCheckZZ` 只接受 `uint8_t` / `hifloat8_t` /
  `float8_e8m0_t`。少了这条就是"verifier 放行、C++ 编译期 `static_assert` 才炸"，
  报错点离 IR 十万八千里；
- `src_wrong_layout` / `dst_wrong_layout`（规则 6/7）；
- `nd_dst_cols_odd`（规则 9）——注意是**奇数列**，不是"非 64 倍数"；
- `dn_src_rows_odd`（规则 10）；
- `dn_src_rows_is_one`（规则 10）——即原 `m = 32` 用例，见 §11.6；
- `tmp_too_small`（规则 12）——`tmp` 取 `63 + ceil(rows/16)*cols`，
  即**只差 1 字节**，用来钉住常数项是 64 而不是 32；
- `nd_src_padded_stride`（规则 15）——`src valid=20x4, physical=32x32`，
  总容量足够但 physical row stride 错，必须拒绝；
- `nd_src_tight_capacity`（规则 16）——`src valid=20x4, physical=20x4`，
  compact 但容纳不了 `32x4` source padding；
- `nd_dst_tight_capacity`（规则 17）——`dst valid=20x4, physical=20x4`，
  只容纳 valid 元素，但 ISA 会写 `32x4`；
- `dn_src_padded_stride`（规则 18）——`src valid=2x16, physical=2x32`，
  实现会按 stride 16 而不是 32 读下一行；
- `nd_dynamic_physical` / `dn_dynamic_physical` / `nd_dynamic_valid` / `dn_dynamic_valid`
  （规则 14）——固定"无法静态证明就拒绝"；
- `dn_tmp_dynamic_physical`（规则 14）——即使 DN ISA 不读 `tmp`，也必须在 verifier
  拒绝其动态 physical rows/cols，防止 PlanMemory 无法分配以及 EmitC 生成
  `Tile<..., -1, ...>`；
- `on_a3`（arch 拒绝）。

另对已有 `pto.tmov` 回归文件增加 `CHECK-NOT: TMOV<0>` 或等价断言，证明 scaling `fp`
不会被重分类为 X-to-ZZ；`PTORemoveIdentityTMov` 与 `PTOA5NormalizeTMov` 各增加一条
非-scaling 第三个 tile 用例，固定它既不能被删除，也不能被重写成普通 copy。

正向侧的 `nd_to_zz_tail` 与这里的 padded-stride/tight-capacity 负向组合在一起，
才能证明 verifier 拒绝的是错误 physical 契约，而不是简单禁掉非 16 对齐 valid 行。

### 11.5 精度用例：`test/samples/TquantMxDn/`

对照现有 `test/samples/TquantMx/` 的结构新建：

```
test/samples/TquantMxDn/
├── tquant_mx_dn.pto             # latest 支持范围内的 DN 完整链路
├── tquant_mx_dn_golden.py       # numpy 参考实现
├── tquant_mx_dn_compare.py      # 逐输出 dtype + 容差比较
└── npu_validation/golden.py     # 板测独立路径
```

golden 需要实现两段：

1. **DN 量化**（沿 axis 0 每 32 行一组）：
   ```text
   absmax  = max(|src[32g : 32g+32, n]|)                每组一个
   e8m0    = 依 OCP 规则由 absmax 推出（emax=8 for e4m3）
   scaling = 2^(254 - e8m0)
   dst     = clamp(src * scaling, -448, 448) → fp8 e4m3fn
   exp/max/scaling shape = [M/32, N]
   ```
2. **DN-to-ZZ 重排**，直接用 issue 给出的映射式：
   ```text
   E_ZZ[col_block, pair, q, delta] = E_DN[2 * pair + delta, 16 * col_block + q]
   q in [0, 16), delta in {0, 1}
   ```

完整链路 case 执行两段。两个 `n=1` case 只留在 §11.6.1 的原生行为审计中，不进入该
sample runner；禁止为了复用 runner 而把 unsupported TQUANT case 混入正向精度套件。

容差沿用 `tquant_mx_compare.py` 的做法：

| 输出 | 文件/存储 dtype | 比较方式与容差 |
|---|---|---|
| MXFP8 `dst` | int8/uint8（fp8 位型） | `atol = 0`，逐字节相等 |
| MXFP4 `dst` | `!pto.f4E2M1x2` 对应的 packed uint8 bytes | `atol = 0`，逐 packed byte 相等，不能展开后只比数值 |
| `exp` / `exp_zz` | uint8 | `atol = 0` |
| `max` | **与 source 相同**：f32 / f16 / bf16 | 按 case 声明的 dtype 和元素数读取；f16/bf16 解码后提升到 float32，再以 `atol = 1e-5` 比较 |
| `scaling` | **与 source 相同**：f32 / f16 / bf16 | 同 max；不得把 B16 文件直接按 float32 解释 |

exponent 与量化结果都是**位精确**的，不允许容差；只有 `max`/`scaling` 走浮点容差。
case metadata 必须同时记录 source dtype、每个输出的 storage dtype/shape 与 packed factor，
reader 先据此计算字节数。尤其 BF16/FP16 与 MXFP4 case 的 max/scaling 文件不能沿用
float32 reader，否则元素数量减半且位解释错误，无法证明输出。
golden 也先舍入到相同 f16/bf16 storage dtype，再提升到 float32 比较，避免拿无限精度参考值
误判合法的存储舍入。

同时给 `test/samples/TquantMx/` 追加一条 ND + `pto.tmov` 非-scaling tmp 的用例，
覆盖 ND-to-ZZ。

实现阶段只增加 §7.1.1 在 latest pin 上明确支持的精度 case。canonical B16 2D、DN FP32
interleave、DN FP16 MXFP4 非 32B packed stride 和不完整 B16 padding-tail 不建立正向 sample；
它们由 verifier 负向测试固定，原生层面的实际行为仅由 §11.6.1 记录。

原因是 PTOAS PlanMemory 会把 VEC allocation slot 对齐到 256B，普通 sample 的相邻 allocation
canary 可能掩盖 footprint 外写；同时 `max` 覆盖发生在合法 payload 内，单纯 redzone 也发现
不了。被 verifier 拒绝的组合不能再作为 PTOAS runtime 正向验收来证明 latest 安全。

`runop.sh` 的 `PTO_PTO_DIRS` 需要把 `TquantMxDn` 加入默认列表。

### 11.6 ST 用例：`test/tilelang_st/npu/a5/src/st/testcase/tquant_mx_dn/`

对照 `testcase/tmatmul_mx/` 的结构：`cases.py`（唯一真值源）、`gen_data.py`、
`compare.py`、`main.cpp`、`launch.cpp`、`CMakeLists.txt`、`tquant_mx_dn.pto`。

`cases.py` 只列 latest pin 下受支持的“DN 量化 + DN-to-ZZ 完整链路”；命中 §7.1.1
unsupported 边界的 TQUANT case 不进入正向 ST：

```python
FULL_CHAIN_CASES = [
    {"name": "dn_fp8_ocp_64x64",    "m": 64,  "n": 64,  "grp_axis": 0, "alg": "ocp", "eps": 0.0},
    {"name": "dn_fp8_ocp_128x32",   "m": 128, "n": 32,  "grp_axis": 0, "alg": "ocp", "eps": 0.0},
    {"name": "dn_fp8_nv_64x64",     "m": 64,  "n": 64,  "grp_axis": 0, "alg": "nv",  "eps": 0.0},
    {"name": "dn_fp4_bf16_ocp_64x64", "m": 64, "n": 64, "grp_axis": 0,
     "alg": "ocp", "fp4": True, "src_type": "bf16"},
    {"name": "dn_fp8_ocp_64x96",    "m": 64,  "n": 96,  "grp_axis": 0, "alg": "ocp"},   # hatM=2 最小合法
    {"name": "nd_fp8_f32_ocp_16x128", "m": 16, "n": 128, "grp_axis": 1,
     "alg": "ocp", "src_type": "f32"},
    {"name": "nd_fp8_f32_ocp_20x128", "m": 20, "n": 128, "grp_axis": 1,
     "alg": "ocp", "src_type": "f32"},
]
```

`m = 32,n = 1` / `m = 64,n = 1` 分别用于 §11.6.1 复现 B16 padding-tail 与 FP32
interleave destination scratch，不是 latest 支持的 PTOAS 正向 case。即使只看 DN-to-ZZ，
前者的 exponent `1x1` 同时违反 rows ≥ 2 与 cols % 16，后者的 `2x1` 仍违反 cols % 16；
因此 case generator 不得把它们拼进完整链路。

当 exponent cols 同时满足 `% 16 == 0` 时，`m = 64`（`hatM = 2`，`numPairs = 1`）
才是 DN-to-ZZ 真正的最小合法行边界；`dn_fp8_ocp_64x96` 单独固定该边界。
`m = 20` 的 ND 用例用来钉住"非 16 对齐行数由 ISA 的 ceil + `ZeroSourcePaddingB16` 处理"
这条结论（§7.2 规则 9 注记）。该 case 的 source/destination 必须显式分配至少
`align16(20) * (128/32) = 128` 字节，source 必须是 compact prefix；它同时验证
source padding 零写、destination padded 写出与 `tmp` 容量公式里的 `ceil` 都没有算少。

DN FP16 MXFP4 非 32B packed stride、FP32 interleave 与不完整 B16 padding-tail 只保留
§11.3 verifier 负向测试和下一节的原生行为证据，不进入 ST case list。

### 11.6.1 latest pin 的 PTO-ISA 原生行为审计（非 pin 门槛）

为保存 §3.1.1 的实证依据，可用独立 C++ harness 直接构造 TileData 并调用
`69a81f3/40e741b` 的 `TQUANT`。它不经 PTOAS parser、verifier、lowering、
`pto.alloc_tile` 或 PlanMemory，因此能准确暴露 latest 的隐藏写。该 harness 不要求合入
pto-isa，也不决定 pin SHA；PTOAS 的合入保护来自 §11.3 对这些组合的 verifier 拒绝。

每个 testcase 使用一个显式对齐的 raw byte backing buffer，并由 harness 手工切分所有
tile payload / redzone；禁止调用会把每个 tile 扩到 256B slot 的 allocator。每个被监测 tile
都按下列方式布局：

```text
aligned tile base
├── declared payload: exactly physicalRows * physicalCols * storageBytes
├── suffix redzone: starts at payloadBase + declaredBytes, without padding
└── optional alignment gap before the next tile base: also filled with sentinel
```

tile base 仍满足 PTO-ISA 的地址对齐要求，但 redzone 的第一个字节必须恰好是声明 footprint
的下一字节；下一个 tile 即使需要再次对齐，中间 gap 也属于 redzone。执行前填充 prefix /
suffix redzone 与 alignment gap，执行后逐字节检查 sentinel。harness 还要记录每个 TileData
指针、declared bytes、redzone 起止和首个破坏 offset，失败时能区分“payload 内可观察输出
不对”和“footprint 外写”。

原生 case 固定为六组：

| case | 被监测 payload / 紧贴 redzone | 最低判据 |
|---|---|---|
| canonical B16 MXFP8 small groups `2x32` | exp `2x1 ui8 = 2B`；redzone 从 byte 2 开始并覆盖 `exp+16` 后的完整临时写范围 | latest 的 `exp+16` 写破坏 redzone，对应 verifier 必须拒绝 |
| canonical B16 MXFP8 eight groups `8x32` | max 是可观察 payload | latest 的 max 与 golden 不一致，对应 verifier 必须拒绝；这是 payload 覆盖检测，不以 redzone 代替 |
| canonical B16 MXFP4 `2x32` | max 是可观察 payload；exp/max/scaling/dst 的 suffix redzone 都紧贴 footprint | latest 覆盖 max，对应 verifier 必须拒绝，dst 仍按 packed byte 审计 |
| DN FP16 MXFP4 unaligned packed stride `64x96` | dst suffix redzone 从 packed destination footprint 的下一字节开始 | latest 从 dst 末尾借 scratch 并命中 redzone，对应 verifier 必须拒绝 |
| DN FP32 interleave `64x1`、src physical cols 128 | tight dst `64x1 ui8 = 64B`；redzone 至少覆盖 offset 128 的第二个 exponent scratch row 及其写宽 | latest 的 source-derived-stride scratch 命中 redzone，对应 verifier 必须拒绝 |
| DN BF16 padding tail `32x1`、src physical `32x2` | source `32x2 bf16 = 128B`；redzone 紧接 byte 128 并至少覆盖到一个完整 256B VL store 末端 | latest 写第 33–64 行并命中 redzone，对应 verifier 必须拒绝 |

redzone 长度由“latest 实现最远可能写到的 byte offset + 一次写宽”推导，并在 harness 中以
`static_assert` / checked arithmetic 固定，不能只取一个经验常量。canonical max overwrite
发生在合法 payload 内，必须靠逐元素 golden 发现；其余四类 footprint 越界行为还必须由
精确 redzone 发现。
`69a81f3` 与 `40e741b` 的六个 case 应分别复现对应 redzone/max-golden failure；若某个 case
意外通过，先重新核对最新头文件与命中分支，再决定是否可缩小 verifier 的 unsupported
范围。该审计结果记录 revision 和首个 failure offset，但不阻止 commit 0 把 pin 对齐 latest。
将来上游修复后，可用同一 harness 证明某条限制消失，再配套放宽 PTOAS verifier。

### 11.7 PTO-BC roundtrip

参照 `tools/ptobc/tests/fp_operand_forms_v0_encode.sh` 的范式新建
`tools/ptobc/tests/mx_grp_axis_v0_encode.sh` + testdata：

1. CTest 用 `${CMAKE_COMMAND} -E env --unset=PTOBC_ALLOW_GENERIC ...` 包装新增脚本，
   明确清除 `PTOBC_ALLOW_GENERIC` 后执行 `ptobc encode` → `ptobc decode`；这一步先证明
   `TQuantMxOp` / X-to-ZZ 的显式 compatibility shim 生效，而不是测试环境全局放行未知 op；
2. testdata 同时放入 `pto.tquant.mx`、既有 scaling-FP `pto.tmov` 和非-scaling-tmp
   `pto.tmov`；grep 断言 `grpAxis = #pto<mx_group_axis axis0>`、`interleave = true`、
   两种 `pto.tmov` 的第三个 tile 地址空间都保持不变；
3. **把 roundtrip 结果重新喂给 `ptoas --emit-pto-ir` 验证**（光 grep 文本证明不了属性字典还自洽）；
4. 对旧 scaling-FP case 检查编码仍选择 `kTMovFpWireOpcode`，并用既有 v0 fixture 做
   byte-for-byte/schema 回归；`fp=`、optional operand index 与旧 FP payload 均不变；
5. 对非-scaling X-to-ZZ case 检查编码选择已有 `kOpcodeGeneric` v0 compatibility record，
   完整携带第三个 operand 与 `grpAxis`，并明确 `CHECK-NOT` / 二进制解析断言它没有选择
   `kTMovFpWireOpcode`；
6. 对 `pto.tquant.mx` 直接解析 binary op record，断言 opcode 是
   `kOpcodeGeneric (0xFFFF)`、generic op 名正确、五个 operand 与属性字典完整；测试还要
   在 `PTOBC_ALLOW_GENERIC` 未设置时单独 encode 一个 TQuantMx-only fixture，防止它从
   compatibility shim 回退到“仅测试环境可用”的 unknown-op 路径；
7. 扩展 `v0_fp_schema_compatibility_check.py`，钉住真正存在的 `pto.tquant`、普通
   `pto.tmov` known-op schema 与 scaling-FP legacy wire schema；同时检查
   `ptobc_opcodes_v0.h` 的 name/opcode 表**不包含** `pto.tquant.mx`。TQuantMx 与 X-to-ZZ
   都复用已有 generic v0 opcode，两层都不分配新的专用 opcode。

### 11.8 运行方式

```bash
# 0) pin 对齐与接口验证（两个 remote 分别 fetch/checkout）
git -C <gitcode-pto-isa> checkout 69a81f3b2d145fe4f9925cfd65a083f78ad1f804
git -C <github-pto-isa> checkout 40e741bf1cfce99da3b1caa514e08c2f72894922
# 可选：运行 §11.6.1 原生 harness，记录 latest 已知限制；失败不是 pin 拒绝条件

# 1) 构建 PTOAS（本地开发树；LLVM_BUILD_DIR 指向已有 LLVM 构建）
LLVM_BUILD_DIR=<llvm-build> ./quick_install.sh
#    或直接用 ninja
ninja -C build ptoas

# 2) 定向 lit（开发迭代用，最快）
ninja -C build ptoas
build/bin/llvm-lit -sv build/test/lit/pto/tquant_mx_grp_axis_emitc.pto \
                       build/test/lit/pto/tquant_mx_grp_axis_invalid.pto \
                       build/test/lit/pto/tmov_x2zz_emitc.pto \
                       build/test/lit/pto/tmov_x2zz_invalid.pto

# 3) 全量 lit
ninja -C build check-pto

# 4) ctest（含 PTO-BC roundtrip 与 schema 守卫；明确清除 generic 全局放行）
ninja -C build check-ctest
env -u PTOBC_ALLOW_GENERIC ctest --test-dir build \
  -R 'ptobc_mx_grp_axis_v0_encode|ptobc_v0_fp_schema_compatibility_check' \
  --output-on-failure

# 5) 精度样例（py -> pto -> cpp，本地不含板卡）
PTOAS_BIN=build/tools/ptoas/ptoas ./test/samples/runop.sh -t TquantMxDn
PTOAS_BIN=build/tools/ptoas/ptoas ./test/samples/runop.sh -t TquantMx

# 6) 生成 C++ 的编译验证（需要匹配的 pto-isa/CANN）
#    确认 TQUANT<0, MxQuantAlg::...> 与 TMOV<0>(...) 能被 pto-isa 头文件解析
#    参考 docs/no_npu_compile_only_guide_zh.md

# 7) A5 ST（需要板卡或 CPU-sim 环境）
#    参考 test/tilelang_st/npu/a5 的既有流程运行 tquant_mx_dn
```

提交 0 必须先跑完第 0 步的 remote 存在性、checkout、头文件接口与既有编译线验证；
功能实现后的最终验收必须跑完 1–6，7 在具备环境时补充。
第 6 步不能省：本特性的全部价值就是生成能被 pto-isa 接受的调用，lit 只能证明文本形态。

## 12. 实施顺序

建议按可独立回滚的粒度切成 8 个提交：

| # | 内容 | 验证 |
|---|---|---|
| 0 | **直接对齐两个 remote 的 latest pin**：GitCode CI 默认值/job env、主 Dockerfile 与 remote-validation 更新到 `69a81f3b2d145fe4f9925cfd65a083f78ad1f804`；GitHub `ci_sim` 更新到 `40e741bf1cfce99da3b1caa514e08c2f72894922`。后者是已合并 PR #239 的 merge commit，明确同步前者。重新引入 repo-aware target map/updater，不创建未来占位 revision，不等待 pto-isa 再合其他修改；`Dockerfile.dev@662d7f2` 继续按 CANN 9.0 独立处理 | 两个 remote 分别 fetch/checkout；核对 grouped/interleave 与 X-to-ZZ 接口；实跑现有 CI/容器线。`Dockerfile.dev` 单独尝试 latest 的 CANN 9.0 编译验证，通过才更新，否则保留旧 pin 并记录依据；§11.6.1 只记录已知行为，不作为 pin gate |
| 1 | `MxGroupAxis` 枚举 + `TQuantMxOp` 两个属性（ODS/parser/printer/CAPI/Python） | 现有 lit 全绿（默认值保证零行为变化） |
| 2 | `TQuantMxOp::verify()` 按轴分派，axis1 同时保留 canonical / legacy flat；落实 §7.1.1 的 flat-tight-source、MXFP8/MXFP4 destination packed stride、完整 physical 矩阵、dynamic reject、latest unsupported 边界与 f16/bf16 padded-source `Read + Write` effects；同提交把 `tquant_mx_a5_emitc.pto` 的 MXFP4 dst physical cols `32→16` | 两个 tight MXFP8 legacy IR 不迁移 + MXFP4 迁移后 lit + flat-padded/dst-stride/physical/动态/latest-unsupported 负向 lit + MemoryEffects/PlanMemory/InsertSync 回归 |
| 3 | `PTOQuantMxToEmitC` 形态 B + §11.1 正向用例 + 更新受影响的既有 lit 期望 | 定向 lit + `check-pto` |
| 4 | 保留 `TMovOp::$fp` API，增加 optional `grpAxis` 与共享 `classifyTMovForm(getFp())`；扩展 verifier / memory effects，包含 compact/stride/capacity/dynamic-physical 契约与 ND `src` 读写效应 + §11.2/11.4 用例 | Python `fp=` / C++ `getFp()` 兼容回归 + 既有 plain/FP TMOV 回归 + 定向 lit + PlanMemory / InsertSync 回归 |
| 5 | 扩展既有 `PTOMovToEmitC` 的 X-to-ZZ 分支，审计 RemoveIdentity/Normalize/InferMemScope 等 TMov 消费者 + §11.2 正向用例 | 定向 lit + `check-pto`，证明 scaling FP 文本不变且非-scaling tmp 不被误删/误改写 |
| 6 | PTO-BC roundtrip + schema 守卫：给 `TQuantMxOp` 增加无环境变量依赖的 generic compatibility shim；scaling FP 保持 `kTMovFpWireOpcode`，X-to-ZZ 走已有 generic v0 record 并保留 `grpAxis`；不分配新专用 opcode，也不虚构 TQuantMx known-op schema | 不设置 `PTOBC_ALLOW_GENERIC` 运行 `check-ctest`；二进制解析确认 TQuantMx/X-to-ZZ 为 `kOpcodeGeneric`，并检查 scaling FP 分支不能互换 |
| 7 | latest 支持范围内的精度样例 + ST 用例 + 手册 / SPEC / ReleaseNotes；`n=1` 与四类 unsafe 组合不作为正向 case | `runop.sh` + 全量 lit + 受支持 sample/ST；§11.3 的 latest-unsupported 负向用例必须全过 |

提交 3 会改动既有 lit 的期望文本（§8.1），单独成一个提交便于 review 和回滚。

提交 0 必须**排在最前**：`interleave` 的 verifier 常量（64 对齐、`align32(2N)`）、
TQuantMx 的 destination/source-derived stride、latest unsupported 边界与 X-to-ZZ 的
compact/padding 契约都绑定目标 pin 的真实头文件。`f03c2454` 的 ancestry 事实仍成立，但不再
参与选择；rev11 的唯一目标就是 GitCode `69a81f3` 与 GitHub `40e741b`，并对两个 remote
独立验证。

提交 0 不引用 §11.5/11.6 的 PTOAS case：这些 case 需要提交 1–5 才新增的属性、verifier
与 lowering。提交 0 只验证 exact SHA、接口与既有构建线；§11.6.1 可作为补充审计记录
latest 的已知限制，但其预期失败不阻止 pin bump。

**合入门槛已经固定**：本 PR 在实现阶段先完成 latest pin 与 CI/容器验证，再按 §7.1.1
拒绝当前头文件不能安全承诺的组合；不等待 pto-isa 隐藏 scratch / padding-tail 修复。
`Dockerfile.dev` 仍按 CANN 9.0 独立选择：latest `69a81f3` 验证通过才更新，否则保留
`662d7f2a` 并记录兼容性依据。本 PR 不把三元组建模、repo-aware target map/updater 或
本特性 pin bump 转交给后续 PR，也不依赖 #1122 的未来工作。

## 13. 风险、结论与行动

1. **~~DN-to-ZZ 的 `tmp` 容量下界未知~~ —— 已关闭，结论是"不使用"。**
   `GenerateB8IndicesDN2ZZToUB` 第一行 `(void)tmp;`，DN 路径不碰 `tmp`。因此不存在"推导
   ISA 下界"的问题，也不需要"大小 `tmp` 对照实验"（两者都会通过，对照不出任何东西）。
   这只免除容量公式与 memory effect，不免除 PTOAS 的静态 local-tile 类型要求；DN `tmp`
   physical shape 仍按 §7.2 规则 14 要求静态。
   → **遗留行动**：这条事实被 §9 的按轴效应建模所依赖，因此把"复核 DN 是否仍不使用
   `tmp`"写进 **pin bump 检查清单**：
   `grep -n "void)tmp" include/pto/npu/a5/TMov.hpp` 应在 DN 函数体内命中。
2. **~~pto-isa pin 是否包含所需接口~~ —— 已关闭，结论是直接 bump 到 latest。**
   评审最初确认 `ce3262e3` 缺少 `bool interleave` overload，而 `f03c2454` 已包含该接口与
   当时的 CPU-sim duplicate-stub 修复；该 ancestry 事实仍成立。但 rev11 重新盘点 PTOAS
   `main@f8912bc7` 后，目标不再做共同后继搜索，也不等待新的修复 SHA：

   | 目标 | 远端 | rev11 当前 SHA | 本特性目标 |
   |---|---|---|---|
   | `.github/workflows/ci.yml` 默认值、job env | GitCode `cann/pto-isa` | `27386d906e8fdcbd93aec84197939bc0b2c6caea` | `69a81f3b2d145fe4f9925cfd65a083f78ad1f804` |
   | `docker/Dockerfile` | GitCode `cann/pto-isa` | `27386d906e8fdcbd93aec84197939bc0b2c6caea` | `69a81f3b2d145fe4f9925cfd65a083f78ad1f804` |
   | `test/npu_validation/scripts/run_remote_npu_validation.sh` | GitCode `cann/pto-isa` | `27386d906e8fdcbd93aec84197939bc0b2c6caea` | `69a81f3b2d145fe4f9925cfd65a083f78ad1f804` |
   | `docker/Dockerfile.dev` | GitCode `cann/pto-isa` | `662d7f2a916d6bbde3109ce4a16ed5c28f5d900a` | CANN 9.0 独立验证；latest 通过才更新，否则保持 |
   | `.github/workflows/ci_sim.yml` | GitHub `hw-native-sys/pto-isa` | `e948507e18ec4f39037a04914b97e77f5b9d75e3` | `40e741bf1cfce99da3b1caa514e08c2f72894922` |

   GitCode `master@69a81f3` 是当前发布头；GitHub PR #239 已合并为
   `main@40e741b`，其 commit message 是 `sync: GitCode master @ 69a81f3b2d14`。因此这里的
   两个 SHA 不是“不同功能版本”，而是同一轮内容同步在两个 remote 上的实际 revision。

   → **pin 落地顺序已定**：

   1. 分别从 GitCode 与 GitHub fetch/checkout 上表 SHA，验证 grouped/interleave、X-to-ZZ
      overload 及 CPU-sim duplicate-stub 契约；不以一个 remote 的可达性代替另一个。
   2. GitCode 的 CI/主 Dockerfile/remote-validation 更新为 `69a81f3`；GitHub `ci_sim`
      更新为 `40e741b`。不创建占位符，不等待 pto-isa 再合隐藏 scratch/padding-tail 修复。
   3. `.github/scripts/update_pto_isa_pin.py` 已在 `e488f9e3d` 删除，当前不存在可“扩展”的
      updater。本 PR 重新引入显式 target map：每项携带
      **(repo, revision, 兼容性约束, files)**，或实现等价的 repo-aware updater；禁止把
      GitCode SHA 原样广播到 GitHub。
   4. `Dockerfile.dev` 的基础镜像是 CANN 9.0.0，因此验证线与 target 独立；尝试 GitCode
      latest `69a81f3`，通过才更新，失败则保留 `662d7f2a` 并记录兼容性依据。
   5. 本 PR 完成 target map/updater 与上述 pin bump，不把工作转交 #1122 或另一个 PTOAS
      后续 PR。
3. **~~ND 分组 shape 兼容策略~~ —— 已关闭，axis1 采用双形态契约。**
   兼容性破坏已由现有 IR 直接证明：`tquant_mx_a5_emitc.pto` 的 MXFP8 case 和
   `quant_mx_tile_native.pto` 的 MXFP8 case 都对 `src=16x32` 使用
   `exp/max/scaling valid=1x16`。
   新 canonical 形状是 `16x1`，如果只做逐维 canonical 校验，会破坏仓库内两个已知
   tight-source legacy IR 的兼容性。
   → **结论**：axis1 的 shape 分类同时识别 canonical `[M, N/32]` 和 legacy flat
   `[1, M*N/32]`；axis0 只接受 `[M/32, N]`。latest pin 下 canonical B16 再由
   §7.1.1 拒绝。这两个 tight MXFP8 case 不迁移 shape；MXFP4 destination 的已知 physical
   cols 迁移单列在问题 11，不再笼统承诺整个仓库零迁移。
4. **§8.1 的生成文本变化。** 现有 ND IR 的输出会从"类型列表模板参数"变成
   `<1, MxQuantAlg::...>`。语义等价但文本变化，属于对下游 golden/期望文件的影响面。
   → **行动**：提交 3 单独处理，并在 ReleaseNotes 说明。
5. **`exp_zz` fused 路径的最终去向。** 本次只标 deprecated。若确认其设备侧不可用，
   应单开 PR 删除，并按 #1122 的做法处理 PTO-BC 兼容（保留 wire alias 或明确不兼容）。
6. **`interleave` 的 exponent 形状。** 早期版本取自 issue 的 `[ceil(M/64), 2N]`，
   评审据较新 pto-isa 修正为：valid `[M/64, 2N]`（**行数 64 严格对齐，不是 ceil**）、
   physical `[M/64, align32(2N)]`。已按此改写 §7.1。
   → **行动**：仍需在每个最终目标 pin 上以代码为准复核一次（同已关闭问题 2）。
7. **~~TQuantMx dynamic shape 如何处理~~ —— 已关闭，无法静态证明即拒绝。**
   PTO-ISA 的 axis0 辅助输出行步长来自 `TileDataSrc::Cols`，axis1 又用
   `TileDataExp::Rows` 选择 flat/2D exp 分支并把 max/scaling 当 compact prefix。
   只比较已知 valid 维会放行无法证明 stride/capacity 的 IR。
   → **结论**：§7.1 涉及的 `src/dst/exp/max/scaling`（以及存在时的 deprecated
   `exp_zz`）valid/physical shape 全部要求静态；任一动态即拒绝。§11.3 用 axis0
   source-derived stride 与 axis1 compact-prefix 两个分支的负向用例固定该策略，
   不再保留 `dn_dynamic` / 全动态正向用例。
8. **~~X-to-ZZ dynamic shape 如何处理~~ —— 已关闭，静态证明不了就拒绝。**
   ND 需要同时证明 compact source、source/destination padded capacity 和 tmp 容量；
   DN 需要证明 source physical row stride 与 destination capacity。`src`/`dst` 的 valid 或
   physical shape 任一动态时，这些都无法由当前 IR 静态证明。DN `tmp` 虽不参与 ISA
   计算，仍必须由 PlanMemory 分配 local slot，并作为静态 `Tile<...>` 实参传给
   `TMOV<0>`；因此其 physical shape 也不能动态。`tmp` valid shape 不驱动 ISA 寻址，
   仍沿用 tile 类型的既有策略。
   → **结论**：按 §7.2 拒绝 `src`/`dst` 的相关动态 shape，以及 ND/DN `tmp` 的动态
   physical shape；本次不隐式创建 compact copy/scratch，也不增加静态 dummy-tile lowering。
9. **~~`TMovOp` 第三个 operand 是否改名~~ —— 已关闭，公开 API 保持 `$fp`。**
   TableGen operand 名会生成 Python `fp=`、C++ `getFp()` 与 builder 参数；把 `$fp`
   改成 `$aux` 不是内部重构，会直接破坏现有调用。
   → **结论**：ODS 继续命名 `$fp`，仅由共享 `classifyTMovForm(getFp())` 按地址空间赋予
   FP 或 X-to-ZZ tmp 语义。旧 scaling FP 保持 legacy FP wire；非-scaling 第三个 tile
   走 generic v0 compatibility record，避免被 `getFp()` 的存在性误编码成 FP payload。
10. **~~PTO-ISA TQuantMx 隐藏 scratch / padding tail 是否阻塞 pin~~ —— 已关闭，不阻塞 latest pin。**
    GitCode `69a81f3` / GitHub `40e741b` 仍会在 canonical B16 小 group case 写
    `exp+16`，在较大 MXFP8 与全部 MXFP4 canonical case 覆盖 `max`，并在 DN FP16
    MXFP4 的非 32B packed stride 下越过 `dst` footprint 使用 scratch；DN FP32 interleave
    还会借用 tight dst，B16 VL-aligned padding 会在最后一个不完整 VL 越过 source rows。
    → **结论**：pin 仍直接对齐 latest，不要求 pto-isa 再合修改；同时不重新定义 `max`，也
    不要求用户伪造更大的 visible input/output。§7.1.1 把四类对应组合标为 unsupported，
    §11.3 逐条验证拒绝。§11.6.1 的 raw-buffer redzone/max-golden 只保存 latest 行为证据，
    不再要求“候选 revision 六组全过”。将来上游行为变化后，必须显式放宽 verifier 并补
    正向精度测试，不能让 pin bump 自动改变合法集合。
11. **~~既有 MXFP4 lit 是否需要迁移~~ —— 已关闭，需要迁移一个 physical shape。**
    `test/lit/pto/tquant_mx_a5_emitc.pto` 的 MXFP4 source physical cols 是 32，packed dst
    physical cols 也是 32；§7.1.1 的真实寻址契约要求后者为 16。
    → **结论**：提交 2 将 `%dst_fp4` 的 `cols=32` 改为 `cols=16`，保持 `v_col=16`，并同步
    `outs` type。ReleaseNotes 记录 MXFP4 destination physical stride 收紧。兼容承诺只指
    问题 3 的两个 tight MXFP8 legacy-flat case，不覆盖这个已知 padded MXFP4 destination。

## 14. 验收标准

- issue #1185 列出的四条契约（axis 0/1 TQUANT、RowMajor 量化数据 + raw ND/DN exponent、
  数据 ND-to-NZ、exponent ND/DN-to-ZZ）都能用 PTO IR 表达；
- `grpAxis` / `interleave` 以及 `pto.tmov` 的 X-to-ZZ 分支生成的 C++ 与 §8 的模板参数表逐字一致；
- 所有不带 `exp_zz` 的 `pto.tquant.mx` 都生成 §8.1 形态 B，既有默认 ND golden 统一更新，
  不保留按属性是否显式出现而分叉的旧类型列表 EmitC 路径；
- 不新增独立 X-to-ZZ op；`pto.tmov` 的 scaling 第三个 tile 继续走 FP lowering，
  非-scaling 第三个 tile 走三参数 X-to-ZZ `TMOV`，无第三个 tile 的普通形态保持不变；
- `TMovOp` 的 ODS operand 名仍为 `$fp`，现有 Python `fp=`、C++ `getFp()` 与 builder
  源码兼容；实现中不存在公开 `aux=` / `getAux()` API；
- 两个既有 tight MXFP8 legacy-flat case **语义与 shape 不变**：tight-source legacy flat
  `[1, M*N/32]` 继续受支持；axis1 canonical `[M, N/32]` 在 latest pin 下先支持 f32，
  B16 canonical 受 §7.1.1 限制；不再对所有现有 `tquant.mx` IR 作零迁移承诺；
- `tquant_mx_a5_emitc.pto` 的 MXFP4 packed dst physical cols 按 §13-11 从 32 迁移为 16，
  valid cols 保持 16，ReleaseNotes 明确记录该 physical-stride 收紧；
- TQuantMx 的 physical 契约与 §7.1.1 逐项一致：axis0 的 max/scaling 及非-interleave exp
  使用 source-derived physical row stride；axis1 max/scaling 是 compact prefix，exp 的
  flat/2D 校验与 `TileDataExp::Rows` 实际选择一致；flat+padded source 被拒，f32 padded source
  使用 canonical 2D；MXFP8/MXFP4 destination 分别按 source stride / packed source stride
  验证 physical cols 与 capacity；相关 valid/physical shape 动态时拒绝；canonical B16 2D、
  DN FP16 MXFP4 非 32B packed stride、DN FP32 interleave 与不完整 B16 padding VL
  在 `69a81f3/40e741b` 下明确拒绝；
- §7 的每一条 verifier 规则都有对应的负向 lit 用例；
- ND X-to-ZZ 的 padded-stride、source 容量不足、destination 容量不足和动态
  valid/physical shape 均被 verifier 拒绝，而紧密且容量足够的非 16 对齐行用例通过；
  DN `tmp` 的动态 physical shape 也被拒绝，静态占位 tile 能由 PlanMemory 分配且 EmitC
  不生成含 `-1` physical template 维的 `Tile`；
- `TMovOp` 的分支效应精确匹配 §9：ND X-to-ZZ 的 `src/tmp` 为 `Read + Write`、
  `dst` 为 `Write`；DN X-to-ZZ 的 `src` 为 `Read`、`dst` 为 `Write`、`tmp` 无效应；
  既有 FP 形态仍是 `src/fp: Read`、`dst: Write`；
- `TQuantMxOp` 对 f16/bf16 且 `srcValidCols < srcPhysicalCols` 的 source 声明
  `Read + Write`，f32 或 tight f16/bf16 source 保持 `Read`；padding 后再次读取 source 的
  PlanMemory/InsertSync 回归证明依赖边未丢失；
- latest 支持范围内的 DN 量化 + DN-to-ZZ 完整链路精度用例在 exponent 与量化结果上
  **位精确**通过；`m=32,n=1` / `m=64,n=1` 只保留为原生行为诊断，不误接 DN-to-ZZ，
  也不作为 PTOAS 正向 TQUANT case；max/scaling 按 source storage dtype 读取后再提升比较，
  MXFP4 dst 按 packed byte 比较；
- §11.6.1 的独立 PTO-ISA harness 使用不经 PlanMemory 的 raw backing buffer，六组 case 的
  suffix redzone 均从声明 footprint 下一字节开始；`69a81f3/40e741b` 的已知 failure 与
  §11.3 的 verifier 拒绝一一对应。harness 是非 gating 行为证据，不要求另一个修复 revision；
- PTO-BC roundtrip 保持真正存在的 `pto.tquant`、普通 `pto.tmov` 与 scaling-FP legacy wire
  schema 不变；known-op 表仍不含 `pto.tquant.mx`。不设置 `PTOBC_ALLOW_GENERIC` 时，
  `TQuantMxOp` 与 X-to-ZZ 都经显式 compatibility shim 选择现有 `kOpcodeGeneric` 并完整
  保留 operands/attributes；scaling FP 继续选择 `kTMovFpWireOpcode`，不得互换；
- updater 能为每个目标显式处理 `(repo, revision, 兼容性约束)`，分别验证 GitCode/GitHub；
  GitCode 的 CI/主 Dockerfile/remote-validation 固定为 `69a81f3`，GitHub `ci_sim` 固定为
  `40e741b`，且本 PR 完成本特性所需 pin bump；`Dockerfile.dev` 有独立的 CANN 9.0 pin
  与验证记录；
- `docs/PTO_IR_manual.md` 与 SPEC 中的 shape 表、约束表与 verifier 实现一致。
