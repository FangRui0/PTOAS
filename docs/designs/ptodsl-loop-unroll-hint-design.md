# PTODSL / PTOAS Loop Unroll Hint 设计文档

> 关联 Issue：
> - [Issue #1242](https://github.com/hw-native-sys/PTOAS/issues/1242) Requirement 2 —— `pto.for_` loop-unroll hint（`unroll="enable"`）
> - [Issue #1000](https://github.com/hw-native-sys/PTOAS/issues/1000) —— 支持 Loop Unroll Hint（含 `pto.range`、factor unroll、两阶段计划）
> - [PR #838](https://github.com/hw-native-sys/PTOAS/pull/838) —— `PTOUnrollSIMTForPass`（规避 BiSheng AICore 后端 bug 的临时方案，本次一并重构）

---

## 1. 背景与动机

### 1.1 需求来源

**Issue #1242 Requirement 2**：SIMTVF codegen 需要表达无 factor 的 `#pragma unroll` 语义——保留 device-side loop，将展开意图交给 LLVM/BiSheng 的 cost model 决定 full 或 partial unroll。当前 PTODSL 只有两个极端：

- `pto.static_range`：trace 阶段强制完全展开，增加 trace/编译时间、IR 体积与寄存器压力；
- `pto.for_`：保留 device loop，但无法携带任何 frontend unroll hint。

**Issue #1000**：TileLang-PTO 后端需要将 `T.unroll(..., explicit=False)` lower 成对等语义（编译器侧 unroll，而非 DSL 前端强制展开）。该 issue 给出了两阶段计划：阶段一由 Bisheng/CCE 执行展开（hint 透传），阶段二由 PTOAS 原生展开。

**PR #838（历史背景）**：`PTOUnrollSIMTForPass` 是为了规避 BiSheng AICore 后端 bug 的临时方案——SIMTVF kernel 中 `scf.for` + `scf.if` 常量分支经 SCF→CF→LLVM lowering 后，AICore 后端未正确处理 `SimtEntry` calling convention，给 `END` 生成了带谓词的 `END @!P0`。规避手段是将 SIMT 上下文内标注 `{pto.unroll = "full"}` 的常量循环强制完全展开，由下游 SCCP + canonicalize 消除常量分支，生成无分支直线代码。Issue #1000 的评论明确指出该 pass 是临时方案，本次应一并重构为通用 unroll 能力。

### 1.2 目标

1. `pto.for_` / 新增 `pto.range` 支持 `unroll="enable"` hint：保留 device loop，hint 透传到 LLVM IR 的 `!llvm.loop.unroll.enable` metadata，由 BiSheng cost model 决定展开策略（#1242 验收标准）。
2. 支持 `unroll="full"` 与 `unroll_factor=N`：PTOAS 优先原生展开，失败时降级为编译器 hint（#1000 阶段二重构）。
3. 将 `PTOUnrollSIMTFor` 从"SIMT-only、full-only 的临时 pass"重构为通用的 attr 驱动 unroll pass，同时保持 #838 的 bug 规避语义不回归。
4. 未指定 hint 的循环行为完全不变。

### 1.3 非目标（Non-goals）

- 不修改 LLVM/BiSheng 的 unroll cost model；
- 不保证 `enable` hint 选择固定展开 factor；
- 不改变 `pto.static_range` 的 trace-time full-unroll 语义；
- 不支持任意动态循环的 full unroll（动态 bound 的 `"full"` 降级为 metadata hint）；
- 不包含 TileLang codegen 侧的修改。

---

## 2. 现状分析

### 2.1 现有 unroll 能力

`PTOUnrollSIMTForPass`（原 `lib/PTO/Transforms/PTOUnrollSIMTForPass.cpp`，本设计中重构为 `PTOUnrollLoopsPass.cpp`）：

- 只处理显式标注 `{pto.unroll = "full"}` 的 `scf.for`（`PTOUnrollSIMTForPass.cpp:58-66`）；
- 只在 SIMT 上下文内生效（`pto.simt_entry` 函数或 inline `pto.section.simt` 区域）；
- 要求静态 lb/ub/step、正 step，通过 `loopUnrollByFactor(tripCount)` 全展开；
- 在 `prepareVPTOForEmission` 中、SCCP/canonicalize/CSE 之前运行（`tools/ptoas/ptoas.cpp:3074`），展开后常量分支被下游折叠。

PTODSL 目前**没有任何 public API** 设置 `pto.unroll` attr——该 attr 只出现在手写 `.pto` 测试中（`test/test_unroll_annotation.mlir`、`test/lit/vpto/unroll_inline_simt_sections.pto`）。

### 2.2 关键发现：vendored MLIR 已具备完整的 loop-annotation 下传链路

无需对 LLVM/BiSheng 对接层做任何改动：

1. **SCF→CF**：vendored LLVM 19（feature-vpto）的 `SCFToControlFlow` **不会**把 `scf.for` 上的 `llvm.loop_annotation` 拷贝到 latch `cf.br`（该能力只在更新的上游版本存在，`llvm-workspace` 的 LLVM 21 已具备）。因此本方案中 Pass B 对带注解的 loop 自行完成 SCF→CF 降级并把注解挂到 latch 上（见 3.1 边界情形）；未注解的 loop 仍走上游 `convert-scf-to-cf`；
2. **CF→LLVM**：`BranchOpLowering` / `CondBranchOpLowering` 将全部 attrs 保留到 `llvm.br` / `llvm.cond_br`
   （`llvm-workspace/llvm-project/mlir/lib/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.cpp:130-170`）；
3. **MLIR→LLVM IR**：`translateModuleToLLVMIR` 把 latch 上的 `#llvm.loop_annotation` 翻译为 `!llvm.loop` metadata
   （`llvm-workspace/llvm-project/mlir/lib/Target/LLVMIR/LoopAnnotationTranslation.cpp:124-138`）：

   | MLIR attr | LLVM IR metadata |
   |---|---|
   | `#llvm.loop_annotation<unroll = <disable = false>>` | `!{!"llvm.loop.unroll.enable"}` |
   | `#llvm.loop_annotation<unroll = <disable = true>>` | `!{!"llvm.loop.unroll.disable"}` |
   | `#llvm.loop_annotation<unroll = <full = true>>` | `!{!"llvm.loop.unroll.full"}` |
   | `#llvm.loop_annotation<unroll = <count = N>>` | `!{!"llvm.loop.unroll.count", i32 N}` |

   其中 `!llvm.loop.unroll.enable` 正是无 factor `#pragma unroll` 的等价物（#1242 req2 的目标语义）。

4. **两条 VPTO emission pipeline 结构一致**：均为 `createConvertSCFToCFPass()` → `createConvertControlFlowToLLVMPass()` → `translateModuleToLLVMIR`
   （`lib/PTO/Transforms/VPTOLLVMEmitter.cpp:14195-14200`、`lib/PTO/Transforms/VPTOCANN900LLVMEmitter.cpp:11875` 附近）。
   `prepareVPTOForEmission`（`tools/ptoas/ptoas.cpp:3252`）先于所有 emission 路径执行。

### 2.3 关键发现：上游 unroll 工具能力足够

`loopUnrollByFactor`（`llvm-workspace/llvm-project/mlir/lib/Dialect/SCF/Utils/Utils.cpp:364`）：

- 支持**动态 bounds**：自动生成 trip count 计算与 epilogue remainder loop；
- **live-out carry values 由 epilogue 正确穿线**：main loop results 作为 epilogue 的 init args，uses 自动替换（现有 pass 注释中"live-out 不能展开"的限制在这条路径上不存在）；
- `loopUnrollFull` 是 `loopUnrollByFactor(tripCount)` 的封装；
- 限制：要求正 step（动态 sign 未知时使用 `ceilDivPositive` 语义），factor 必须为正。

因此 #1000 阶段二中最重的工作（remainder loop 生成、loop-carried SSA 穿线）已由上游 util 完成，本方案只需实现 attr 驱动的调度与降级逻辑。

### 2.4 attr 存活风险排查

`prepareVPTOForEmission` 与 emission 之间唯一重建 `scf.for` 的 pass 是 `PTONarrowVPTOLoopCounters`，其实现保留了全部 attrs（`lib/PTO/Transforms/PTONarrowVPTOLoopCounters.cpp:115-117`，`newFor->setAttrs(forOp->getAttrs())`）。discardable attr 在 canonicalize/SCCP/CSE/LICM 中默认保留。方案中仍以 lit 测试锁定该行为。

---

## 3. 总体设计

### 3.1 语义矩阵

一套 attr 编码（`scf.for` 的 discardable attrs），两种消费方式（native 展开 / metadata 透传）：

| 前端写法 | `scf.for` attr | 语义 | Native 处理（Pass A） | 降级/透传（Pass B）→ LLVM metadata |
|---|---|---|---|---|
| （无 hint） | 无 | 现状不变 | 不处理 | 不处理 |
| `unroll="enable"` | `pto.unroll = "enable"` | 保留 loop，编译器 cost model 决定 full/partial | **不处理** | `llvm.loop.unroll.enable` |
| `unroll="disable"` | `pto.unroll = "disable"` | 禁止展开 | **不处理** | `llvm.loop.unroll.disable` |
| `unroll="full"` | `pto.unroll = "full"` | PTOAS 强制全展开 | `loopUnrollByFactor(tripCount)`，loop 消失 | 展开失败 → `llvm.loop.unroll.full` + remark |
| `unroll_factor=N` | `pto.unroll_factor = N`（i32） | PTOAS 按 N 展开 | `loopUnrollByFactor(N)`，生成 main + epilogue | 展开失败 → `llvm.loop.unroll.count = N` + remark |

约束：

- `unroll` 与 `unroll_factor` 互斥；
- `unroll_factor` 必须是 ≥ 1 的 Python 编译期整数常量；
- hint 不改变 `.carry(...)` 的 loop-carried value 语义；
- 未指定 hint 的 `pto.for_` / `range` 行为与 IR 完全不变。

边界情形（实现决策）：

- `unroll_factor=1` 无 native 意义，直接按 `llvm.loop.unroll.count = 1` 透传；
- 空 body 的 loop（`loopUnrollByFactor` 对其为 no-op）不 native 展开，hint 降级为 metadata；
- 带 `break`/`continue`/`else` 的 `pto.range(...)` loop 走 `scf.while` 降级路径，无法承载 hint，前端直接报错；
- LLVM 19 的 `convert-scf-to-cf` 不会把 `llvm.loop_annotation` 从 `scf.for` 传到 latch `cf.br`（上游新版本才支持），因此 Pass B 对带注解的 loop 自行完成 SCF→CF 降级并把注解挂到 latch 分支上；未注解的 loop 仍走上游转换；
- 挂在 latch 上的注解必须使用 ODS 裸名 `loop_annotation`（而非 `scf.for` 上的 discardable 名 `llvm.loop_annotation`）：MLIR→LLVM IR 翻译通过 `BrOp::getLoopAnnotationAttr()` 按裸名查找，`convert-cf-to-llvm` 原样转发分支属性（`op->getAttrs()`），因此 Pass B 在挂注解时改存裸名，之后的转换链无需特殊处理即可生成 `!llvm.loop` metadata；
- Pass B 的 SCF→CF 改写必须只作用于带注解的 loop（`applyOpPatternsAndFold` + `ExistingOps`，worklist 仅含带注解的 `scf.for`）：对整函数跑 `applyPatternsAndFoldGreedily` 会顺带常量折叠所有 op（即使没有 pattern 命中），在 VPTO emission 管线里会把 ub→llvm config word 的 `arith.ori` 链提前折叠掉，改变最终 LLVM IR 文本；
- `unroll_factor` 还须 ≤ 2^31−1：attr 编码为 signless i32，按有符号读回并写入 32 位 LLVM loop metadata，更大的值会回绕成负数。前端在构造期报错；后端同样坚持该契约——Pass A 对类型/范围不合约的 factor（如手写 IR 里的 i64 attr）不消费，Pass B 对其报错，避免截断成负的 `llvm.loop.unroll.count`；
- Pass A 的展开 fixpoint 不设轮数上限：每轮重新 walk 拾取外层展开克隆出的内层带注解 loop，直到某轮不再有任何变化为止。固定轮数上限会让超过该深度的嵌套 `full` hint 静默残留并降级为 metadata，违反 `full` 对静态循环强制 native 展开的契约。

### 3.2 重复展开的 by-construction 排除

#1000 担心"阶段一 CCE bypass 与阶段二 native unroll 对同一循环重复展开"。本设计中每个 loop 的 attr 只被消费一次：

- native 展开成功 → attr 被移除，Pass B 看不到该 loop；
- native 展开失败或不适用（`enable`/`disable`）→ attr 保留，由 Pass B 翻译成 metadata。

无需额外的阶段互斥配置。

**防二次展开**：factor 展开成功后给 epilogue loop 挂 `pto.unroll = "disable"`（等价 clang 对 remainder loop 打 `llvm.loop.unroll.disable` 的行为），由 Pass B 统一翻译；main loop 不再携带 unroll metadata。

### 3.3 架构总览

```
PTODSL 前端                          PTOAS 后端
─────────────                        ─────────────────────────────────────────────
pto.for_(..., unroll=...)            prepareVPTOForEmission:
pto.range(...)  (AST rewrite)          [Pass A] pto-unroll-loops        ← 消费 "full" / factor
        │                                SCCP / canonicalize / CSE      ← 折叠展开后的常量分支
        ▼                                ...（其余 VPTO 优化 pass）
scf.for {pto.unroll = "...",           ─────────────────────────────────────────────
         pto.unroll_factor = N}      VPTO emission pipeline（Beta1 / CANN900）:
        │                                ...
        └──────────────────────────▶     [Pass B] pto-lower-loop-hints  ← 翻译残留 attr → #llvm.loop_annotation，
                                         │                                并自行将带注解 loop 降为 CF、注解挂到 latch cf.br
                                         convert-scf-to-cf              ← 仅处理未注解 loop
                                         convert-cf-to-llvm             ← 自动保留到 llvm.br
                                         translateModuleToLLVMIR        ← 自动生成 !llvm.loop metadata
                                               │
                                               ▼
                                         BiSheng（cost model 决定 full/partial unroll）
```

---

## 4. PTODSL 前端设计

### 4.1 `pto.for_` 扩展

签名（`ptodsl/ptodsl/_control_flow.py:168`）：

```python
def for_(start, stop, *, step, unroll=None, unroll_factor=None):
    ...
```

- `unroll`：取值 `None | "enable" | "full" | "disable"`；
- `unroll_factor`：`None` 或 ≥ 1 的 `int`；
- 入口参数校验：非法取值 / 互斥冲突 / 非正整数 factor 抛 `TypeError` 或 `ValueError`，诊断信息可定位到调用点；
- hint 沿 `_ForBuilder` → `_ForCM.__enter__`（`_control_flow.py`）传递，`scf.ForOp` 创建后立即通过共享 helper（`_tracing/control_flow.py` 的 `apply_unroll_hint`，配套校验函数 `normalize_unroll_hint`）挂 attr：

```python
def apply_unroll_hint(for_op, unroll, unroll_factor):
    if unroll is not None:
        for_op.operation.attributes["pto.unroll"] = StringAttr.get(unroll)
    if unroll_factor is not None:
        for_op.operation.attributes["pto.unroll_factor"] = IntegerAttr.get(
            IntegerType.get_signless(32), unroll_factor)
```

### 4.2 三条建 loop 路径统一接入

hint 必须在以下所有创建 `scf.for` 的路径上一致生效，全部走 `apply_unroll_hint`：

1. **普通路径**：`_ForCM.__enter__`（`_control_flow.py`）；
2. **carry / session 路径**：`_CarryForCM` → `Session.begin_carry_loop`（`_tracing/session.py`）→ `build_carry_loop_frame`（`_tracing/control_flow.py`），在该处 ForOp 创建后挂 attr；`begin_carry_loop` 签名透传 hint 参数；
3. **tile template tracing 路径**：`_tile_template_tracing.py` 的 `for_` 同步扩展。

### 4.3 新增 `pto.range`（Python 原生 `for` 的 hint carrier）

```python
for i in pto.range(0, N, unroll="enable"):
    ...

for i in pto.range(0, N, unroll_factor=4):
    ...
```

- 只在 AST rewrite 场景下有意义；`pto.range(...)` 的 runtime 实现是一个立即报错的 marker（"pto.range 仅可用于被 AST rewrite 的 for 循环"），防止被当作普通 iterable 误用；
- `_ast_rewrite.py` 中扩展 `_range_triplet`（`:934`）与各 `visit_For` / `_rewrite_for`（`:1847`、`:1906`、`:2018`）：识别 `_is_pto_attr_call(stmt.iter, "range")`，抽取 start/stop/step 与 unroll kwargs，规范化为 `pto.for_(start, stop, step=step, unroll=..., unroll_factor=...)` 调用节点，与现有 `range(...)` → `pto.for_` 改写共用同一路径；
- `range`、`pto.range`、`pto.for_` 在无 hint 时生成的 IR 完全一致；
- `pto.static_range` 判别逻辑（`:542` 等）保持精确匹配，不得把 `pto.range` 误识别为 `static_range`；
- `break` / `continue` / loop-carried value 等现有原生控制流限制不变，并对 `pto.range` 提供同样的明确诊断；
- 嵌套循环只影响直接使用 `pto.range` 的层级。

---

## 5. PTOAS 后端设计

### 5.1 共享常量

将 `kUnrollAttrName` / `kUnrollFullValue`（现为 `PTOUnrollSIMTForPass.cpp:58-59` 私有）提升为共享常量（如 `include/PTO/IR/PTO.h` 或 Transforms 公共头），新增 `kUnrollFactorAttrName = "pto.unroll_factor"` 及 `"enable"` / `"disable"` 取值常量，供两个 pass 与文档共用。

### 5.2 Pass A：`PTOUnrollLoops`（重构 `PTOUnrollSIMTFor`）

- **新 pass 名**：`pto-unroll-loops`；保留 `pto-unroll-simt-for` 作为 alias（两个现存测试通过 `--mlir-print-ir-after=pto-unroll-simt-for` 引用，行为不变，零回归）；
- **位置不变**：`prepareVPTOForEmission` 内、SCCP/canonicalize/CSE 之前（`tools/ptoas/ptoas.cpp:3074`），保留 #838 "展开后常量分支被折叠"的收益；
- **处理逻辑**（walk `scf.for`）：
  - `pto.unroll = "full"`：静态 lb/ub/step、正 step、可计算 trip count → `loopUnrollByFactor(tripCount)` 全展开；成功则 loop 与 attr 一并消失；失败（动态 bound 等）则**保留 attr**，本 pass 不发诊断，由 Pass B 降级为 metadata 并 emit remark 说明降级原因；
  - `pto.unroll_factor = N`：先校验 attr 符合 signless i32 正数契约（`isValidUnrollFactorAttr`），不合约（如手写 IR 的 i64 attr）一律不消费、留给 Pass B 报错；合约时 `loopUnrollByFactor(N)`（动态 bounds 同样支持，上游 util 自动生成 epilogue 并穿线 live-out carry）；成功后移除 main loop 的 attr，并给 epilogue loop（若存在）挂 `pto.unroll = "disable"`；失败（非正 step 等）则保留 attr，由 Pass B 降级 + remark；
  - `pto.unroll = "enable"` / `"disable"`：直接跳过；
- **放开 SIMT-context 限制**：#838 的 auto-detect（trip count ≤ 64 自动展开）已移除，现存逻辑只认显式 attr——显式 attr 即用户意图，在非 SIMT 函数中静默忽略反而违反直觉。删除 `isInSIMTContext` 检查，pass 文档同步更新；
- **可选护栏**：full unroll 静态 trip count 超过阈值（默认如 1024，可用 pass option 调整）时 emit warning，防止 IR 体积爆炸；
- 非法 attr 组合不在本 pass 报错（留给 Pass B 统一诊断），本 pass 对无法识别的 attr 一律不触碰。

### 5.3 Pass B：`PTOLowerLoopHints`（新增）

- **pass 名**：`pto-lower-loop-hints`，func-level；
- **插入点**：两个 emitter pipeline 的 `createConvertSCFToCFPass()` 之前（`VPTOLLVMEmitter.cpp:14198`、`VPTOCANN900LLVMEmitter.cpp:11878`），保证翻译之后不再有任何 pass 有机会丢弃 attr；
- **翻译逻辑**（walk `scf.for`）：

  | 残留 attr | 生成的 annotation |
  |---|---|
  | `pto.unroll = "enable"` | `#llvm.loop_annotation<unroll = <disable = false>>` |
  | `pto.unroll = "disable"` | `#llvm.loop_annotation<unroll = <disable = true>>` |
  | `pto.unroll = "full"`（降级残留） | `#llvm.loop_annotation<unroll = <full = true>>` |
  | `pto.unroll_factor = N`（降级残留） | `#llvm.loop_annotation<unroll = <count = N>>` |

  翻译后移除 `pto.unroll` / `pto.unroll_factor` attr；
- **合并而非覆盖**：loop 上已有 `llvm.loop_annotation` 时（未来可能挂 vectorize 等其他 annotation），合并 `unroll` 字段而非替换整个 attr；
- **诊断**：未知 `pto.unroll` 字符串、factor < 1、attr 类型错误 → `emitError`，不静默忽略（满足 #1000 "不得静默改变语义" 的要求）；
- 之后的 cf→llvm→`!llvm.loop` metadata 由上游机制自动完成；带注解 loop 的 SCF→CF 降级由本 pass 自行完成（LLVM 19 的上游转换不传注解，见 3.1 边界情形），未注解 loop 仍走上游 `convert-scf-to-cf`。emitter 的全部改动就是在两条 emission pipeline 各插一行 `addNestedPass`，translation 层零改动。

### 5.4 与 #838 bug 规避语义的关系

- SIMT 内显式 `{pto.unroll = "full"}` 的常量循环仍在 SCCP/canonicalize 之前被强制全展开，`END @!P0` 规避路径不回归；
- 现存测试 `test/test_unroll_annotation.mlir` 与 `test/lit/vpto/unroll_inline_simt_sections.pto` 不需要修改（pass alias 保持名字与行为）。

---

## 6. 诊断与错误处理

| 场景 | 行为 |
|---|---|
| `unroll` 与 `unroll_factor` 同时指定 | PTODSL 前端 `ValueError` |
| `unroll` 取值非法 | PTODSL 前端 `ValueError`（列出合法取值） |
| `unroll_factor` 非正整数 / 非编译期常量 | PTODSL 前端 `TypeError` / `ValueError` |
| `unroll_factor` 超过 signless i32 上限（2^31−1） | PTODSL 前端 `ValueError` |
| 普通路径 `range(...)` / `pto.range(...)` 使用常量非正 step | PTODSL 前端 `PTODSLAstRewriteError`（负 step 仅带 break/continue 的 `pto._while` 路径支持） |
| `pto.range` 在非 AST-rewrite 上下文被调用 | `RuntimeError`（提示仅用于 rewrite 场景） |
| 手写 IR 中 `pto.unroll` 未知字符串 / attr 类型错误 | Pass B `emitError` |
| 手写 IR 中 `pto.unroll_factor` 类型/范围不合约（非 signless i32 或非正） | Pass B `emitError`（Pass A 对不合约 factor 一律不消费） |
| 手写 IR 中 `pto.unroll` 与 `pto.unroll_factor` 同时出现在一个 loop 上 | Pass B `emitError`（互斥） |
| `"full"` / factor native 展开失败 | Pass A 保留 attr（不发诊断）；Pass B emit remark + 降级 metadata，编译继续 |
| full unroll trip count 超过护栏阈值 | Pass A warning，仍执行展开 |

---

## 7. 测试计划

### 7.1 PTODSL 前端测试（`ptodsl/tests/`）

- `for_(..., unroll="enable")` / `unroll="full"` / `unroll_factor=4` 生成的 `scf.for` 携带正确 attr；
- `for i in pto.range(...)` 与 `with pto.for_(...)` 生成相同 IR（bounds / step / attr / SSA 语义逐字节一致）；
- `.carry(...)` 循环携带 hint 并正确编译（live-out carry 值正确）；
- `range` / `pto.range` / `pto.for_` 无 hint 时 IR 完全一致；
- 互斥参数、非法 `unroll` 取值、非整数 / 动态 factor 的稳定诊断；
- `pto.range` 的 start/stop/step 组合与 Python `range` 语义一致（单参数形式、负数界等）；普通路径（无 break/continue）下降为 `scf.for`，仅支持正 step，常量非正 step 由前端报错；负 step 仅在带 break/continue 的 `pto._while` 路径受支持；
- 嵌套循环中只有直接使用 `pto.range` 的层级携带 hint。

### 7.2 PTOAS lit 测试（`test/`、`test/lit/vpto/`）

**Native unroll（Pass A）**：

- factor 整除（无 epilogue）/ 不整除（epilogue 存在且 init args 正确）；
- trip count < factor、0 次、1 次迭代；
- 动态 upper bound 的 factor 展开；
- 带 live-out carry values 的 factor 展开（验证 epilogue 穿线）；
- 嵌套循环、循环内含条件分支与 memory side effect；
- epilogue loop 携带 `pto.unroll = "disable"`；
- 非 SIMT 上下文中的显式 attr 循环同样被展开（放开限制后的行为锁定）。

**Hint 透传（Pass B）**：

- `pto.unroll = "enable"` → FileCheck `#llvm.loop_annotation<unroll = <disable = false>>`；
- `--emit-vpto-llvm-dialect` 端到端：latch `llvm.br` 携带 annotation；
- 翻译后 `.ll` 检查 `!{!"llvm.loop.unroll.enable"}` / `count` / `full` / `disable`；
- 动态 bound 的 `"full"` 降级为 `llvm.loop.unroll.full` + warning；
- 非法 attr 组合产生 error 而非静默通过；
- attr 存活锁定：跑完整 `prepareVPTOForEmission` 后 annotation 仍在 loop 上。

**回归**：

- `test/test_unroll_annotation.mlir`、`test/lit/vpto/unroll_inline_simt_sections.pto` 不修改、不回归（#838 规避路径）；
- 无 hint 循环的 IR 与最终产物逐字节不变。

### 7.3 端到端验证

- SIMTVF sample（`test/samples/`）使用 `unroll="enable"` 编译到 A5：loop 保留、metadata 出现在 BiSheng 输入 `.ll` 中、kernel 正确编译；
- 带 `.carry()` 的循环与普通循环均完成 PTODSL → PTOAS → BiSheng 闭环。

---

## 8. 跨层同步清单（依据 `.claude/rules/cross-layer-sync.md`）

- [ ] PTODSL：`_control_flow.py`、`_tracing/control_flow.py`、`_tracing/session.py`、`_tile_template_tracing.py`、`_ast_rewrite.py`、`pto.py`（导出 `range`）
- [ ] 共享常量头文件（`pto.unroll` / `pto.unroll_factor` attr 名与取值）
- [ ] Pass A：`include/PTO/Transforms/Passes.td`（`pto-unroll-loops` + `pto-unroll-simt-for` alias）、`lib/PTO/Transforms/PTOUnrollSIMTForPass.cpp`（重构，视情况改名）
- [ ] Pass B：`Passes.td` + `lib/PTO/Transforms/PTOLowerLoopHintsPass.cpp`（新文件需 PR386 OAT.3 license header）
- [ ] `tools/ptoas/ptoas.cpp`（Pass A 替换）、两个 VPTO emitter（Pass B 插入）
- [ ] `docs/`：PTODSL user guide 增加 unroll hint 章节（语义矩阵、降级行为、与 `static_range` 的区别）
- [ ] 上述全部测试

## 9. 工作量评估

合并实现（enable hint + native full/factor unroll + #838 重构）相比"只做 enable hint"多出：Pass A 的 factor 分支与降级逻辑（约百余行，重活均在上游 `loopUnrollByFactor`）及对应测试。相比分两期实施，省掉"先纯透传、后插 native unroll 再处理阶段互斥"的返工——互斥问题在合并设计中天然不存在。

---

---

# PTODSL / PTOAS Loop Unroll Hint — Design Document

> Related issues:
> - [Issue #1242](https://github.com/hw-native-sys/PTOAS/issues/1242) Requirement 2 — `pto.for_` loop-unroll hint (`unroll="enable"`)
> - [Issue #1000](https://github.com/hw-native-sys/PTOAS/issues/1000) — Loop Unroll Hint support (incl. `pto.range`, factor unroll, two-phase plan)
> - [PR #838](https://github.com/hw-native-sys/PTOAS/pull/838) — `PTOUnrollSIMTForPass` (temporary workaround for a BiSheng AICore backend bug, refactored by this design)

---

## 1. Background and Motivation

### 1.1 Requirements

**Issue #1242 Requirement 2**: SIMTVF codegen needs the semantics of a no-factor `#pragma unroll` — keep the device-side loop and delegate the unrolling decision to the LLVM/BiSheng cost model (full or partial). PTODSL today offers only two extremes:

- `pto.static_range`: forces full unrolling at trace time, increasing trace/compile time, IR size, and register pressure;
- `pto.for_`: keeps the device loop but cannot carry any frontend unroll hint.

**Issue #1000**: the TileLang-PTO backend needs to lower `T.unroll(..., explicit=False)` into equivalent semantics (compiler-side unrolling rather than DSL frontend forced unrolling). The issue proposed a two-phase plan: phase 1 lets Bisheng/CCE perform the unrolling (hint pass-through); phase 2 unrolls natively in PTOAS.

**PR #838 (historical context)**: `PTOUnrollSIMTForPass` was a temporary workaround for a BiSheng AICore backend bug — after SCF→CF→LLVM lowering of `scf.for` + constant-condition `scf.if` in SIMTVF kernels, the AICore backend mishandles the `SimtEntry` calling convention and emits a predicated `END @!P0`. The workaround force-unrolls constant-trip-count loops annotated `{pto.unroll = "full"}` inside SIMT contexts, letting downstream SCCP + canonicalize eliminate the constant branches and produce branch-free straight-line code. The comment on issue #1000 explicitly calls this pass a temporary workaround to be refactored into a general unroll capability this time.

### 1.2 Goals

1. Support `unroll="enable"` on `pto.for_` / the new `pto.range`: keep the device loop and pass the hint through to the `!llvm.loop.unroll.enable` LLVM IR metadata, letting the BiSheng cost model choose the unrolling strategy (#1242 acceptance criteria).
2. Support `unroll="full"` and `unroll_factor=N`: PTOAS unrolls natively first; on failure, degrade to a compiler hint (#1000 phase-2 refactor).
3. Refactor `PTOUnrollSIMTFor` from a SIMT-only, full-only temporary pass into a general attribute-driven unroll pass, without regressing the #838 bug-workaround semantics.
4. Loops without any hint behave exactly as before.

### 1.3 Non-goals

- No changes to the LLVM/BiSheng unroll cost model;
- No fixed unroll factor guaranteed for the `enable` hint;
- No change to `pto.static_range`'s trace-time full-unroll semantics;
- No full unrolling of arbitrary dynamic loops (dynamic-bound `"full"` degrades to a metadata hint);
- No TileLang codegen changes.

---

## 2. Current-State Analysis

### 2.1 Existing unroll capability

`PTOUnrollSIMTForPass` (formerly `lib/PTO/Transforms/PTOUnrollSIMTForPass.cpp`, refactored into `PTOUnrollLoopsPass.cpp` by this design):

- Only handles `scf.for` explicitly annotated `{pto.unroll = "full"}` (`PTOUnrollSIMTForPass.cpp:58-66`);
- Only applies inside SIMT contexts (`pto.simt_entry` functions or inline `pto.section.simt` regions);
- Requires static lb/ub/step and a positive step; fully unrolls via `loopUnrollByFactor(tripCount)`;
- Runs in `prepareVPTOForEmission` before SCCP/canonicalize/CSE (`tools/ptoas/ptoas.cpp:3074`), so constant branches are folded downstream after unrolling.

PTODSL currently has **no public API** that sets the `pto.unroll` attribute — it appears only in hand-written `.pto` tests (`test/test_unroll_annotation.mlir`, `test/lit/vpto/unroll_inline_simt_sections.pto`).

### 2.2 Key finding: the vendored MLIR already has a complete loop-annotation delivery chain

No changes are needed in the LLVM/BiSheng interface layers:

1. **SCF→CF**: the vendored LLVM 19 (feature-vpto) `SCFToControlFlow` does **not** copy `llvm.loop_annotation` from `scf.for` to the latch `cf.br` (that support only exists in newer upstream versions; the LLVM 21 in `llvm-workspace` already has it).  Pass B therefore lowers annotated loops to control flow itself and attaches the annotation to the latch (see the edge cases in 3.1); unannotated loops still go through upstream `convert-scf-to-cf`;
2. **CF→LLVM**: `BranchOpLowering` / `CondBranchOpLowering` preserve all attributes onto `llvm.br` / `llvm.cond_br`
   (`llvm-workspace/llvm-project/mlir/lib/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.cpp:130-170`);
3. **MLIR→LLVM IR**: `translateModuleToLLVMIR` translates `#llvm.loop_annotation` on the latch into `!llvm.loop` metadata
   (`llvm-workspace/llvm-project/mlir/lib/Target/LLVMIR/LoopAnnotationTranslation.cpp:124-138`):

   | MLIR attr | LLVM IR metadata |
   |---|---|
   | `#llvm.loop_annotation<unroll = <disable = false>>` | `!{!"llvm.loop.unroll.enable"}` |
   | `#llvm.loop_annotation<unroll = <disable = true>>` | `!{!"llvm.loop.unroll.disable"}` |
   | `#llvm.loop_annotation<unroll = <full = true>>` | `!{!"llvm.loop.unroll.full"}` |
   | `#llvm.loop_annotation<unroll = <count = N>>` | `!{!"llvm.loop.unroll.count", i32 N}` |

   `!llvm.loop.unroll.enable` is exactly the equivalent of a no-factor `#pragma unroll` (the target semantics of #1242 req2).

4. **Both VPTO emission pipelines share the same structure**: `createConvertSCFToCFPass()` → `createConvertControlFlowToLLVMPass()` → `translateModuleToLLVMIR`
   (`lib/PTO/Transforms/VPTOLLVMEmitter.cpp:14195-14200`, near `lib/PTO/Transforms/VPTOCANN900LLVMEmitter.cpp:11875`).
   `prepareVPTOForEmission` (`tools/ptoas/ptoas.cpp:3252`) runs before every emission path.

### 2.3 Key finding: upstream unroll utilities are sufficient

`loopUnrollByFactor` (`llvm-workspace/llvm-project/mlir/lib/Dialect/SCF/Utils/Utils.cpp:364`):

- Supports **dynamic bounds**: automatically generates the trip-count computation and an epilogue remainder loop;
- **Live-out carry values are threaded correctly through the epilogue**: main-loop results become the epilogue's init args and uses are replaced automatically (the "live-out values cannot be unrolled" restriction noted in the current pass does not apply on this path);
- `loopUnrollFull` is a wrapper around `loopUnrollByFactor(tripCount)`;
- Restrictions: a positive step is required (`ceilDivPositive` semantics when the sign is not statically known); the factor must be positive.

The heaviest work of #1000 phase 2 (remainder-loop generation, loop-carried SSA threading) is therefore already done by the upstream utility; this design only needs attribute-driven scheduling and degradation logic.

### 2.4 Attribute-survival risk audit

The only pass between `prepareVPTOForEmission` and emission that rebuilds `scf.for` is `PTONarrowVPTOLoopCounters`, which preserves all attributes (`lib/PTO/Transforms/PTONarrowVPTOLoopCounters.cpp:115-117`, `newFor->setAttrs(forOp->getAttrs())`). Discardable attributes survive canonicalize/SCCP/CSE/LICM by default. A lit test locks this behavior in place.

---

## 3. Overall Design

### 3.1 Semantics matrix

One attribute encoding (discardable attrs on `scf.for`), two consumption paths (native unroll / metadata pass-through):

| Frontend syntax | `scf.for` attr | Semantics | Native handling (Pass A) | Degradation/pass-through (Pass B) → LLVM metadata |
|---|---|---|---|---|
| (no hint) | none | unchanged | not handled | not handled |
| `unroll="enable"` | `pto.unroll = "enable"` | keep the loop; compiler cost model decides full/partial | **not handled** | `llvm.loop.unroll.enable` |
| `unroll="disable"` | `pto.unroll = "disable"` | forbid unrolling | **not handled** | `llvm.loop.unroll.disable` |
| `unroll="full"` | `pto.unroll = "full"` | PTOAS forced full unroll | `loopUnrollByFactor(tripCount)`; the loop disappears | on failure → `llvm.loop.unroll.full` + remark |
| `unroll_factor=N` | `pto.unroll_factor = N` (i32) | PTOAS unrolls by N | `loopUnrollByFactor(N)`; main + epilogue loops | on failure → `llvm.loop.unroll.count = N` + remark |

Constraints:

- `unroll` and `unroll_factor` are mutually exclusive;
- `unroll_factor` must be a Python compile-time integer constant ≥ 1;
- hints do not change `.carry(...)` loop-carried-value semantics;
- `pto.for_` / `range` without a hint behave identically to today, IR included.

Edge cases (implementation decisions):

- `unroll_factor=1` has no native meaning and is forwarded as
  `llvm.loop.unroll.count = 1`;
- empty-body loops (a no-op for `loopUnrollByFactor`) are not unrolled
  natively; the hint degrades to metadata;
- `pto.range(...)` loops with `break`/`continue`/`else` lower through
  `scf.while` and cannot carry a hint, so the frontend reports an error;
- LLVM 19's `convert-scf-to-cf` does not propagate `llvm.loop_annotation`
  from `scf.for` to the latch `cf.br` (only newer upstream versions do), so
  Pass B lowers annotated loops to control flow itself and attaches the
  annotation to the latch branch; unannotated loops still go through the
  upstream conversion;
- the annotation on the latch must use the bare ODS name `loop_annotation`
  (not the discardable name `llvm.loop_annotation` used on `scf.for`): the
  MLIR-to-LLVM-IR translation looks it up via `BrOp::getLoopAnnotationAttr()`
  under the bare name, and `convert-cf-to-llvm` forwards branch attributes
  verbatim (`op->getAttrs()`), so Pass B stores the bare name on the latch and
  the rest of the conversion chain needs no special handling to emit
  `!llvm.loop` metadata;
- Pass B's SCF→CF rewrite must only touch annotated loops
  (`applyOpPatternsAndFold` + `ExistingOps`, worklist = annotated `scf.for`
  ops only): a function-wide `applyPatternsAndFoldGreedily` would also
  constant-fold every op it visits even with no pattern match, which in the
  VPTO emission pipeline folds the `arith.ori` chains of the ub-to-llvm
  config words ahead of time and changes the emitted LLVM IR text;
- `unroll_factor` must also be ≤ 2^31−1: the attribute is encoded as a
  signless i32, read back as a signed value, and forwarded into 32-bit LLVM
  loop metadata, so larger values wrap negative.  The frontend rejects them
  at construction time; the backend enforces the same contract — Pass A does
  not consume an out-of-contract factor (e.g. an i64 attribute in handwritten
  IR), and Pass B reports an error for it instead of truncating it into a
  negative `llvm.loop.unroll.count`;
- Pass A's unroll fixpoint has no round cap: each round re-walks the function
  to pick up annotated inner loops cloned by an outer unroll, and stops only
  when a round changes nothing.  A fixed round budget would silently leave
  hints behind on loops nested deeper than the budget, and a leftover `full`
  hint degrading to metadata would violate the forced native-unroll contract
  for static loops.

### 3.2 Double unrolling excluded by construction

#1000 worried about "phase-1 CCE bypass and phase-2 native unroll both unrolling the same loop". In this design each loop's attribute is consumed exactly once:

- native unroll succeeds → the attribute is removed; Pass B never sees the loop;
- native unroll fails or does not apply (`enable`/`disable`) → the attribute survives and Pass B translates it into metadata.

No extra phase-mutual-exclusion configuration is needed.

**Preventing re-unrolling**: after a successful factor unroll, the epilogue loop gets `pto.unroll = "disable"` (equivalent to clang attaching `llvm.loop.unroll.disable` to remainder loops), translated uniformly by Pass B; the main loop carries no further unroll metadata.

### 3.3 Architecture overview

```
PTODSL frontend                      PTOAS backend
─────────────                        ─────────────────────────────────────────────
pto.for_(..., unroll=...)            prepareVPTOForEmission:
pto.range(...)  (AST rewrite)          [Pass A] pto-unroll-loops        ← consumes "full" / factor
        │                                SCCP / canonicalize / CSE      ← folds constant branches
        ▼                                ... (remaining VPTO opt passes)
scf.for {pto.unroll = "...",           ─────────────────────────────────────────────
         pto.unroll_factor = N}      VPTO emission pipeline (Beta1 / CANN900):
        │                                ...
        └──────────────────────────▶     [Pass B] pto-lower-loop-hints  ← leftover attrs → #llvm.loop_annotation,
                                         │                                self-lowers annotated loops to CF and
                                         │                                attaches the annotation to the latch cf.br
                                         convert-scf-to-cf              ← unannotated loops only
                                         convert-cf-to-llvm             ← auto-preserved on llvm.br
                                         translateModuleToLLVMIR        ← auto-generated !llvm.loop metadata
                                               │
                                               ▼
                                         BiSheng (cost model decides full/partial unroll)
```

---

## 4. PTODSL Frontend Design

### 4.1 `pto.for_` extension

Signature (`ptodsl/ptodsl/_control_flow.py:168`):

```python
def for_(start, stop, *, step, unroll=None, unroll_factor=None):
    ...
```

- `unroll`: one of `None | "enable" | "full" | "disable"`;
- `unroll_factor`: `None` or an `int` ≥ 1;
- Entry-point validation: illegal values / mutually-exclusive conflicts / non-positive factor raise `TypeError` or `ValueError` with diagnostics that locate the call site;
- The hint flows through `_ForBuilder` → `_ForCM.__enter__` (`_control_flow.py`); the attribute is attached immediately after `scf.ForOp` creation via a shared helper (`apply_unroll_hint` in `_tracing/control_flow.py`, with `normalize_unroll_hint` as the companion validator):

```python
def apply_unroll_hint(for_op, unroll, unroll_factor):
    if unroll is not None:
        for_op.operation.attributes["pto.unroll"] = StringAttr.get(unroll)
    if unroll_factor is not None:
        for_op.operation.attributes["pto.unroll_factor"] = IntegerAttr.get(
            IntegerType.get_signless(32), unroll_factor)
```

### 4.2 Unified integration across all three loop-building paths

The hint must take effect consistently on every path that creates an `scf.for`, all going through `apply_unroll_hint`:

1. **Plain path**: `_ForCM.__enter__` (`_control_flow.py`);
2. **Carry / session path**: `_CarryForCM` → `Session.begin_carry_loop` (`_tracing/session.py`) → `build_carry_loop_frame` (`_tracing/control_flow.py`), attaching the attribute right after the ForOp is created there; `begin_carry_loop` forwards the hint parameters;
3. **Tile-template tracing path**: the `for_` in `_tile_template_tracing.py` is extended in lockstep.

### 4.3 New `pto.range` (hint carrier for native Python `for`)

```python
for i in pto.range(0, N, unroll="enable"):
    ...

for i in pto.range(0, N, unroll_factor=4):
    ...
```

- Meaningful only under AST rewrite; the runtime implementation of `pto.range(...)` is a marker that raises immediately ("pto.range may only be used in AST-rewritten for loops"), preventing misuse as a plain iterable;
- In `_ast_rewrite.py`, extend `_range_triplet` (`:934`) and the various `visit_For` / `_rewrite_for` sites (`:1847`, `:1906`, `:2018`): recognize `_is_pto_attr_call(stmt.iter, "range")`, extract start/stop/step and the unroll kwargs, and normalize to a `pto.for_(start, stop, step=step, unroll=..., unroll_factor=...)` call node, sharing the existing `range(...)` → `pto.for_` rewrite path;
- `range`, `pto.range`, and `pto.for_` produce identical IR when no hint is given;
- The `pto.static_range` detection (`:542` etc.) keeps exact matching and must never misidentify `pto.range` as `static_range`;
- Existing native-control-flow restrictions (`break` / `continue` / loop-carried values) are unchanged, with the same clear diagnostics for `pto.range`;
- Nested loops: only the levels directly using `pto.range` carry the hint.

---

## 5. PTOAS Backend Design

### 5.1 Shared constants

Promote `kUnrollAttrName` / `kUnrollFullValue` (currently private at `PTOUnrollSIMTForPass.cpp:58-59`) into a shared header (e.g. `include/PTO/IR/PTO.h` or a Transforms common header), and add `kUnrollFactorAttrName = "pto.unroll_factor"` plus the `"enable"` / `"disable"` value constants, shared by both passes and the docs.

### 5.2 Pass A: `PTOUnrollLoops` (refactor of `PTOUnrollSIMTFor`)

- **New pass name**: `pto-unroll-loops`; keep `pto-unroll-simt-for` as an alias (two existing tests reference it via `--mlir-print-ir-after=pto-unroll-simt-for`; behavior is unchanged, zero regression);
- **Position unchanged**: inside `prepareVPTOForEmission`, before SCCP/canonicalize/CSE (`tools/ptoas/ptoas.cpp:3074`), preserving the #838 benefit of folding constant branches after unrolling;
- **Handling logic** (walk `scf.for`):
  - `pto.unroll = "full"`: static lb/ub/step, positive step, computable trip count → fully unroll via `loopUnrollByFactor(tripCount)`; on success the loop and the attribute disappear; on failure (e.g. dynamic bounds) the attribute is **kept** — this pass emits no diagnostic, and Pass B degrades it to metadata with a remark explaining the degradation;
  - `pto.unroll_factor = N`: the attribute is first checked against the signless-i32 positive-factor contract (`isValidUnrollFactorAttr`); an out-of-contract attribute (e.g. an i64 in handwritten IR) is never consumed here and is left for Pass B to diagnose.  A valid factor is unrolled via `loopUnrollByFactor(N)` (dynamic bounds supported; the upstream utility generates the epilogue and threads live-out carries); on success remove the attribute from the main loop and attach `pto.unroll = "disable"` to the epilogue loop (if any); on failure (e.g. non-positive step) keep the attribute — Pass B degrades it with a remark;
  - `pto.unroll = "enable"` / `"disable"`: skipped;
- **Lift the SIMT-context restriction**: #838's auto-detection (auto-unroll for trip count ≤ 64) has already been removed; only explicit attributes remain — an explicit attribute is user intent, and silently ignoring it outside SIMT contexts would be counterintuitive. Remove the `isInSIMTContext` check and update the pass documentation;
- **Optional guardrail**: warn when a full-unroll static trip count exceeds a threshold (default e.g. 1024, tunable via pass option) to prevent IR explosion;
- Illegal attribute combinations are not diagnosed here (left to Pass B's unified diagnostics); this pass never touches attributes it does not recognize.

### 5.3 Pass B: `PTOLowerLoopHints` (new)

- **Pass name**: `pto-lower-loop-hints`, func-level;
- **Insertion points**: immediately before `createConvertSCFToCFPass()` in both emitter pipelines (`VPTOLLVMEmitter.cpp:14198`, `VPTOCANN900LLVMEmitter.cpp:11878`), guaranteeing no later pass can drop the attribute;
- **Translation logic** (walk `scf.for`):

  | Leftover attr | Generated annotation |
  |---|---|
  | `pto.unroll = "enable"` | `#llvm.loop_annotation<unroll = <disable = false>>` |
  | `pto.unroll = "disable"` | `#llvm.loop_annotation<unroll = <disable = true>>` |
  | `pto.unroll = "full"` (degraded leftover) | `#llvm.loop_annotation<unroll = <full = true>>` |
  | `pto.unroll_factor = N` (degraded leftover) | `#llvm.loop_annotation<unroll = <count = N>>` |

  The `pto.unroll` / `pto.unroll_factor` attrs are removed after translation;
- **Merge, don't overwrite**: if the loop already carries `llvm.loop_annotation` (e.g. vectorize annotations in the future), merge the `unroll` field instead of replacing the whole attribute;
- **Diagnostics**: unknown `pto.unroll` strings, factor < 1, or attribute type errors → `emitError`, never silently ignored (per #1000's "never silently change semantics" requirement);
- Everything after — cf→llvm and the `!llvm.loop` metadata translation — is handled by the upstream mechanism; the SCF→CF lowering of annotated loops is done by this pass itself (LLVM 19's upstream conversion does not propagate the annotation, see the edge cases in 3.1), while unannotated loops still go through upstream `convert-scf-to-cf`.  The only emitter change is one `addNestedPass` line in each emission pipeline; the translation layer is untouched.

### 5.4 Relationship to the #838 bug-workaround semantics

- Constant loops explicitly annotated `{pto.unroll = "full"}` inside SIMT are still force-unrolled before SCCP/canonicalize; the `END @!P0` workaround path does not regress;
- The existing tests `test/test_unroll_annotation.mlir` and `test/lit/vpto/unroll_inline_simt_sections.pto` need no modification (the pass alias keeps both name and behavior).

---

## 6. Diagnostics and Error Handling

| Scenario | Behavior |
|---|---|
| `unroll` and `unroll_factor` given together | PTODSL frontend `ValueError` |
| Illegal `unroll` value | PTODSL frontend `ValueError` (listing legal values) |
| `unroll_factor` not a positive integer / not a compile-time constant | PTODSL frontend `TypeError` / `ValueError` |
| `unroll_factor` exceeds the signless i32 limit (2^31−1) | PTODSL frontend `ValueError` |
| Constant non-positive step on the plain `range(...)` / `pto.range(...)` path | PTODSL frontend `PTODSLAstRewriteError` (negative steps are only supported on the break/continue `pto._while` path) |
| `pto.range` called outside an AST-rewrite context | `RuntimeError` (rewrite-only hint) |
| Hand-written IR with unknown `pto.unroll` string / wrong attr type | Pass B `emitError` |
| Hand-written IR with an out-of-contract `pto.unroll_factor` (not signless i32, or non-positive) | Pass B `emitError` (Pass A never consumes an out-of-contract factor) |
| Hand-written IR with both `pto.unroll` and `pto.unroll_factor` on one loop | Pass B `emitError` (mutual exclusion) |
| `"full"` / factor native unroll fails | Pass A keeps the attribute (no diagnostic); Pass B emits a remark + degrades to metadata; compilation continues |
| Full-unroll trip count exceeds the guardrail threshold | Pass A warning; unroll still proceeds |

---

## 7. Test Plan

### 7.1 PTODSL frontend tests (`ptodsl/tests/`)

- `for_(..., unroll="enable")` / `unroll="full"` / `unroll_factor=4` produce `scf.for` with the correct attributes;
- `for i in pto.range(...)` and `with pto.for_(...)` produce identical IR (bounds / step / attributes / SSA semantics, byte-for-byte);
- `.carry(...)` loops carry the hint and compile correctly (correct live-out carry values);
- `range` / `pto.range` / `pto.for_` without hints produce identical IR;
- Stable diagnostics for mutually-exclusive arguments, illegal `unroll` values, and non-integer / dynamic factors;
- `pto.range` start/stop/step combinations match Python `range` semantics (single-argument form, negative bounds, etc.); the plain path (no break/continue) lowers to `scf.for` and only supports a positive step — a constant non-positive step is rejected by the frontend; negative steps are only supported on the break/continue `pto._while` path;
- In nested loops, only the levels directly using `pto.range` carry the hint.

### 7.2 PTOAS lit tests (`test/`, `test/lit/vpto/`)

**Native unroll (Pass A)**:

- Factor divides trip count (no epilogue) / does not divide (epilogue present with correct init args);
- Trip count < factor, 0 iterations, 1 iteration;
- Factor unroll with a dynamic upper bound;
- Factor unroll with live-out carry values (verify epilogue threading);
- Nested loops, conditional branches and memory side effects inside the body;
- The epilogue loop carries `pto.unroll = "disable"`;
- Explicitly annotated loops outside SIMT contexts are also unrolled (locks in the lifted restriction).

**Hint pass-through (Pass B)**:

- `pto.unroll = "enable"` → FileCheck `#llvm.loop_annotation<unroll = <disable = false>>`;
- `--emit-vpto-llvm-dialect` end-to-end: the latch `llvm.br` carries the annotation;
- The translated `.ll` contains `!{!"llvm.loop.unroll.enable"}` / `count` / `full` / `disable`;
- Dynamic-bound `"full"` degrades to `llvm.loop.unroll.full` + warning;
- Illegal attribute combinations produce errors rather than silently passing;
- Attribute-survival lock: after the full `prepareVPTOForEmission`, the annotation is still on the loop.

**Regression**:

- `test/test_unroll_annotation.mlir` and `test/lit/vpto/unroll_inline_simt_sections.pto` unchanged and green (the #838 workaround path);
- Loops without hints produce byte-identical IR and final artifacts.

### 7.3 End-to-end validation

- A SIMTVF sample (`test/samples/`) using `unroll="enable"` compiled to A5: the loop is kept, the metadata appears in the BiSheng input `.ll`, and the kernel compiles correctly;
- Both `.carry()` loops and plain loops complete the PTODSL → PTOAS → BiSheng loop.

---

## 8. Cross-Layer Synchronization Checklist (per `.claude/rules/cross-layer-sync.md`)

- [ ] PTODSL: `_control_flow.py`, `_tracing/control_flow.py`, `_tracing/session.py`, `_tile_template_tracing.py`, `_ast_rewrite.py`, `pto.py` (export `range`)
- [ ] Shared-constants header (`pto.unroll` / `pto.unroll_factor` attr names and values)
- [ ] Pass A: `include/PTO/Transforms/Passes.td` (`pto-unroll-loops` + `pto-unroll-simt-for` alias), `lib/PTO/Transforms/PTOUnrollSIMTForPass.cpp` (refactored, possibly renamed)
- [ ] Pass B: `Passes.td` + `lib/PTO/Transforms/PTOLowerLoopHintsPass.cpp` (new files require the PR386 OAT.3 license header)
- [ ] `tools/ptoas/ptoas.cpp` (Pass A replacement), both VPTO emitters (Pass B insertion)
- [ ] `docs/`: PTODSL user guide gains an unroll-hint section (semantics matrix, degradation behavior, difference from `static_range`)
- [ ] All tests listed above

## 9. Effort Estimate

The combined implementation (enable hint + native full/factor unroll + #838 refactor) adds, compared to "enable hint only": the factor branch and degradation logic in Pass A (roughly a hundred lines — the heavy lifting lives in the upstream `loopUnrollByFactor`) plus the corresponding tests. Compared to a two-phase rollout, it avoids the rework of "pass metadata through first, then insert native unrolling and solve phase mutual exclusion later" — mutual exclusion does not exist as a problem in the combined design.
