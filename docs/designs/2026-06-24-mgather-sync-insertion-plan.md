# MGather 同步插入方案

> 目标：在 **暂时不能修改 pto-isa** 的前提下，由 ptoas 的 `InsertSync` pass 为
> `MGATHER` 建立正确的多 pipe 同步模型，并通过 `hiddenEvents` 避开 pto-isa
> 库内部仍然保留的固定 `event_id`。本文只覆盖 **ptoas 现有 lowering 真实可达**
> 的路径，与 pto-isa 源码一一对应。
>
> 本方案面向 **ptoas 现有 lowering 真实可达的 `MGather` 路径**，
> 采用「依赖驱动 phase 划分」思路：优先只声明每个 phase 的
> pipe + def/use，让求解器按真实 RAW/WAW 依赖自动推导同步；对 pto-isa
> 库内部仍保留的固定 `event_id`，额外用 `hiddenEvents` 进行避让。


---

## 1. 背景与问题

### 1.1 现状

`pto::MGatherOp` 实现了 `OpPipeInterface`，其 `getPipe()`（`include/PTO/IR/PTOOps.td:2874`）
只返回**单一** pipe：

| 场景 | dst 地址空间 | getPipe() 返回 |
|------|------------|---------------|
| GM → L1 | MAT (cube) | `PIPE_MTE2` |
| GM → UB @ A5 | VEC | `PIPE_V` |
| GM → UB @ A2/A3 | VEC | `PIPE_MTE2` |

`PTOIRTranslator::getOpPipeline()`（`lib/PTO/Transforms/InsertSync/PTOIRTranslator.cpp:655`）
直接把这个单一值填进 `CompoundInstanceElement::kPipeValue`，于是 `InsertSyncAnalysis`
把整个 mgather 当成**一个单 pipe 指令**，所有 operand 读写都挂在该 pipe 上。

### 1.2 为什么是错的

pto-isa 的 `MGATHER` 是一个 macro-like 库调用，内部跨多个 pipe（见 §2）。把
所有 def/use 挂在同一个 compound 节点上，`InsertSyncAnalysis` 无法区分各隐藏 pipe
的内部读写，导致：

- 内部 pipe 之间的顺序约束（如 S 读 idx → MTE2 做 DMA）对编译器不可见，库只能
  自己塞 `set_flag/wait_flag` 兜底；
- 外部 producer/consumer 与 mgather 隐藏 pipe 的依赖无法精确建模。

### 1.3 已有的正确范式：SyncMacroModel

仓库里已有为「多 pipe 宏」建模的成熟机制（`comm.tgather`，见
`test/lit/pto/issue706_tgather_hidden_mte_sync.pto`；`tscatter`/`tgather` 同类）。
核心思路：

1. 一条 op → 展开成多个 `SyncMacroPhase`（每 phase 一个 pipe + 该 phase 的 def/use 列表），
   挂同一个 `macroOpInstanceId`；
2. `hiddenEvents` 声明库内部占用的固定 event id，让 `SyncEventIdAllocation` 避让；
3. 之后 `InsertSyncAnalysis` 按 phase 间的真实 def/use 依赖自动插 set/wait。

`getSyncMacroModel()`（`lib/PTO/Transforms/InsertSync/SyncMacroModel.cpp:198`）的分发链
**当前没有 `MGatherOp` 分支** —— 这就是根因。

### 1.4 代表性故障案例：Issue #861

[PTOAS#861](https://github.com/hw-native-sys/PTOAS/issues/861) 报告 a2a3 上 `pto.mgather`
**row** 模式数据竞争：hang（AICore 507018）或 wrong values，同 case 跨运行在两者间翻转
（= race）。**Elem 模式通过**。

确证证据（设备验证，每 case 单独在重置后的设备上跑）：

- row FP32 leading / reversed idx：507018 hang
- row FP32 / FP16 random idx：wrong values（480/512 mismatched）
- row FP32 large（mem [128,64], 32 rows）：wrong values（1088/2048）
- elem FP32 / FP16 / INT32（含 valid<physical）：PASS
- 同一 row case 跨运行在 507018 与 wrong-values 间翻转 → race
- **手工给 `MGatherRowImpl` 加 `PtoSetWaitFlag<PIPE_MTE2, PIPE_S>()`，5 个失败 row case
  全部 PASS（5 fail → 0 fail）→ 唯一缺陷就是缺 `MTE2→S`，应由 ptoas 插**

根因（与 §1.2 一致）：MGATHER 在 **S pipe** 读 idx tile（`idxPtr[r]`）算 GM 地址，但
`getPipe()` 返回单一 `PIPE_MTE2`，求解器把 idx 读也当成 MTE2。idx 由 `pto.tload`（MTE2）
生产，于是求解器只插 `MTE2<->MTE2 pipe_barrier`，**永不插 `MTE2→S`**。S 读 idx 与 idx DMA
竞争 → 旧/垃圾索引 → wrong rows 或越界地址（507018）。

> **关键洞察**：`MGatherElemImpl` 之所以"能过"，仅因为 pto-isa 模板里**手工**塞了
> `PtoSetWaitFlag<PIPE_MTE2, PIPE_S>()`；`MGatherRowImpl` 漏了。长期来看，这类手工同步应由
> ptoas 接管；但在当前阶段 **pto-isa 不是本次修改范围**，因此需要在 ptoas 侧先补齐缺失同步，
> 同时把库内仍占用的固定 `event_id` 通过 `hiddenEvents` 预留出去，避免冲突。

---

## 2. pto-isa 源码分析与可达路径裁剪

源码位置：
- A5：`pto-isa/include/pto/npu/a5/MGather.hpp`
- A2/A3：`pto-isa/include/pto/npu/a2a3/MGather.hpp`

### 2.1 ptoas lowering 实际生成的调用形式

`PTOMGatherToMGATHER`（`lib/PTO/Transforms/PTOToEmitC.cpp:3070`）生成的 `MGATHER`
emitc 调用，模板参数**只有两类**，按位序：

| 模板参数位 | 内容 | 条件 |
|----------|------|------|
| 第 1 位 | `coalesceTok`（`Row`/`Elem`）| 始终存在（L3122） |
| 第 2 位（可选）| `gatherOobTok` | 仅当 `GatherOOB != Undefined`（L3124）|

实参：`{dst, mem, idx}` 或 `{dst, mem, idx, scratch}`（GM→L1 Elem 带 scratch）。

### 2.2 不可达路径的裁剪（关键）

| pto-isa 内部路径 | 是否可达 | 依据 |
|-----------------|---------|------|
| A5 GM→L1 Elem **SIMT exec**（`MGatherGm2L1ElemSimtImpl`，含 `set_intra_block/wait_intra_block(SyncId=2)` 跨核同步）| ❌ 不可达 | 该重载要求第 3 模板参数 `GatherExec::Simt`（`a5/MGather.hpp:601`），而 ptoas 全仓无 `GatherExec`/`simt` 字样，lowering 永远不传该参数，只走默认 Scalar exec 重载（`a5/MGather.hpp:574`）|
| A2A3 GM→UB **NZ table**（`MGatherRowNzImpl`/`MGatherElemNzImpl`）| ❌ 不可达 | `verifyMGatherMScatterMemOperand`（`lib/PTO/IR/PTO.cpp:3831`）对 partition_view 形式的 mem 强制 `Layout::ND`，NZ table 无法通过 verify |
| A5 GM→UB **1x1 标量**（`MGatherScalarImpl`，含 `set/wait(V→S)` + `set/wait(S→V)`）| ⚠️ 边缘可达 | 触发条件 `TileDst::ValidRow==1 && ValidCol==1`（`a5/MGather.hpp:564`）。ptoas 可构造 valid=1x1 的 dst tile，但属极端场景，**本期保守纳入建模**，避免漏同步 |

### 2.3 可达路径的数据流与 pipe（最终建模依据）

> 与本文建模原则一致：phase 划分**只声明各 pipe 真正读写的 buffer**，不手写同步对。
> 同步由 `InsertSyncAnalysis` 按跨 phase 的 RAW/WAW 依赖自动推导。

| # | 架构 | 场景 | pto-isa 实现 | pipe 划分（def / use） | pto-isa库内现有同步（当前保留，ptoas 需避让）|
|---|------|------|-------------|----------------------|----------------------------------|
| P1 | A5/A2A3 | GM→L1, Row | `MGatherGm2L1RowImpl` | phase0 S: def={} use={idx}；phase1 MTE2: def={dst} use={mem} | `set/wait(S→MTE2, EVENT_ID0)` |
| P2 | A5/A2A3 | GM→L1, Elem | `MGatherGm2L1ElemImpl` | phase0 S: def={} use={idx}；phase1 MTE2: def={dst} use={mem,scratch} | `set/wait(S→MTE2, EVENT_ID0)` |
| P3 | A5 | GM→UB, Row | `MGatherRowImpl`→SIMT | phase0 V: def={dst} use={mem,idx} | 无（纯 SIMT async_invoke）|
| P4 | A5 | GM→UB, Elem(非1x1) | `MGatherElemImpl`→SIMT | phase0 V: def={dst} use={mem,idx} | 无（纯 SIMT async_invoke）|
| P5 | A5 | GM→UB, Elem(1x1) | `MGatherScalarImpl` | phase0 V: def={} use={idx}；phase1 S: def={dst} use={mem} | 前 `set/wait(V→S)`；后 `set/wait(S→V)` |
| **P6** | **A2A3** | **GM→UB, Row(ND)** | `MGatherRowImpl` | **phase0 S: def={} use={idx}；phase1 MTE2: def={dst} use={mem}** | 入口 `V→S`、`MTE3→S`（**历史上缺 `MTE2→S`，即 #861 暴露的问题**）；出口 `S→MTE2`、`MTE2→V`、`MTE2→MTE3`、`S→V`、`S→MTE3` |
| P7 | A2A3 | GM→UB, Elem(ND) | `MGatherElemImpl` | **phase0 S: def={dst} use={idx,mem}**（单 phase，Elem 全程标量）| 入口 `V→S`、`MTE3→S`、`MTE2→S`；出口 `S→V`、`S→MTE2`、`S→MTE3` |

**P6 是 #861 暴露出的代表性失败场景**（粗体标注）。其修复就是 phase0 把 idx 挂给 S，让求解器对
「`tload`(MTE2) 产 idx → `mgather`(S) 读 idx」这条 RAW 自动插 `set/wait(MTE2→S)`。这条同步
不只服务于该 issue，而是 `MGather` 现有 lowering 路径正确同步建模的一部分。与此同时，`MGATHER` 库内部已经占用的固定 `event_id`
仍需通过 `hiddenEvents` 对求解器显式声明，避免与 ptoas 新插入的同步复用同一 id。

---

## 3. 同步插入方案（依赖驱动）

### 3.1 核心原则

**phase 依赖驱动 + hiddenEvent 资源避让并存**。理由：
- mgather 的跨 pipe 数据依赖，仍然尽量通过 **SSA 可见的 buffer**（idx/mem/dst/scratch）
  的 RAW/WAW 让求解器自动推导；
- 但当前 `pto-isa` 中仍保留了一批 `set_flag/wait_flag(EVENT_ID0)` 之类的固定同步，它们对
  ptoas 求解器不可见，如果不显式声明，编译器可能把同一个 pipe 对上的 `EVENT_ID0`
  再分配给新插入同步，造成冲突；
- 因此本期必须为这些库内固定事件号补 `hiddenEvents`，把对应资源从求解器可分配池中预留出去。

也就是说：**同步边由 def/use 推导，event 资源冲突由 hiddenEvents 规避**。这是当前约束下的
过渡方案；后续若 pto-isa 能配合删除库内同步，再退回到纯依赖驱动模型。

### 3.2 场景判定与建模

```cpp
std::optional<SyncMacroModel> getMGatherSyncMacroModel(pto::MGatherOp op) {
  const bool isGm2L1 = /* dst 为 MAT，同 getPipe() 内 getAddressSpace 判定 */;
  const PTOArch arch = getTargetArch(op.getOperation());
  const pto::Coalesce coalesce =
      op.getCoalesceAttr() ? op.getCoalesceAttr().getValue()
                           : (isRowCoalescedMGatherIndexType(op.getDst().getType(),
                                                            op.getIdx().getType())
                                  ? pto::Coalesce::Row : pto::Coalesce::Elem);

  SyncMacroModel model;

  // ========== P1/P2: GM -> L1 (Row + Elem, A5/A2A3 统一) ==========
  // S 读 GM idx 算地址 -> MTE2 copy_gm_to_cbuf(nd2nz) -> dst(L1)
  if (isGm2L1) {
    addPhase(model, PipelineType::PIPE_S,    /*def=*/{},            /*use=*/{op.getIdx()});
    addPhase(model, PipelineType::PIPE_MTE2, /*def=*/{op.getDst()}, /*use=*/{op.getMem()});
    addHiddenEvent(model, PipelineType::PIPE_S, PipelineType::PIPE_MTE2, {0});
    return model;
  }

  // ========== P3/P4: A5 GM -> UB, SIMT (Row / Elem 非1x1) ==========
  if (arch == PTOArch::A5 && !isElem1x1(op)) {
    addPhase(model, PipelineType::PIPE_V, /*def=*/{op.getDst()},
             /*use=*/{op.getMem(), op.getIdx()});
    return model;
  }

  // ========== P5: A5 GM -> UB, Elem 1x1 标量 ==========
  if (arch == PTOArch::A5 && isElem1x1(op)) {
    addPhase(model, PipelineType::PIPE_V, /*def=*/{},            /*use=*/{op.getIdx()});
    addPhase(model, PipelineType::PIPE_S, /*def=*/{op.getDst()}, /*use=*/{op.getMem()});
    addHiddenEvent(model, PipelineType::PIPE_V, PipelineType::PIPE_S, {0});
    addHiddenEvent(model, PipelineType::PIPE_S, PipelineType::PIPE_V, {0});
    return model;
  }

  // ========== P6: A2A3 GM -> UB, Row (ND) ==========
  if (arch != PTOArch::A5 && coalesce == pto::Coalesce::Row) {
    addPhase(model, PipelineType::PIPE_S,    /*def=*/{},            /*use=*/{op.getIdx()});
    addPhase(model, PipelineType::PIPE_MTE2, /*def=*/{op.getDst()}, /*use=*/{op.getMem()});
    addHiddenEvent(model, PipelineType::PIPE_V, PipelineType::PIPE_S, {0});
    addHiddenEvent(model, PipelineType::PIPE_MTE3, PipelineType::PIPE_S, {0});
    addHiddenEvent(model, PipelineType::PIPE_S, PipelineType::PIPE_MTE2, {0});
    addHiddenEvent(model, PipelineType::PIPE_MTE2, PipelineType::PIPE_V, {0});
    addHiddenEvent(model, PipelineType::PIPE_MTE2, PipelineType::PIPE_MTE3, {0});
    addHiddenEvent(model, PipelineType::PIPE_S, PipelineType::PIPE_V, {0});
    addHiddenEvent(model, PipelineType::PIPE_S, PipelineType::PIPE_MTE3, {0});
    return model;
  }

  // ========== P7: A2A3 GM -> UB, Elem (ND)  <-- 单 phase, 全标量 ==========
  if (arch != PTOArch::A5 && coalesce == pto::Coalesce::Elem) {
    addPhase(model, PipelineType::PIPE_S, /*def=*/{op.getDst()},
             /*use=*/{op.getIdx(), op.getMem()});
    addHiddenEvent(model, PipelineType::PIPE_V, PipelineType::PIPE_S, {0});
    addHiddenEvent(model, PipelineType::PIPE_MTE3, PipelineType::PIPE_S, {0});
    addHiddenEvent(model, PipelineType::PIPE_MTE2, PipelineType::PIPE_S, {0});
    addHiddenEvent(model, PipelineType::PIPE_S, PipelineType::PIPE_V, {0});
    addHiddenEvent(model, PipelineType::PIPE_S, PipelineType::PIPE_MTE2, {0});
    addHiddenEvent(model, PipelineType::PIPE_S, PipelineType::PIPE_MTE3, {0});
    return model;
  }

  return std::nullopt;
}
```

注册（`getSyncMacroModel` 末尾）：

```cpp
  if (auto mgather = dyn_cast<pto::MGatherOp>(op))
    return getMGatherSyncMacroModel(mgather);
```

### 3.3 求解器如何自动推导同步、并避开库内固定 event

以 **P6（A2/A3 GM→UB Row）** 为例，设 idx 由 `pto.tload`(MTE2) 生产：

```
[tload idx]  MTE2: def={idx}
[mgather p0] S:    use={idx}        <- RAW: idx, MTE2->S
[mgather p1] MTE2: def={dst} use={mem}
```

`InsertSyncAnalysis` 反扫 `mgather p0`(S) 时，命中 `tload`(MTE2) 对 idx 的 RAW（now.use
idx ↔ front.def idx），跨 pipe → 自动插 `set_flag(MTE2, S)` @ tload 后 + `wait_flag(MTE2, S)`
@ mgather p0 前。**这也是该类路径应当自动得到的同步**。随后 `hiddenEvents` 会把 `MGATHER`
库内部已经占用的 `EVENT_ID0` 从对应 pipe 对的可分配池里扣掉，确保新增这条 `MTE2→S`
同步不会撞到库里的固定事件号。

同理覆盖各场景：
- **P1/P2**：`tload`(MTE2) idx → `mgather p0`(S) 读 idx，插 `MTE2→S`；同宏内 p0(S)→p1(MTE2)
  顺序由 phase 顺序保证（无共享 buffer 时由 macroOpInstanceId 关联）。
- **P5**：`tload`(MTE2) idx → `mgather p0`(V) 读 idx，插 `MTE2→V`；p0(V)→p1(S) 由 idx 共享
  buffer 的 RAW（p0 use idx，但 p1 不读 idx——这里 p0/p1 间无共享 buffer，顺序靠 phase 排列
  + macroOpInstanceId，见 §3.4）。
- **P3/P4**：单 phase V，idx/mem/dst 都在 V，求解器对 idx 的 `MTE2→V` 自动推导。
- **P7**：单 phase S，`tload`(MTE2) idx → `mgather`(S)，插 `MTE2→S`。

### 3.4 同宏内相邻 phase 的顺序保证

`MakeMacroCompound`（`PTOIRTranslator.cpp:622`）为每个 phase 生成独立 compound，挂同一
`macroOpInstanceId`。相邻 phase（如 P6 的 p0 S、p1 MTE2）若无共享 buffer，标准 RAW/WAW 不
直接命中。但：

- **P6/P1/P2 的 p0→p1**：p0(S) 读 idx、p1(MTE2) 写 dst，两者操作不同 buffer，无 RAW。但 p0
  必须**先于** p1 完成（S 算完地址 MTE2 才能 DMA）——这是控制流顺序，由 `InsertSeqSync` 的
  反扫天然覆盖：反扫 p1(MTE2) 时向前遇到 p0(S)，同 pipe 才插 barrier，不同 pipe 且无内存
  依赖则不插。**这意味着 P1/P2/P6 的 p0→p1 顺序依赖"同宏内 S 与 MTE2 不共享 buffer 但语义
  上 S 先"——当前 `InsertSyncAnalysis` 对此无内存依赖时不插同步**。

  在当前过渡方案里，这部分不再作为“可选兜底”，而是直接由 `hiddenEvents` 预留
  `S→MTE2(EVENT_ID0)`，与 pto-isa 现状保持一致，避免求解器和库内部同步争用同一资源。

---

## 4. 实现落点

| 文件 | 改动 | 说明 |
|------|------|------|
| `lib/PTO/Transforms/InsertSync/SyncMacroModel.cpp` | 新增 `getMGatherSyncMacroModel()`（匿名 namespace）；`getSyncMacroModel()` 末尾加 `MGatherOp` 分支 | 核心，phase 建模 + hiddenEvent 资源避让 |
| `lib/PTO/Transforms/InsertSync/SyncMacroModel.cpp` | 复用已有 `addPhase` | 无需新增基础设施 |
| `include/PTO/IR/PTO.h` 的 `getTargetArch` | 已存在，直接用 | 无改动 |
| `lib/PTO/IR/PTO.cpp` `MGatherOp::getEffects` (L12541) | 保持现状 | phase 拆分后 def/use 由 `SyncMacroPhase` 显式提供，覆盖 getEffects |
| `lib/PTO/Transforms/InsertSync/PTOInsertSync.cpp:49` `hasGatherScatterLikeOps` | 已含 `MGatherOp`，跳过 `RemoveRedundantSync` | 无改动（correctness-first）|
| `lib/PTO/Transforms/InsertSync/PTOIRTranslator.cpp:642` `UpdateMacroOpInfo` | 已支持 macro phase 展开 | 无改动 |
| pto-isa `npu/a5/MGather.hpp` | **本期不改** | 继续保留库内固定 `event_id` 同步，由 ptoas `hiddenEvents` 避让 |
| pto-isa `npu/a2a3/MGather.hpp` | **本期不改** | 继续保留库内固定 `event_id` 同步，由 ptoas `hiddenEvents` 避让 |

### 4.1 `__PTO_AUTO__` 宏的配合

`PtoSetWaitFlag` 在 `__PTO_AUTO__` 下是 no-op，但当前问题不在于 auto 路径本身，而在于：
**pto-isa 源码层面仍保留了一组固定 `event_id` 的同步语句，且本次无法删除**。因此本期策略不是
依赖 `__PTO_AUTO__` 去“假装它们不存在”，而是明确把这些固定事件号反映到 ptoas 的
`hiddenEvents` 中，让求解器知道哪些 pipe 对上的哪些 id 已经被库占用。

---

## 5. 验证与回归

### 5.1 代表性设备验证矩阵（含 #861 场景）

| case | 场景 | 修复前 | 修复后预期 |
|------|------|-------|----------|
| row FP32 leading/reversed idx | P6 | 507018 hang | PASS |
| row FP32/FP16 random idx | P6 | wrong (480/512) | PASS |
| row FP32 large (mem [128,64], 32 rows) | P6 | wrong (1088/2048) | PASS |
| elem FP32/FP16/INT32 (含 valid<physical) | P7 | PASS | PASS（不回归）|

### 5.2 lit 测试

| 测试文件 | 覆盖路径 | 期望 |
|---------|---------|------|
| `test/lit/pto/mgather_gm2l1_sync_a5.pto`（现有）| P1 | 保持 `set_flag(PIPE_MTE2, PIPE_MTE1)`（mgather 作 MTE2 producer 喂下游 TMOV/MTE1）|
| 新增 `mgather_gm2l1_elem_sync_a5.pto` | P2 | dst 喂 TMOV，见 `set_flag(PIPE_MTE2, PIPE_MTE1)` |
| 新增 `mgather_gm2ub_a5_simt_sync.pto` | P3/P4 | A5 SIMT，下游 V op 消费 dst，见 V barrier 或 `set_flag(PIPE_V,...)` |
| 新增 `mgather_gm2ub_a5_scalar_1x1_sync.pto` | P5 | 见 `set/wait(MTE2→V)`（idx 由 tload 产）|
| 新增 `mgather_gm2ub_a2a3_row_sync.pto` | **P6** | **见 `set_flag(PIPE_MTE2, PIPE_S)` + `wait_flag(PIPE_MTE2, PIPE_S)`** —— A2/A3 row 路径的关键同步点；同时不与库内固定 `EVENT_ID0` 冲突 |
| 新增 `mgather_gm2ub_a2a3_elem_sync.pto` | P7 | 见 `set_flag(PIPE_MTE2, PIPE_S)` |
| `test/lit/pto/issue706_tgather_hidden_mte_sync.pto`（现有）| 范式参照 | 不回归 |

### 5.3 P6 回归的精确 FileCheck（A2/A3 row 关键检查）

```
pto.tload ... -> idx   // MTE2 产 idx
pto.mgather ins(%mem, %idx ...) {coalesce = row}   // A2A3 P6
```

修复后期望：

```
TLOAD
set_flag(PIPE_MTE2, PIPE_S)     // <- 历史缺失的关键同步, 现由 ptoas 自动插
wait_flag(PIPE_MTE2, PIPE_S)
MGATHER<pto::Coalesce::Row>
```

---

## 6. 后续优化（非本期）

1. **删除 pto-isa 库内同步，回归纯依赖驱动**：待具备修改 `pto-isa` 的条件后，删除
   `MGATHER` 内部固定 `event_id` 同步，再移除本方案里为其保留的 `hiddenEvents`。
2. **A5 SIMT exec 路径**：若未来 ptoas lowering 支持 `GatherExec::Simt`，需扩展
   `SyncMacroModel::hiddenEvents` 支持跨核 `set_intra_block/wait_intra_block`（SyncId=2）。
   当前不可达，不实现。
3. **A2A3 NZ table 路径**：若未来放宽 `verifyMGatherMScatterMemOperand` 允许 NZ table，
   再补 `MGatherRowNzImpl`/`MGatherElemNzImpl` 的同步建模。
4. **hidden event 收敛**：若后续确认某些库内同步在 `__PTO_AUTO__` 真实产物中不会生效，
   可缩减对应 pipe 对上的 `hiddenEvents` 声明，释放更多 event 资源。

---

## 7. 风险与缓解

| 风险 | 缓解 |
|------|------|
| ptoas 新插入同步与 pto-isa 库内固定 `event_id` 冲突 | 通过 `hiddenEvents` 显式预留对应 pipe 对上的 `EVENT_ID0`，禁止求解器复用 |
| 同宏相邻 phase 无共享 buffer 时漏插顺序同步（§3.4）| 保留与库现状一致的 `hiddenEvents` 约束；优先用设备验证覆盖 P1/P2/P6 |
| hidden event 预留过多，压缩可分配 event 资源 | 仅为真实可达路径和库中真实存在的固定事件号建模；必要时按 pipe 对精细化裁剪 |
| A5 1x1 标量（P5）边缘场景漏建模 | 本期已纳入（§3.2 P5），保守覆盖 |
| 求解器对 GM idx 的依赖追踪不精确（GM root 别名）| `buffer2MemInfoMap` 已登记 GM tensor为 GM root；`MemoryDependentAnalyzer` 对 GM 做 may-alias 保守判定，不会漏 RAW |
