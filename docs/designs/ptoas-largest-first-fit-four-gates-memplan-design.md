# PTOAS Largest-First-Fit 与四道冲突闸门内存规划设计

## 1. 总体方案

PTOAS 当前保留两套 local memory planner：

- legacy memplan：默认启用，保留旧版 SPEC_LEVEL_0/1/2 三级投机复用，内存不够时回滚的策略。
- modern memplan：通过 `--plan-memory-impl=modern` 显式启用，当前实现Largest-First-Fit 与四道冲突闸门内存规划设计

PTOAS 当前已经具备 largest-first 风格的规划开关：`--plan-memory-order-by-size`。该选项会让 planner 在同一 AddressSpace 内优先处理更大的 buffer。当用户显式选择 `--plan-memory-impl=modern` 且未显式指定 `--plan-memory-order-by-size` 时，modern memplan 默认开启该排序；legacy memplan 仍保持默认关闭。

当前 `pto.alloc_tile` 的 lowering 路径保持 tile-native：

```text
pto.alloc_tile(no addr)
  -> PTOViewToMemref 透传，不转换成 memref.alloc
  -> pto-plan-memory 收集为 local allocation root
  -> modern/legacy memplan 按 level 校验并规划 local addr
  -> 直接给 pto.alloc_tile 补常量 addr
  -> 后续 pto.t* tile op 继续使用 !pto.tile_buf 形态
```

用户显式写 `pto.alloc_tile addr` 时，level 语义仍由 memplan 校验：level1/level2 禁止显式 addr，level3 要求显式 local addr。也就是说，`pto.alloc_tile` 不再通过 `memref.alloc -> pto.bind_tile` 这类中间链路表达地址；普通 local tile allocation 由 memplan 直接把常量地址回写到 `pto.alloc_tile addr`。历史 `pto.pointer_cast` bridge 已删除，不再作为 memplan materialize 结果。




### 1.1 当前实现状态

| 能力 | 状态 | 说明 |
| --- | --- | --- |
| modern planner | 已实现 | `--plan-memory-impl=modern` |
| largest-first | 已实现 | modern 未显式指定时默认打开；cube local space 例外 |
| 生命周期、phi family、target hazard、semantic no-alias | 已实现 | 见第 3 章四道闸门 |
| MTE3 -> MTE2 ranking | 已实现 | Phase 1，使用确定性优先级 |
| capacity escalation | 已实现 | level 2 -> 1 -> 0 逐档整趟重算 |
| PIPE_V、bank conflict、hot cluster | 预留 | 当前不参与 candidate ranking |
| pipeline stage load conflict | 预留 | 显式 multi-buffer 仍由 slot 分离保证 |

legacy planner 的默认行为不由本文改变。


## 2. 设计目标与非目标

### 2.1 设计目标

本设计目标如下：

- 复用 PTOAS 已有 `--plan-memory-order-by-size` largest-first 能力。
- 将复用判定收敛为统一的 `canShare(a, b)` 谓词。
- 引入统一的扁平冲突闸门模型，避免不安全复用。
- 保持按 AddressSpace 独立规划。
- 不做性能驱动的投机回滚与 per-entry 降级；容量兜底仅用**溢出触发、单调收敛、有限档位**的确定性全局退让（见“容量退让：单调优先档位”节），不撤销部分已提交规划、不需防重入补丁。
- 性能优化采用**确定性优先级策略**替代加权打分 cost model：在四道闸门正确性之上，按固定优先级依次规避冲突，最优先 pipeline conflict，其次 bank conflict，层间严格字典序、不做魔数加权。
- 分阶段落地：Phase 1 只做 MTE pipeline conflict 规避，Phase 2 再引入 bank conflict；PIPE_V 同 pipe 共址与 hot cluster 聚集暂不纳入。
- 保持 legacy memplan 默认行为不变。



### 2.2 非目标

本设计不包含以下内容：

- 不修改 legacy memplan 的 StorageEntry / SPEC_LEVEL_1 / SPEC_LEVEL_2 逻辑。
- 不在 legacy memplan 中实现四道闸门。
- 不恢复旧版 per-entry 逐级降级与部分撤销式投机回滚（新版仅保留溢出触发、整趟重算的单调档位退让）。
- 不把 modern memplan 作为默认实现。
- 不在本阶段实现跨函数、跨 module 的全局内存规划。




## 3. 正确性模型（四道闸门）

四道闸门是扁平 AND 关系，任意一道闸门失败，两个 root 就不能进入同一个 `ReuseGroup`，也就不能复用同一个 base offset。

在 root/alias 归约后，复用合法性可以抽象为：

```text
canShare(a, b) =
    sameAddressSpace(a, b)
    && aliasAndShapeAreRepresentable(a, b)
    && lifetimeCompatibleOrBranchExclusive(a, b)
    && !targetHazard(a, b)
    && !semanticNoAlias(a, b)
```

`fits` 是 candidate 的容量可行性检查，不是可以放宽的性能条件；它在 candidate ranking
中优先于性能层，并在每个 capacity level 完成后再次检查。

### 3.1 Root 与 alias、生命周期

**目的：** 禁止两个运行时可能同时存活的 local root 复用同一块物理地址。

基础规则：

```text
如果 a 和 b 生命周期重叠，则不能复用。
如果 a.lastUse == b.def 或 b.lastUse == a.def，则认为 touching，可继续检查其它闸门。
```

PTOAS 现有 `lifetimesOverlap` 可扩展为：

```text
overlap = !(a.lastUse <= b.def || b.lastUse <= a.def)
```

`<=` 表示 touching 不算生命周期冲突。

**适用场景 sample：普通直线代码中的临时 buffer 复用。**

```text
%a = alloc vec[1024]
use(%a)              // %a.lastUse

%b = alloc vec[1024] // %b.def
use(%b)
```

如果 `%a.lastUse <= %b.def`，闸门 1 允许 `%a` 和 `%b` 进入后续闸门。只要其它闸门也通过，二者可以复用同一个 offset：

```text
%a.offset = 0
%b.offset = 0
```

反例：

```text
%a = alloc vec[1024]
%b = alloc vec[1024]
use(%a)
use(%b)
```

此时 `%a` 和 `%b` 的生命周期重叠，闸门 1 直接失败，必须分配不同 offset。

### 3.2 phi family

**目的：** 处理互斥控制流分支，避免过度保守的 liveness extension 阻止本来安全的复用。

场景：

```text
%r = scf.if %cond -> tile {
  scf.yield %a
} else {
  scf.yield %b
}
```

`%a` 和 `%b` 在程序序生命周期上可能被扩展到 `%r` 的最后使用点，看起来重叠；但运行时两个分支互斥，不会同时存在。因此同一 phi family 的 yield source 可以豁免闸门 1 的生命周期冲突。

需要收集：

```text
phiFamilyIds[root] = set<familyId>
```

规则：

- 同 family 且无其它非 family 真重叠成员：允许共享。
- 如果混入外部 live root 或非 family 真重叠 root：不豁免。

实现建议：

```text
Gate2_LifetimeAndPhi(a, b):
  if !lifetimeOverlap(a, b):
    return true

  if samePhiFamily(a, b):
    return true

  return false
```

因此闸门 2 在实现上可以嵌入闸门 1，也可以保留独立函数名但由闸门 1 调用。

**适用场景 sample：if/else 分支内 local root 互斥复用。**

```text
%r = scf.if %cond -> tile {
  %then_buf = alloc vec[1024]
  produce(%then_buf)
  scf.yield %then_buf
} else {
  %else_buf = alloc vec[1024]
  produce(%else_buf)
  scf.yield %else_buf
}

consume(%r)
```

如果简单地把 `%then_buf` 和 `%else_buf` 的生命周期都延伸到 `consume(%r)`，二者看起来重叠，闸门 1 会失败。但运行时 then/else 互斥，不会同时分配这两个 branch-local root。因此它们属于同一个 phi family 时，可以共享同一 offset：

```text
%then_buf.offset = 0
%else_buf.offset = 0
```

反例：

```text
%outer = alloc vec[1024]

%r = scf.if %cond -> tile {
  %then_buf = alloc vec[1024]
  use(%outer)
  scf.yield %then_buf
} else {
  %else_buf = alloc vec[1024]
  use(%outer)
  scf.yield %else_buf
}
```

`%then_buf` 和 `%else_buf` 之间可以因为 phi family 互斥而复用；但它们不能借这个豁免和 `%outer` 这种外部 live root 错误复用。


### 3.3 target-specific load/tpop hazard

**目的：** 表达特定 target 上的局部硬件/后端 hazard。它不是通用生命周期问题，而是某些 load-derived buffer 与 consumes-tpop writer 在 touching 点复用会触发目标相关错误。

参考 PyPTO 的 Ascend910B split-AIV hazard。PTOAS 中设计为 target-gated 闸门：

```text
如果目标架构/后端不存在该 hazard，则闸门恒通过。
如果存在该 hazard，则禁止 load-derived buffer 与 consumes-split-tpop 的 writer 在 touching 点复用。
```

当前 PTOAS modern memplan 的启用条件：
A5 上该闸门恒通过。A3 上也只有识别到 split tpop 派生值时才可能触发。

#### 设计原理

这道闸门保护的是一种普通静态生命周期分析看不到的 target-specific touching 复用风险。普通 memplan 只看 allocation root 的 def/use 区间：如果一个 load-derived buffer 的最后一次使用正好等于另一个 writer output 的写入点，生命周期上属于 touching，通常可以复用同一个 offset。但在特定 target 上，`tload/tprefetch` 产生的 load-derived buffer 与 consuming split-tpop 的 writer 可能在后端流水或指令调度中存在更细粒度的运行时 overlap；如果二者复用同一地址，writer 的写入可能破坏 load-derived input 在该 target 上仍然需要保持稳定的内容。

因此该闸门不把问题抽象成通用“生命周期重叠”，而是显式识别一类 target hazard：

```text
load-derived input 的 last use
touching
consumes split-tpop 的 writer output 写入
```

只有当目标架构确认存在该 hazard 时才禁止复用。当前 PTOAS 中 A3 开启该闸门，A5 恒通过，避免把 A3 的局部硬件/后端约束错误扩散成所有 target 的通用内存规划规则。

实现上使用 writer op index，而不是 allocation root 的 def index，原因是 hazard 发生在“某个 writer op 同时读取 load-derived input 和 split-tpop-derived value，并写出 DPS output”的这个操作点。allocation root 的 def 只表示 buffer 被声明出来，不一定等于真正发生写入和触发 target hazard 的位置；用 writer op index 可以把 touching 条件精确绑定到产生风险的 op 上。

相关事实拆开收集：

```text
loadDerivedRoots:
  标记来自 tload/tprefetch 的 DPS dst root

split-tpop-derived values:
  标记 split tpop 结果、split tpop tile operand 以及它们经过 alias/view op 派生出的 value

tpopConsumerRoots:
  标记同时读取 load-derived root 和 split-tpop-derived value 的 DPS writer output root

tpopConsumerWriteIndices:
  记录上述 writer op 的 op index，用于判断是否正好在 touching 点复用
```

这种拆分可以避免过度保守：不是所有 load buffer 都禁止复用，也不是所有 tpop 相关 buffer 都禁止复用，只有“load-derived input 的最后使用点”碰到“consumes split-tpop writer 的写入点”时才失败。

当前 PTOAS modern memplan 对 DPS output root 使用 writer-def liveness：如果某个 DPS output 是 pure overwrite，即该 output 没有被 memory effects 标记为 Read，则它的 `allocIndex` 会从 `pto.alloc_tile` / `memref.alloc` 收缩到第一次有效 writer op。这样 load-derived input 的 `freeIndex` 与 writer output 的 `allocIndex` 可以形成 touching，是否允许复用继续由 target-specific load/tpop hazard 和 op semantic no-alias 闸门判断。若 output 需要旧值，例如 read-modify-write / accumulate 语义，则保持 allocation-start 的保守生命周期。

需要收集：

```text
loadDerivedRoots
tpopConsumerRoots
tpopConsumerWriteIndices
targetHazardEnabled
```

判定：

```text
input.root in loadDerivedRoots
&& writer.root in tpopConsumerRoots
&& input.lastUseIndex in tpopConsumerWriteIndices[writer.root]
```

双向检查：

```text
hazard(a, b) || hazard(b, a)
```

**适用场景 sample：target-gated touching 复用禁止。**

```text
%load_buf = alloc vec[1024]
load_or_tpop_producer(%load_buf)
last_use(%load_buf)

%writer_dst = alloc vec[1024]
writer_consumes_tpop(...) outs(%writer_dst)
```

从普通生命周期看，`%load_buf.lastUse == %writer_dst.def`，属于 touching，可以复用。但如果 target 标记了：

```text
%load_buf in loadDerivedRoots
%writer_dst in tpopConsumerRoots
tpopConsumerWriteIndices[%writer_dst] contains writer op index
targetHazardEnabled = true
```

则闸门 3 失败，二者不能复用。若目标没有该 hazard，闸门 3 恒通过，不影响复用。

当前实现的 fact 来源：

```text
loadDerivedRoots:
  pto.tload / pto.tprefetch 的 DPS dst root

split tpop derived value:
  A3 上 split != 0 的 pto.tpop_from_aic result
  A3 上 split != 0 的 pto.tpop tile operand
  以及从这些 value 经过 bind_tile / slot_marker / cast / subview / select 等 alias/view op 派生出的 value

tpopConsumerRoots:
  某个 DPS writer 同时读取 split tpop derived value 和 load-derived root 时，
  记录该 writer 的 DPS output root
```


### 3.4 op semantic no-alias

**目的：** 表达“从生命周期看可以复用，但从 op 语义看不能 alias”的约束。闸门 4 对应 PyPTO 的 inplace 机制：`not_inplace_safe()` 与 `forbid_output_alias(i)`，并覆盖 PTOAS legacy memplan 中已有的 scratch-output conflict。

在 modern memplan 中，闸门 4 不应再只叫笼统的 `semantic conflict`，而应建模为一张明确的 forbid-alias side table：

```text
forbidAlias[root] = forbiddenRootSet
```

这里的 `root` 是 plannable local allocation root，而不是任意 SSA value。收集时需要先通过 `valueToRoots` 把 operand、view、alias value 归约到 root，再记录 root 与 root 之间的禁止复用关系。

#### 3.4.1 scratch-output conflict

PTOAS legacy memplan 的 semantic conflict 主要就是 scratch-output conflict。modern memplan 应继续覆盖这类场景：

```text
op implements PTO_DpsInitOpInterface
dpsInits = op outs(...)

effects = MemoryEffectOpInterface::getEffects(op)
scratchOperands =
  Write operand
  && operand in op operands
  && operand not in dpsInits

for scratch in scratchOperands:
  for dst in dpsInits:
    forbidAlias[root(scratch)].insert(root(dst))
    forbidAlias[root(dst)].insert(root(scratch))
```

**适用场景 sample：`tmp` scratch 不能和 output 复用。**

```text
%src = alloc vec[1024]
%tmp = alloc vec[1024]
%dst = alloc vec[1024]

pto.ttrans ins(%src, %tmp) outs(%dst)
```

`%tmp` 是 op 执行过程中的 scratch workspace，`%dst` 是最终 output。即使未来 liveness 把 `%dst` 看作在 op 上定义、从而让 `%tmp.lastUse == %dst.def` 成为 touching，二者也不能复用，否则 scratch 写入可能覆盖 output。闸门 4 记录：

```text
forbidAlias[%tmp].insert(%dst)
forbidAlias[%dst].insert(%tmp)
```

#### 3.4.2 not_inplace_safe

PyPTO 中 `not_inplace_safe()` 表示该 op 不能做 `src == dst` 的 inplace 执行。映射到 PTOAS 时，规则是：

```text
if opPolicy.notInplaceSafe:
  for operand in op operands excluding dpsInits:
    for dst in dpsInits:
      forbidAlias[root(operand)].insert(root(dst))
      forbidAlias[root(dst)].insert(root(operand))
```

典型 op 包括：

```text
pto.ttrans
pto.tgather
pto.tands / pto.tors / pto.txors
pto.tfillpad // dst physical shape is larger than src
pto.tfmod / pto.tfmods
pto.trecip / pto.trsqrt
pto.trowmax / pto.trowmin / pto.trowsum / pto.trowprod
pto.trowargmax / pto.trowargmin
pto.tcolargmax / pto.tcolargmin
pto.tsort32 / pto.tmrgsort
```

其中 `pto.tands` / `pto.tors` / `pto.txors` 和推导为 expand lowering 的 `pto.tfillpad` 是 PTOAS 侧额外保守标记的 non-inplace-safe op。它们虽然不是 scratch-output conflict，但后端/ISA 语义没有明确承诺 input/output alias 安全，memplan 不应通过地址复用隐式把它们变成 inplace 执行。

**适用场景 sample：算法本身不支持 input/output alias。**

```text
%x = alloc vec[1024]
%y = alloc vec[1024]

pto.tfmod ins(%x, %rhs) outs(%y)
```

`tfmod` 的实现可能会在计算中间覆盖某个源值，但后续仍需要原始源值。因此 `%y` 不能复用 `%x` 的物理地址。即使生命周期或 touching 规则允许，也必须由闸门 4 禁止：

```text
forbidAlias[%x].insert(%y)
```

#### 3.4.3 forbid_output_alias(i)

PyPTO 中 `forbid_output_alias(i)` 表示 op 整体可以对某些 value operand 做 inplace，但 output 不能 alias 第 `i` 个特定 operand。PTOAS 中需要按 PTO IR 的 DPS operand 布局映射到具体 operand。

典型场景：

```text
pto.tsel:
  forbid mask
  forbid tmp

pto.trowexpand / pto.tcolexpand:
  forbid broadcast source

pto.trowexpand* / pto.tcolexpand*:
  forbid row/column vector operand
```

**适用场景 sample：broadcast vector 不能被 output 覆盖。**

```text
%row = alloc vec[16x1]
%dst = alloc vec[16x64]

pto.trowexpand ins(%row) outs(%dst)
```

`%row` 会被重复读取并广播到 `%dst` 的多个位置。如果 `%dst` 复用 `%row` 的地址，写 output 的过程中可能覆盖后续仍要读取的 broadcast source。因此闸门 4 记录：

```text
forbidAlias[%row].insert(%dst)
```

**适用场景 sample：select 的 mask/tmp 不能和 output alias。**

```text
%mask = alloc vec[1024]
%tmp  = alloc vec[1024]
%dst  = alloc vec[1024]

pto.tsel ins(%mask, %lhs, %rhs, %tmp) outs(%dst)
```

`%lhs` / `%rhs` 是否允许和 `%dst` inplace 取决于 op 语义；但 `%mask` 和 `%tmp` 是被 op 读取或作为 scratch 使用的特殊 operand，不能被 `%dst` 覆盖。闸门 4 记录：

```text
forbidAlias[%mask].insert(%dst)
forbidAlias[%tmp].insert(%dst)
```

#### 3.4.4 闸门 4 判定

`canShare(a, b)` 中的闸门 4 是双向检查：

```text
Gate4_OpSemanticNoAlias(a, b):
  return !forbidAlias[a].contains(b)
      && !forbidAlias[b].contains(a)
```

由于 PTOAS 中 `pto.bind_tile`、`pto.slot_marker`、`memref.subview` 等 view/alias value 不一定是 root，收集 forbid-alias 时必须先做 root 归约：

```text
for value in op operands / dpsInits:
  roots = valueToRoots[value]
```

如果一个 value 通过 alias closure 对应多个 root，则需要记录所有 root 组合。这样后续 `ReuseGroup` 只需要比较 root 与 root，不需要在装箱阶段重新理解每个 op 的 operand 语义。



## 4. 规划算法

### 4.1 Root 收集

modern memplan 继续收集以下 local root：

- `pto.alloc_tile(no addr)`。
- plannable local `memref.alloc`。
- 后续如有其它明确的 local address root，可扩展到同一 root 表。

不参与 local memplan 的 root：

- GM space。
- `AddressSpace::Zero`。
- 已经由 level3 显式指定地址的 local `pto.alloc_tile addr`。
- 非静态 shape 或无法计算 element byte size 的 root。

### 4.2 LifetimeInterval

每个 root 生成一个 `LifetimeInterval`：

```text
  Value root;
  Operation *defOp;
  AddressSpace space;
  uint64_t slotBytes;
  uint64_t totalBytes;
  uint64_t alignmentBytes;
  uint64_t slotCount;
  unsigned allocIndex;
  unsigned freeIndex;
  unsigned stableOrder;
  SmallVector<uint64_t> offsets;
```

现代 planner 中已有线性化 walk 和 root alias 传播，本设计在此基础上补齐：

- loop-aware lifetime extension。
- branch / yield / iter_arg alias family 信息。
- per-root semantic metadata。

### 4.3 按 AddressSpace 分桶

复用只允许发生在同一 AddressSpace 内：

```text
Vec 只和 Vec 复用
Mat 只和 Mat 复用
Left/Right/Acc/Bias/Scaling 各自独立
GM 不参与 local memplan
```

分桶是硬约束，在进入装箱前完成。

### 4.4 Largest-First-Fit 排序与装箱

PTOAS 已有 `--plan-memory-order-by-size` 对应的 largest-first 排序能力。打开该选项后，每个 AddressSpace 桶内排序：

```text
sizeBytes 降序
sizeBytes 相同则按 defIndex 升序
defIndex 相同则按 stableOrder 升序
```

这样先让大 buffer 更早参与规划，再把生命周期不冲突的小 buffer 放入可复用位置，避免 definition-order greedy 只能“小复用先出现的大”的单向问题。

将一个物理 local buffer 抽象成一个 bin：

```text
ReuseGroup
  representative root
  members
  slot size bytes
  address space
  chosen offset
```

因此，本设计不是重新新增一个独立的 largest-first 开关，而是要求四道闸门的 `canShare` 判定接入现有 `--plan-memory-order-by-size` 路径。

特点：

- 不做 per-entry 降级、不撤销部分已提交规划。
- 在所有可容纳该 item 的 candidate（含 fresh 地址）中，按确定性优先级比较键选择，而不是"第一个可容纳即复用"，也不做加权打分。
- 单趟规划溢出时，按“容量退让：单调优先档位”对该 AddressSpace 整趟升档重算；触及终点档位（最大复用）仍溢出才报错。任何情况下都不为了适配容量放松四道闸门。



## 5. 性能模型

四道闸门只回答“两个 root 复用是否语义安全”。但对 A2/A3 这类 PIPE_V、MTE2、MTE3 可以重叠执行的后端，语义安全不等价于性能最优。若 modern memplan 为了最小化 local footprint，把本来分开的 scratch live range 压到同一小段物理 UB 地址，后续 InsertSync 会基于物理地址重叠补出更强的流水依赖；即使显式同步数量没有增加，MTE3 store 与下一段 MTE2 load 的可重叠窗口也会变窄。

因此，modern memplan 在通过四道安全闸门之后，还需要在合法 candidate 之间做性能取舍。本设计**不采用加权打分的 cost model**，而是使用一套**确定性优先级策略**：把性能关注点表达成一组从高到低的优先级层，用字典序比较，逐层严格压制、层间不做加权求和。这样决策可预测、可解释，且不依赖任何需要调参的魔数权重。



### 5.1 MTE access 收集

对每个可能影响流水的 op，收集其 pipe 与内存访问集合：

```text
OpAccess {
  Operation *op;
  Pipe pipe;
  unsigned opIndex;
  SmallVector<PhysicalInterval> reads;
  SmallVector<PhysicalInterval> writes;
}
```

`reads` / `writes` 来自 `MemoryEffectsOpInterface`，并通过 `valueToRoots` 归约到 root。规则与 InsertSync 保持一致：

`opIndex` 使用 pipe/memory-access op 的序号，而不是完整 IR linear op 序号；这样中间夹着 `arith.constant`、`pto.alloc_tile addr`、`TASSIGN` materialization 或结构性 op 时，仍能识别真正相邻的 PIPE_V/MTE 访问。

- DPS input 是 read。
- DPS output 是 write；若 op 明确 read-modify-write，则同时是 read + write。
- scratch/tmp operand 若被 MemoryEffects 建模为 write，则进入 writes；若接口语义要求 scratch 读旧值，则进入 reads + writes。
- 控制流 result、`pto.fusion_region` result、`pto.subview`、`pto.treshape`、`pto.bitcast`、`pto.multi_tile_get` 等 alias value 必须穿透到 root。

Phase 1 只有 `pipe == PIPE_MTE2` / `pipe == PIPE_MTE3` 的 op 参与 MTE 共址判定；PIPE_V op 的 access 可以一并收集，但 PIPE_V 同 pipe 共址判定留待后续阶段，本阶段不参与比较键。



### 5.2 PIPE_V、bank、hot cluster 预留

连续 PIPE_V producer/consumer 共址会收窄同 pipe 内的流水窗口，但 legacy pipe conflict 模型本身就不建模纯 vector-vector 复用（同 pipe 顺序执行，无跨 pipe overlap 可失去），且其收益相对 MTE 跨 pipe 串行化更小。因此本设计**Phase 1 不实现 PIPE_V 共址判定**，比较键中也不含 PIPE_V 项。

若后续阶段确认需要，可按如下形态引入（仅作预留，不属于当前范围）：

```text
pipeVCoLocated(opA, opB):
  opA.pipe == PIPE_V
  opB.pipe == PIPE_V
  distance(opA, opB) <= pipeVLookahead
  && (
       overlap(any opA.writes, any opB.reads)
    || overlap(any opA.writes, any opB.writes)
    || overlap(any opA.reads,  any opB.writes)
  )
```

届时以布尔形式接入比较键（是否制造 PIPE_V 共址），与 MTE 层同样保持纯布尔、层间字典序，不引入加权。



### 5.3 总体优先级

从高到低：

```text
1. 正确性     —— 四道闸门（硬约束，不可放松）
2. 容量可行   —— fits：加入该 candidate 后不超出 AddressSpace 容量
3. pipeline conflict 规避
4. bank conflict 规避
5. 紧凑度 / 确定性 tie-break —— projectedBytes、offset、stableOrder
```

层间是严格字典序：只有前一层完全打平，才由后一层决定。性能层（3/4）永远压不过容量可行（2），容量可行永远压不过正确性（1）。该策略不是第五道正确性闸门——它只在合法 candidate 之间排序，绝不为了性能放松四道闸门。



### 5.4 分阶段实施

本设计分阶段落地，避免一次性引入过多性能维度：

```text
Phase 1（本次实现）：只做 MTE pipeline conflict
  - 仅识别 MTE3-store 源 tile → MTE2-load 目的 tile 的物理共址。
  - 循环体内的 MTE 共址单独排一档，优先于直线代码的 MTE 共址。
  - PIPE_V 同 pipe 共址、hot cluster 聚集本阶段不做。

Phase 2（后续）：bank conflict 规避
  - 在 pipeline 层之下、tie-break 之上引入 bank conflict 层。

暂不纳入（后续可选）：
  - PIPE_V 同 pipe producer/consumer 窗口收窄。
  - hot cluster 过度聚集惩罚。
```



### 5.5 Phase 1 确定性比较键

每个 candidate（已通过四道闸门）计算如下比较键，选字典序**最小**者（布尔 false < true，因此 `!fits`、`hasLoopMteConflict`、`hasMteConflict` 都是越 false 越好）：

```text
candidateKey = (
  !fits,                // 容量可行优先：放不下最差
  hasLoopMteConflict,   // 在循环体内制造 MTE3→MTE2 共址（bool）
  hasMteConflict,       // 制造任意 MTE3→MTE2 共址（bool，含循环内）
  projectedBytes,       // 紧凑度：占用越小越好
  offset,               // 确定性 tie-break
  stableOrder           // 确定性 tie-break
)
```

- `hasLoopMteConflict` 与 `hasMteConflict` 都是**纯布尔**，不统计冲突数量；层内不做计数比较。
- 循环内 MTE 共址排在直线代码 MTE 共址之前：`(loop=true, any=true)` 比 `(loop=false, any=true)` 更差，因此 planner 会优先拆开循环体内的 MTE 共址。
- fresh（全新地址）天然不与任何已放置成员同址，`hasLoopMteConflict = hasMteConflict = false`；一个零冲突的 reuse 与 fresh 在 pipeline 层打平，再由 `projectedBytes` 让更紧凑的 reuse 胜出——因此真正零冲突的复用仍保留紧凑布局，只有当复用会制造 MTE 共址时，planner 才更倾向 fresh 地址来保住 DMA 流水并行。
- 容量压力下的兜底见后文“容量退让：单调优先档位”：单趟规划始终按当前档位比较键选择，`!fits` 已把放不下的 candidate 排到最差；若整趟仍溢出，则整趟升档重算，而不是在单个 candidate 处做局部回退。



### 5.6 物理区间建模

判断“两个 op 是否共址”不能只看 SSA value 是否相同，而要看 memplan materialize 后的物理 local 区间：

```text
PhysicalInterval {
  AddressSpace space;
  uint64_t startByte;
  uint64_t endByte;
}
```

每个 root 的区间由规划结果得到：

```text
startByte = plannedOffset
endByte   = plannedOffset + slotBytes
```

alias/view value 需要先归约到 root，再把 subview/treshape/bitcast/multi_tile_get 等局部 offset 合入区间。对无法精确计算 byte range 的 alias，性能模型应保守地使用 root 的完整 slot range，不能为了少报冲突而截断未知范围。

当前 modern memplan 的首版实现运行在 materialize 之前，因此用“prospective `ReuseGroup` 共址”作为物理区间代理：如果一个 root 加入已有 group，就认为它和 group 内成员共享同一 local 区间；如果选择 fresh group，就认为它不和已有 group 产生共址代价。后续若 planner 支持更细粒度 subview byte range，可把该代理替换为精确 `PhysicalInterval`。

区间重叠判定：

```text
overlap(a, b):
  a.space == b.space
  && a.startByte < b.endByte
  && b.startByte < a.endByte
```



### 5.7 MTE3 -> MTE2 共址判定（Phase 1 核心）

对 store-heavy kernel，需要识别同一物理 UB 区间被 MTE3 store 源 tile 使用后，又马上作为 MTE2 load 目的 tile 复用的场景。此时 InsertSync 往往需要建立更强的 `MTE3 -> MTE2` 依赖，后续 load 不能像不同地址时那样提前发起，DMA 跨 pipe 流水被串行化：

```text
mteStoreThenLoadCoLocated(opA, opB):
  opA.pipe == PIPE_MTE3
  opB.pipe == PIPE_MTE2
  distance(opA, opB) <= mteLookahead
  && overlap(any opA.reads, any opB.writes)
```

`mteLookahead` 初始取 1，仅覆盖真正近邻的 store→load；它是识别范围参数，不影响四道安全闸门。

该判定按 candidate 聚合成比较键里的两个布尔：

```text
hasMteConflict(candidate):
  存在任意一对 (opA, opB) 满足 mteStoreThenLoadCoLocated，
  且该共址由“把当前 root 放到此 candidate offset”引入。

hasLoopMteConflict(candidate):
  上述 (opA, opB) 中至少有一处发生在循环体内。
```

- 两者都是**纯布尔**，不统计共址对的数量。
- 循环体内的 store→load 共址会在每次迭代重复触发串行化，危害显著大于直线代码，因此 `hasLoopMteConflict` 单独排一档、优先于 `hasMteConflict`（见比较键定义）。这取代了旧模型中 `loopWeight` 的加权做法：不再乘系数，而是用更高的优先级层严格压制。
- `tstore` 后接下一段 `tload` 的 loop 是最典型的敏感场景。



### 5.8 确定性 candidate 选择

不使用 penalty 累加，而是对每个已过四道闸门的 candidate 构造前文定义的比较键，取字典序最小者：

```text
选择流程：
  candidates = 已有 ReuseGroup 的复用 offset + fresh offset
  candidates = filterByFourGates(candidates)         // 正确性硬过滤
  for c in candidates:
    key(c) = (
      !fits(c),
      hasLoopMteConflict(c),
      hasMteConflict(c),
      projectedBytes(c),
      offset(c),
      stableOrder(c)
    )
  choose argmin key(c)                                // 字典序最小
```

要点：

- 层间严格字典序、层内纯布尔，没有任何魔数权重；给定输入，选择结果唯一确定。
- pipeline 层（`hasLoopMteConflict` / `hasMteConflict`）优先于紧凑度（`projectedBytes`）：为避免 MTE 共址宁可多占字节，只要仍 `fits`。
- 零冲突的 reuse 与 fresh 在 pipeline 层打平，由 `projectedBytes` 让更紧凑的 reuse 胜出，保持紧凑布局。

单趟规划不做任何 per-item 局部容量回退：始终按当前档位比较键取 argmin。容量兜底完全交给跨趟的“容量退让：单调优先档位”（§5.2），保持单一机制、便于推理与测试。

Cube 相关 local space（`MAT` / `LEFT` / `RIGHT` / `ACC`）不使用 largest-first 排序，即使用户显式选择 modern planner 默认的 `orderBySize`。这些空间的性能更依赖计算流附近的 L1/L0A/L0B/ACC 地址规律；把大 tile 全部提前规划会打散 `TLOAD -> TEXTRACT -> TMATMUL` 周边的 operand 地址模式，可能触发不利 bank pattern。`VEC` 仍保留 largest-first，因为 VEC scratch 的主要风险通常是容量压力和跨 MTE store/load 复用，而不是 cube operand 的固定 L0 bank 节奏。



### 5.9 Phase 2 预留：bank conflict 层

Phase 2 在 pipeline 层之下、`projectedBytes` tie-break 之上插入 bank conflict 布尔层，比较键扩展为：

```text
(!fits, hasLoopMteConflict, hasMteConflict, hasBankConflict, projectedBytes, offset, stableOrder)
```

`hasBankConflict` 首版只建模最强信号：candidate reuse 会让两个 hot root 精确同址，因此一定落入同一 bank pattern；后续若 planner 在 materialize 前具备精确 offset/stride interval，可扩展为 `offset % bankModulo` 的更细判断。该层同样保持纯布尔、层间字典序，不引入加权。Phase 1 不实现本层。



### 5.10 容量退让：单调优先档位（overflow-triggered）

§5 的比较键在容量充足时最大化性能（优先拆开 MTE 共址、倾向 fresh）。但贪心是逐个 item 处理的：早期 item 空间宽裕会抢到 fresh / 零冲突地址，把 footprint 抬高；等空间被占满，后期 item 只能退回“能 fits 就行”的压缩 reuse。这种**顺序偏置**会让性能优化不公平地倾斜给先处理的 item，甚至整体溢出。单趟规划内不做任何 per-item 局部回退，无法前瞻整个尾部需求，因此需要一层跨趟的全局兜底。

为在**不引入魔数权重、不做 legacy 式投机回滚**的前提下兜住容量，引入**溢出触发、单调收敛、有限档位**的确定性容量退让：把比较键组织成从“最激进（性能最优）”到“最紧凑（最大复用）”的有限档位；乐观档先跑，若某个 AddressSpace 溢出，则**只对该空间**升一档整趟重算，直到放下或触及终点档位。

**档位定义（Phase 1）**

档位通过**逐层剥离 pipeline 层**来单调增强紧凑度，四道闸门与 `!fits` 始终保留：

```text
档位 0（性能态）：
  (!fits, hasLoopMteConflict, hasMteConflict, projectedBytes, offset, stableOrder)
  —— 拆开所有 MTE 共址，容量充足时倾向 fresh。

档位 1（半压缩）：剥离 hasMteConflict，仅保护循环体
  (!fits, hasLoopMteConflict, projectedBytes, offset, stableOrder)
  —— 直线代码 MTE 共址不再逼向 fresh，只保护危害最大的循环内共址。

档位 2（全压缩，≈ legacy L0）：剥离所有 pipeline 层
  (!fits, projectedBytes, offset, stableOrder)
  —— 纯紧凑：总是取占用最小的合法 reuse，fresh 仅在无 reuse 可用时选。
     即四道闸门下的最大复用，为终点档位。
```

剥离顺序始终是“先弱信号、后强信号”：先放弃直线 MTE 保护，再放弃循环 MTE 保护，终点是纯紧凑。Phase 2 引入 bank 层后，在剥离所有 pipeline 层之后、剥离到纯 tie-break 之前，多一档“保留 bank 层、已剥离 pipeline”的中间档，终点不变。

**单调性证明（为何一定收敛、无需防重入补丁）**

- 档位 k+1 的比较键 = 档位 k 去掉最高一层“偏好 fresh”的 pipeline 判据。去掉一个偏好 fresh 的层，只会让紧凑 reuse 赢得更多或打平，绝不会让 footprint 变大——因此 footprint 随档位**单调不增**。
- 档位号每次严格 +1，且档位数有限。过程必在有限步内停止：要么某档放下，要么在终点档（最大复用）仍溢出 → 判定为真 overflow 报错。
- 因为档位号严格递增，绝不会回到已试过的档位，所以**不需要** legacy 的 `IsSamePlanAsLastRollBack` / `specStartIdx` 防重入机制——legacy 需要它，正是因为它的 per-entry 降级不是严格单调的。

**退让循环（per-AddressSpace）**

```text
for each AddressSpace space:
  level = 0
  loop:
    plan = runGreedyPlacement(space.items, keyForLevel(level))   // 整趟从头规划
    if plan.maxRequiredBytes <= space.capacityBytes:
      commit(plan); break                                        // 放下 → 成功
    if level == MAX_LEVEL:
      emitError("<space> overflow ..."); return failure          // 终点仍溢出 → 真 overflow
    level = level + 1                                             // 严格 +1，单调升档
```

- **触发条件只有硬溢出**（`offset + size > capacity`），绝不由性能信号触发——保证它是容量可行性兜底，而非把加权打分从后门放回。
- **每档整趟重算**，丢弃上一档的 group 分配，而不是 legacy 那样撤销部分已提交 PlanRecord。全量重算天然确定。
- **per-AddressSpace 升档**：只升溢出的空间，其余空间保留在各自最优（性能）档位，比 legacy 单一全局 `specLevel` 更精确。
- **确定性与代价**：档位序列与每档结果都是输入的纯函数，同输入必得同输出；代价是溢出空间最多跑 `MAX_LEVEL+1` 趟（档位数很小，约 2~3），每趟 O(items×groups)。

**与 legacy 三级复用 + 回滚的对照**

| 维度 | legacy | 本设计单调档位 |
| --- | --- | --- |
| 触发 | 溢出 | 溢出（相同） |
| 单位 | 全局 `specLevel` + per-entry 降级 | per-AddressSpace 档位，无 per-entry |
| 回滚方式 | 撤销部分已提交 PlanRecord | 整趟丢弃重算 |
| 防死循环 | 需 `IsSamePlanAsLastRollBack` | 靠单调 + 有限档位天然保证 |
| 性能表达 | SPEC_LEVEL_1/2 的分层投机 | 已由比较键表达；档位只做剥离 |





### 5.11 prefill_c4_state_update 类场景

典型退化模式：

```text
legacy:
%t        addr = 0
%pool_dep addr = 256
%ape_row  addr = 512
%tmp0     addr = 768
%out0     addr = 1024
%tmp1     addr = 1280
%out1     addr = 1536
%tmp2     addr = 1792

modern aggressive:
%t        addr = 0
%pool_dep addr = 0
%ape_row  addr = 256
%tmp0     addr = 512
%out0     addr = 512
%tmp1     addr = 512
%out1     addr = 256
%tmp2     addr = 256
```

从四道闸门看，modern aggressive 规划可能是合法的：这些 scratch 的静态生命周期不重叠，op semantic no-alias 也未禁止它们复用。但从流水性能看，它把多个连续 V 计算和 store/load 交错阶段压到同一物理地址，使后续同步分析必须把本可 overlap 的阶段串起来。

确定性策略在 Vec 容量充足时，通过 `hasLoopMteConflict` / `hasMteConflict` 层优先拆开 store→load 共址，选择更接近 legacy 的展开地址；只有当 Vec 容量确实不足（档位退让升到更紧凑档位）时，才逐步接受 `0/256/512` 这类压缩复用。其中连续 V 计算阶段的展开属于 PIPE_V 共址维度，由 Phase 2/预留项处理；Phase 1 只保证 store/load 的 MTE 共址被优先拆开。



### 5.12 与 largest-first-fit 的关系

Largest-first 仍决定 item 处理顺序；确定性优先级策略只影响“当前 item 放入哪个 candidate offset”。单趟规划外层再套一层单调档位退让（见 §5.2），推荐流程：

```text
for space in addressSpaces:
  for level in 0 .. MAX_LEVEL:               // 单调升档，仅在溢出时进入下一档
    reset(space)                             // 整趟重算，丢弃上一档 group 分配
    for item in largestFirstOrder(space):
      candidates = collectExistingReuseGroupsAndFreshOffsets(item)
      candidates = filterByFourGates(candidates)
      choose argmin keyForLevel(level, candidate)
        // level 0: (!fits, hasLoopMteConflict, hasMteConflict, projectedBytes, offset, stableOrder)
        // level 1: (!fits, hasLoopMteConflict,                 projectedBytes, offset, stableOrder)
        // level 2: (!fits,                                     projectedBytes, offset, stableOrder)
    if fits(space): break                     // 放下 → 采用该档结果
  if not fits(space): emitError(overflow)     // 终点档仍溢出 → 真 overflow
```

比较键整体保持 deterministic：容量可行 → 循环内 MTE 共址 → 任意 MTE 共址 → 占用字节 → offset → stableOrder，逐层升序（布尔 false 先于 true）。这样既保留 largest-first 的确定性，也避免“第一个空洞”把高频 scratch 过早压到低地址、制造 MTE 串行化。外层档位退让保证：容量充足时停在 level 0（性能最优），只有真正的容量压力才逐级放弃 pipeline 保护、换取更紧凑复用，且过程单调、可终止、可复现。



## 6. 实现映射

### RootInfo

```cpp
struct RootInfo {
  Value root;
  Operation *defOp = nullptr;
  AddressSpace space = AddressSpace::Zero;
  uint64_t slotBytes = 0;
  uint64_t totalBytes = 0;
  uint64_t alignmentBytes = 1;
  uint64_t slotCount = 1;
  unsigned allocIndex = 0;
  unsigned freeIndex = 0;
  unsigned stableOrder = 0;
  SmallVector<uint64_t> offsets;
};
```

本设计保持当前 modern memplan 的 `RootInfo` 字段不变。`RootInfo` 只表达 local allocation root 的基础事实：

```text
root identity
definition op
address space
slot size / total size / slot count
lifetime interval
stable order
planned offsets
```

四道闸门所需的额外事实不直接塞进 `RootInfo`，而是放在 `ConflictFacts` 这类 side table 中。这样可以避免 root 结构随着每个闸门膨胀，也便于后续按阶段启用或删除某个闸门。

### ReuseGroup

```cpp
struct ReuseGroup {
  Value representative;
  SmallVector<unsigned> memberIndices;
  AddressSpace space;
  uint64_t slotSizeBytes;
  uint64_t offsetBytes;
};
```

### ConflictFacts

```cpp
struct ConflictFacts {
  DenseMap<Value, SmallVector<Value>> forbidAlias;
  DenseSet<Value> loadDerivedRoots;
  DenseSet<Value> tpopConsumerRoots;
  DenseMap<Value, SmallVector<unsigned>> phiFamilyIds;

  // Performance-only facts. These do not decide whether reuse is legal; they
  // feed the deterministic priority key that ranks already-legal candidate
  // offsets. Phase 1 only populates MTE (PIPE_MTE2 / PIPE_MTE3) accesses used
  // to detect MTE3-store -> MTE2-load co-location; PIPE_V co-location and bank
  // conflict facts are deferred to Phase 2.
  SmallVector<OpAccess> opAccesses;
  DenseMap<Value, SmallVector<PhysicalInterval>> plannedIntervals;

  // Reserved for future implicit pipeline lowering support. Not used by the
  // current PTOAS design because explicit multibuffer owns ping-pong slot
  // separation.
  // DenseMap<Value, SmallVector<PipelineMembership>> pipelineMembership;
  // DenseSet<Value> pipelineLoadRoots;
};
```

### 说明：pipeline stage load conflict（预留，不纳入当前实现）

- 当前不实现 pipeline stage load conflict；该项仍是后续预留能力。
- 不新增 pipeline stage load 复用负例。
- 保留设计占位：若未来 PTOAS 引入隐式 pipeline lowering，并能稳定提供 `pipelineMembership[root] = (group, stage)`，再接入该闸门。
- 显式 `pto.alloc_multi_tile count=N` 继续由 multibuffer slot 分配保证 ping-pong 正确性。



### 6.1 代码映射

| 设计内容 | 当前实现 |
| --- | --- |
| modern planner | `lib/PTO/Transforms/PTOPlanMemoryModern.cpp` |
| conflict facts / forbidAlias | `PTOPlanMemoryModern.cpp` 中的 `ConflictFacts` |
| canShare / target hazard | `PTOPlanMemoryModern.cpp` 中的 gate helpers |
| MTE candidate ranking | `PTOPlanMemoryModern.cpp` 中的 `MteConflict` / `ReuseKey` |
| CLI planner selection | `tools/ptoas/ptoas.cpp` |
| order-by-size option | `include/PTO/Transforms/Passes.td` |



## 7. 测试与验证

### 7.1 lit 测试

新增或扩展以下测试：

- `plan_memory_order_by_size_*.pto` 继续作为 largest-first 覆盖。
- `plan_memory_five_gates_lifetime_touching.pto`
- `plan_memory_five_gates_phi_family.pto`
- `plan_memory_five_gates_semantic_no_alias.pto`
- `plan_memory_five_gates_target_hazard.pto`
- `plan_memory_mte3_mte2_reuse_cost.pto`（**Phase 1 核心**）：构造 `tstore` 源 tile 后接 `tload` 目的 tile 的近邻复用场景，验证 planner 优先选择不会制造 `MTE3 -> MTE2` 共址依赖的地址；并补一条 loop-body 变体，验证循环内 MTE 共址被排在更高一档（`hasLoopMteConflict` 优先于 `hasMteConflict`）。
- `plan_memory_pipev_reuse_cost_state_update.pto`（**Phase 2 预留**）：构造多个连续 PIPE_V scratch，验证容量充足时 modern memplan 不把所有 touching live range 压到同一小段 Vec 地址。Phase 1 不实现 PIPE_V 共址，暂不启用。
- `plan_memory_pipev_reuse_cost_capacity_pressure.pto`（**Phase 2 预留**）：构造 Vec 容量接近上限的场景，验证性能层只影响 candidate 排序，不会因为偏好 fresh address 而错误拒绝合法复用。
- `plan_memory_capacity_escalation_monotonic.pto`（**Phase 1 核心**）：构造 Vec 在 level 0（全展开）必溢出、level 1/2 逐步压缩后才放下的场景，验证 planner 按单调档位升档、最终得到确定且可放下的布局，而非直接报 overflow；再构造一个连终点档位都放不下的场景，验证报出真 overflow。
- 暂不新增 `plan_memory_five_gates_pipeline_load.pto`；闸门 4 当前为预留设计。

已有 `plan_memory_*.pto` 应继续保留 legacy + modern 双 RUN。

### 7.2 验证命令

```bash
cmake --build build --target ptoas -j8

PATH=/Users/fangrui/workspace/huawei/llvm21-workspace/llvm-project/llvm/build-assert/bin:$PATH \
  /Users/fangrui/workspace/huawei/llvm21-workspace/llvm-project/llvm/build-assert/bin/llvm-lit \
  -sv build/test/lit \
  --filter 'plan_memory'

ctest --test-dir build --output-on-failure -L PTODSL
```



## 8. 风险、限制与后续工作

- `--plan-memory-order-by-size` 本身会改变 modern memplan 的 offset 分配顺序，测试应避免把 order-sensitive 预期错误地复用于默认路径。
- 四道闸门是硬约束，不应为了容量不足而放松。
- target hazard 需要先确认 PTOAS IR 中稳定的标记来源。
- pipeline metadata 闸门当前不实现；如果未来启用，需要先定义稳定的 pipeline membership 来源。
- phi family 豁免必须保守，不能让外部 live alias 借互斥分支错误复用。
- 确定性优先级层只是性能排序，不是安全闸门；容量不足时不能因为存在 MTE 共址就报错，除非四道安全闸门本身失败。
- 布尔层（`hasLoopMteConflict` / `hasMteConflict`）会丢失“多个小冲突 vs 一个大冲突”的定量权衡能力；这是刻意取舍，换取 deterministic 且无魔数加权。若后续确有需要，应新增独立布尔层而非退回加权打分。
- Phase 1 的 MTE 识别范围（PIPE_MTE2 / PIPE_MTE3）必须与 InsertSync 的 pipe 归类保持一致；否则 planner 认为无冲突的地址，后续同步分析仍可能补出 MTE 串行依赖。
- 物理区间必须和 InsertSync 使用的 root/alias/MemoryEffects 视角保持一致；否则 planner 认为低冲突的地址，后续同步分析仍可能补出强依赖。
- 单调档位退让必须保证 footprint 随档位单调不增：新增或调整档位时，只能“剥离偏好 fresh 的层”，不得引入会让某些 item 反而更占空间的规则，否则收敛性与终止性不再成立。
- 档位退让的触发条件必须严格限定为硬溢出，绝不能由性能信号触发；否则等于把加权/投机回滚从后门放回，违背 deterministic 目标。
- 每档整趟重算而非部分撤销：实现时不要复用 legacy 的 PlanRecord 撤销路径；per-AddressSpace 升档需保证各空间档位相互独立、互不影响。
- 档位数应保持很小（约 2~3）；档位过多会放大重跑代价，也说明性能层与紧凑度的边界没切干净。
- legacy memplan 不应受该设计影响，默认行为保持不变。
