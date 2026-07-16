# PTOAS Implicit Tmp Materialization Design

## 背景

PTOAS 前端 IR 中很多 tile op 的 `tmp` operand 是可选的。当前如果用户没有显式写 `tmp`，PTOAS 会继续 lowering 到后端不带 `tmp` 的 C++ 接口。

后续希望改成：前端仍允许省略 `tmp`，但 PTOAS 在内部为需要 tmp-aware 后端接口的 op 自动补充合法 tmp tile，并让这些 tmp tile 和其它 tile buffer 一起进入 memplan 做 local addr 规划。

本文档的目标是给所有类似 `pto.tci` 的 optional tmp op 提供整体改造方案。每个 op 再根据自己的后端 tmp 规格，补充 op-specific 的 tmp requirement、MemoryEffects、verifier 和测试。

## 目标

- 保持前端 IR 兼容：用户仍然可以写不带 `tmp` 的目标 op。
- 在 memplan 之前补齐隐式 tmp，使 tmp 作为普通 local allocation root 参与地址规划。
- 对需要 tmp-aware 后端接口的 op，EmitC lowering 统一走带 tmp 的 C++ overload，避免继续选择 no-tmp 后端接口。
- 每个 op 的 tmp shape、dtype、address space、layout、容量等约束由该 op 的后端接口规格决定。
- 不在 EmitC lowering 中临时分配 tmp 地址。
- 不在 memplan 中特殊创建 tmp；memplan 只负责规划已经存在的 root。

## 非目标

- 本阶段不一次性覆盖所有 optional tmp op。
- 本阶段不改变用户显式提供 tmp 的语义。
- 本阶段不为 level3 自动分配 tmp 地址。
- 本阶段不引入新的全局 workspace 规划。
- 本阶段不把所有 op 的 tmp 规格抽象成完全统一的 shape；不同 op 可以有不同 tmp requirement。

## 总体方案

新增一个 IR 规范化 pass：

```text
pto-materialize-implicit-tmp
```

该 pass 运行在 `PTOViewToMemref` 之后、`pto-plan-memory` 之前：

```text
PTOViewToMemref
  -> pto-materialize-implicit-tmp
  -> pto-plan-memory
  -> PTOResolveReservedBuffers
  -> sync passes
  -> PTOMaterializeTileHandles
  -> PTOToEmitC
```

pass 的职责是扫描所有已纳入改造的目标 op。如果 op 没有 tmp operand，就根据该 op 的 `TmpRequirement` 在 op 前插入 `pto.alloc_tile(no addr)`，并重写原 op，使其显式携带 tmp。

抽象流程：

```text
target_op(no tmp)
  -> lookup TmpRequirement(target_op)
  -> create pto.alloc_tile(no addr) tmp
  -> rewrite target_op(with tmp)
  -> memplan assigns addr to tmp
  -> EmitC sees tmp and emits tmp-aware overload
```

`TmpRequirement` 至少应包含：

```text
AddressSpace space;
Type elementType;
StaticShape or MinBytes requirement;
Layout/layout-family requirement;
uint64_t minBytes;
bool requireExplicitAtLevel3;
```

对自动生成的 tmp：

- 使用 tile-native `pto.alloc_tile(no addr)`。
- 不创建 `memref.alloc`。
- 不创建 `pto.pointer_cast` / `pto.bind_tile`。
- 不设置 `addr`，由 memplan 统一规划。
- 尽量使用静态 full-valid shape，即 `v_row/v_col` 与 `rows/cols` 一致，不额外携带 `valid_row` / `valid_col` operand。
- 定义位置必须支配目标 op。
- 生命周期由后续 liveness/memplan 根据真实 use 计算。

## Memplan 接入

自动生成的 tmp 是 tile-native `pto.alloc_tile(no addr)`，因此复用当前 memplan 路径：

```text
pto.alloc_tile(no addr)
  -> local allocation root
  -> legacy/modern memplan 分配 offset
  -> pto.alloc_tile addr = ...
```

legacy memplan 和 modern memplan 都应把自动生成的 tmp 当成普通 local allocation root。memplan 不应该知道“这是某个 op 的隐式 tmp”，也不应该在内部临时创建 tmp。

memplan 侧需要依赖 op 的 MemoryEffects / semantic no-alias 信息保证正确复用：

- tmp 如果是 scratch buffer，应通过 `Write(tmp)` 建模，使 scratch-output conflict 能禁止 tmp 与同 op output 错误复用。
- 如果某个 op 的 tmp 与 output 不能 alias，但 tmp 不适合建模成 scratch write，则应在 semantic no-alias side table 中显式加入 `forbidAlias(tmp, output)`。
- 每个 op 的专项改造必须说明 tmp 和 output、input 之间的 alias 约束。

## Level 行为

### level1 / level2

level1/level2 下 memplan 会运行，因此允许省略 tmp：

```text
target_op(no tmp)
  -> pto-materialize-implicit-tmp
  -> pto.alloc_tile(no addr) tmp
  -> pto-plan-memory 补 addr
```

用户显式提供 tmp 时，仍需满足该 op 的 tmp verifier 约束。level1/level2 下用户不应显式指定 local addr，地址由 memplan 统一规划。

### level3

level3 下用户显式管理 local 地址，memplan 通常跳过。因此不应自动创建无地址 tmp。

通用规则：

```text
level3 + target_op(no tmp) => pass/verifier 报错
```

用户在 level3 使用已纳入改造的目标 op 时，必须显式提供合法 tmp，并保证 tmp 自身带合法 local addr，或满足现有 level3 显式地址规则。

诊断信息示例：

```text
<op> requires explicit tmp when compiling at level3 because PlanMemory is skipped
```

## EmitC Lowering

目标 op 的 EmitC lowering 应保持简单：

- `op.getTmp()` 为空：生成 no-tmp C++ 调用，或者在该 op 改造完成后仅作为未经过 materialize pass 的兜底路径。
- `op.getTmp()` 非空：生成带 tmp 的 C++ 调用。

引入 `pto-materialize-implicit-tmp` 后，level1/level2 的目标 op 在 EmitC 前都会携带 tmp，因此会自然走带 tmp 的 overload。

不建议在 PTOToEmitC 中补 tmp，原因：

- EmitC 阶段已经错过 memplan。
- 临时生成 tmp 无法获得 local addr。
- 会绕过 liveness、sync 和 semantic no-alias 分析。

## TCI 针对性改造

本节描述 `pto.tci` 作为第一批目标 op 的具体落地规则。后续其它 optional tmp op 应新增类似小节，分别说明自己的 tmp 规格、pass 行为、MemoryEffects、verifier 和测试计划。

### TCI Tmp 约束

`pto.tci` 当前 ODS 已经支持可选 tmp：

```td
Optional<PTODpsType>:$tmp
```

PTOToEmitC 也已经根据 `op.getTmp()` 选择带 tmp 或不带 tmp 的 C++ 调用。因此 TCI 改造不需要改 `pto.tci` 的 IR 语法，关键是保证进入 EmitC 前缺省 tmp 已经被显式 materialize。

TCI 后端 C++ 接口存在两类 overload：

```cpp
TCI(dst, start)
TCI(dst, start, tmp)
```

A2/A3 上 no-tmp overload 可能走 scalar loop；带 tmp overload 才能走更优路径。A5 接受 tmp，但 tmp 可以作为兼容占位，不额外引入有效计算约束。

TCI tmp 不应要求固定 shape，应按 PTO-ISA 文档中的精细化 tmp 约束校验容量。PTOAS 对用户显式 tmp 和自动生成 tmp 采用同一组 A2/A3 合法性规则：

```text
loc      = vec
dtype    = 4-byte type: f32 / i32 / ui32
shape    = static shape
layout   = row_major
fractal  = 512
capacity = product(shape) * sizeof(dtype)
```

A2/A3 的最小容量由 dst 元素类型决定：

```text
b32 dst: i32 / ui32  -> tmp capacity >= 768 bytes
b16 dst: i16 / ui16  -> tmp capacity >= 1792 bytes
```

其中 `shape` 可以是任意静态形状，只要总容量满足对应 dst 类型的最小容量。例如 b32 dst 可以使用 `1x192xf32`，b16 dst 可以使用 `1x448xf32`。`Tile<TileType::Vec, float, 1, 512>` 是 PTO-ISA 文档中推荐的方便形状无关分配，容量为 2048 bytes (2KiB)，可以同时覆盖 b32/b16。

A5 上 `tmp` Tile 被接受但不使用；A5 硬件直接使用 `vci` 向量指令，无需临时缓冲区。因此 A5 下 `pto.tci(no tmp)` 不需要自动 materialize tmp，用户显式传 tmp 时也不按 A2/A3 的容量规则校验。

### Pass 行为

对每个 `pto.tci`：

- 如果已经有 `tmp`，pass 不修改。
- A5 如果没有 `tmp`，pass 不修改；A5 后端直接使用 `vci`，不需要 tmp。
- A2/A3 如果没有 `tmp`，且当前 build level 会运行 memplan，则自动补 tmp。
- A2/A3 如果没有 `tmp`，但当前 level3 会跳过 memplan，则报错，要求用户显式提供带地址的 tmp。

重写前：

```mlir
pto.tci ins(%s : i32)
  outs(%dst : !pto.tile_buf<loc=vec, dtype=i32, rows=1, cols=128, ...>)
```

A2/A3 重写后，以下以 b32 dst 自动生成 `f32 1x192` tmp 为例：

```mlir
%tmp = pto.alloc_tile
  : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=192,
                  v_row=1, v_col=192, blayout=row_major,
                  slayout=none_box, fractal=512, pad=0>

pto.tci ins(%s, %tmp : i32,
            !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=192,
                          v_row=1, v_col=192, blayout=row_major,
                          slayout=none_box, fractal=512, pad=0>)
  outs(%dst : !pto.tile_buf<loc=vec, dtype=i32, rows=1, cols=128, ...>)
```

随后 memplan 会把 `%tmp` 当成普通 tile-native local allocation root，和其它 `pto.alloc_tile(no addr)` 一起规划 local address：

```mlir
%tmp = pto.alloc_tile addr = %c4096_i64
  : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=192, ...>
```

TCI rewrite 需要保留原 op 的：

- scalar operand `S`。
- dst operand。
- `descending` attr。
- location。
- 其它已有属性。

### MemoryEffects

当前 `TCIOp::getEffects()` 只建模为：

```text
Write(dst)
```

A2/A3 自动补 tmp 后，应改为：

```text
Read(tmp) if tmp exists
Write(tmp) if tmp exists
Write(dst)
```

A5 上 tmp 被接受但不使用，因此不应把 tmp 建模为 Read/Write：

```text
Write(dst)
```

原因：

- liveness 需要看到 tmp 在 `pto.tci` 被使用。
- sync pass 需要知道 `pto.tci` 会读 tmp 地址。
- memplan 需要把 tmp 识别为 scratch buffer，避免 tmp 和同 op 的 dst 错误复用。
- modern memplan 的 op semantic no-alias 和 root use 传播需要真实 use 信息。

如果 tmp 不被建模为 Read/Write，tmp 可能被认为没有 use 或不是 scratch，导致生命周期、复用或同步分析不准确。

TCI 也可以在 semantic no-alias side table 中显式加入：

```text
op = pto.tci
forbidAlias(tmp, dst)
```

这不是 scratch conflict 生效的必要条件；A2/A3 只要 `TCIOp::getEffects()` 建模了 `Write(tmp)`，tmp 就会进入 scratch buffer conflict。但显式 side table 能防止未来有人调整 MemoryEffects 后破坏 tmp/dst no-alias 语义。

### Verifier

`TCIOp::verify()` 应检查 tmp 是合法 tile buf，并满足后端接口容量约束：

```text
如果 tmp 存在：
  A5: tmp 被接受但不使用，不执行 A2/A3 tmp 容量校验。
  A2/A3: tmp 必须是 vec tile。
  A2/A3: tmp element type 必须是 4 字节类型（f32 / i32 / ui32）。
  A2/A3: tmp shape 必须是静态 shape。
  A2/A3: tmp layout 必须满足后端 TCI tmp 接口要求。
  A2/A3 b32 dst: tmp capacity 必须大于等于 768 bytes。
  A2/A3 b16 dst: tmp capacity 必须大于等于 1792 bytes。
```

这里的关键是“容量满足接口约束”，而不是“shape 必须等于某个固定值”。TCI 按 dst 元素类型精细化检查：

```text
// b32 dst: 768B 即可
!pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=192, ...>  // 合法
!pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=128, ...>  // 非法，容量不足

// b16 dst: 1792B 即可
!pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=448, ...>  // 合法
!pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=192, ...>  // 非法，容量不足
```

自动生成 tmp 选择满足最小容量的 canonical shape：

- b32 dst: `f32 1x192`。
- b16 dst: `f32 1x448`。
- A5: 不自动生成 tmp。

### 测试计划

#### lit：自动补 tmp

新增用例：

```text
test/lit/pto/tci_implicit_tmp_materialization.pto
```

检查：

```text
CHECK: pto.alloc_tile
CHECK-SAME: dtype=f32
CHECK: pto.tci ins(%{{.*}}, %{{.*}}
CHECK-NOT: memref.alloc
```

可以额外检查自动生成 shape：b32 dst 为 `f32 1x192`，b16 dst 为 `f32 1x448`。同时应覆盖 A5 下不自动生成 tmp。

#### lit：memplan 回写 addr

检查 plan memory 后：

```text
CHECK: pto.alloc_tile addr =
CHECK: pto.tci ins(%{{.*}}, %{{.*}}
```

legacy 和 modern 都应覆盖：

```text
// RUN: ptoas --pto-level=level2 --plan-memory-impl=legacy ...
// RUN: ptoas --pto-level=level2 --plan-memory-impl=modern ...
```

#### lit：EmitC 走带 tmp overload

检查 C++ 输出：

```text
CHECK: TCI<
CHECK-SAME: Tile<
CHECK-SAME: float
CHECK: TCI{{.*}}({{.*}}, {{.*}}, {{.*}})
```

#### lit：level3 负例

```text
level3 + pto.tci(no tmp)
```

期望：

```text
expected-error {{pto.tci requires explicit tmp when compiling at level3}}
```

#### lit：verifier 负例

用户显式提供非法 tmp：

- 非 vec space。
- 非 f32 dtype。
- dynamic shape。
- layout 不满足 TCI tmp 接口约束。
- A2/A3 b32 dst 的 tmp capacity 小于 768 bytes。
- A2/A3 b16 dst 的 tmp capacity 小于 1792 bytes。

期望 verifier 报错。

## TROWEXPAND 二元 op 针对性改造

本节覆盖以下 row-expand 二元 op：

```text
pto.trowexpandadd
pto.trowexpandsub
pto.trowexpandmul
pto.trowexpanddiv
pto.trowexpandmax
pto.trowexpandmin
```

这些 op 的 PTO-ISA 文档对 tmp 的描述一致：带 `TileDataTmp &tmp` 的 C++ overload 仅支持模式 1；A2/A3 上 tmp 用作行广播缓冲区；A5 接受 tmp 但不使用。

### RowExpand Tmp 约束

这些 op 有两种 row-broadcast 模式：

- 模式 1：扩展操作数为 `ColMajor`，每行一个标量。带 tmp overload 仅支持该模式。
- 模式 2：扩展操作数为 `RowMajor`，每行一个 32 字节块。该模式不需要 tmp，不应为了 tmp-aware overload 强行改写。

A2/A3 模式 1 下，tmp 作为 `vbrcb` 广播缓冲区使用。扩展操作数的每行标量会广播成一个 32 字节块；`vbrcb` repeat stride 为 8 个块，即 256 字节，每个 repeat 处理 8 行。

tmp 最小容量由 `R = dst.validRow` 决定：

```text
if R < 256:
  tmpBytes = ceil(R / 8) * 256
else:
  tmpBytes = 30 * 256 = 7680
```

说明：

- 当 `R >= 256` 时，后端按循环处理，每次循环最多 30 个 repeat，也就是 240 行；tmp 在循环间复用，因此每次循环只需要 7680 字节。
- 一个紧凑的形状无关上界是 8KB，即 8192 字节。该上界可作为自动 materialize 的保守 canonical tmp 大小。
- 不带 tmp 的 3 参数 overload 支持模式 1 和模式 2；对 A2/A3 的模式 1，后端使用内部 8KB 缓冲区 `TMP_UB_OFFSET`；模式 2 不需要广播缓冲区。
- A5 硬件通过 `vlds` 广播模式原生支持行广播，tmp 被接口接受但不使用。

PTOAS 对用户显式 tmp 的合法性规则：

```text
A2/A3:
  op 必须是模式 1，才能使用显式 tmp。
  tmp 必须是 vec tile。
  tmp shape 必须静态可计算容量，或后续 verifier 能证明容量满足公式。
  tmp capacity >= min(ceil(R / 8) * 256, 7680)。

A5:
  tmp 被接受但不使用，不执行 A2/A3 tmp 容量校验。
```

### Pass 行为

对每个目标 row-expand 二元 op：

- 如果已经有 `tmp`，pass 不修改，但 verifier 需要保证它只用于合法模式。
- A5 如果没有 `tmp`，pass 不修改。
- A2/A3 如果没有 `tmp`，且 op 是模式 1、当前 build level 会运行 memplan，则自动补 tmp。
- A2/A3 如果没有 `tmp`，op 是模式 1、但当前 level3 会跳过 memplan，则报错，要求用户显式提供带地址的 tmp。
- 模式 2 不需要 tmp；pass 不应自动补 tmp，也不应强制改成带 tmp overload。

自动补 tmp 的 canonical shape 建议采用形状无关上界：

```mlir
%tmp = pto.alloc_tile
  : !pto.tile_buf<loc=vec, dtype=<dst element type>,
                  rows=1, cols=<8192 / sizeof(dst element type)>,
                  v_row=1, v_col=<8192 / sizeof(dst element type)>,
                  blayout=row_major, slayout=none_box,
                  fractal=512, pad=0>
```

默认使用 `dst element type` 作为 tmp element type，以贴合 row-expand 后端模板参数；如果后续确认某些后端实现允许更宽松的 tmp dtype，可在对应 op-specific verifier 中放宽。

这样不需要在 materialize pass 中依赖 `dst.validRow` 是否为静态值，也能覆盖 A2/A3 模式 1 的最大每轮 tmp 需求。后续如果希望节省 UB，可以在能静态证明 `R` 时生成更小 tmp：

```text
tmpBytes = min(ceil(R / 8) * 256, 7680)
```

### MemoryEffects

A2/A3 上这些 op 的 tmp 是广播 scratch buffer，应该建模为：

```text
Read(non-tmp inputs)
Read(tmp) if tmp exists
Write(tmp) if tmp exists
Write(dst)
```

其中 `Write(tmp)` 用于让 memplan 的 scratch-output conflict 禁止 tmp 与同 op 的 `dst` 错误复用。

A5 上 tmp 被接受但不使用，因此不应把 tmp 建模为 Read/Write：

```text
Read(non-tmp inputs)
Write(dst)
```

如果未来某个 row-expand op 的 MemoryEffects 不适合用 `Write(tmp)` 表达 scratch 语义，也应在 semantic no-alias side table 中显式加入：

```text
op = pto.trowexpand*
forbidAlias(tmp, dst)
```

### Verifier

这些 op 的 verifier 需要区分模式和 arch：

```text
如果 tmp 存在：
  A5: tmp 被接受但不使用，不执行 A2/A3 tmp 容量校验。
  A2/A3: op 必须是模式 1，即扩展操作数为 ColMajor 每行标量。
  A2/A3: tmp 必须是 vec tile。
  A2/A3: tmp capacity 必须满足 min(ceil(dst.validRow / 8) * 256, 7680)。
```

模式识别规则沿用 ISA 文档：

- `src0` 或 `src1` 中恰好一个与 `dst` 有相同 valid shape，该 operand 是全尺寸操作数。
- 另一个 operand 是扩展操作数。
- 扩展操作数为 `ColMajor` 且每行一个标量时是模式 1。
- 扩展操作数为 `RowMajor` 且每行 `32 / sizeof(T)` 列时是模式 2。

如果 `dst.validRow` 是动态值，verifier 无法精确证明用户显式 tmp 是否足够小时，可以采用保守规则：

- 用户显式 tmp 至少 8192 字节；或
- 后续引入运行时/符号约束证明 tmp capacity 满足公式。

自动生成 tmp 建议先使用 8192 字节 canonical 上界，因此不会受动态 `dst.validRow` 影响。

### 测试计划

#### lit：自动补 tmp

为至少一个代表 op 增加 A2/A3 模式 1 用例，例如 `pto.trowexpandadd(no tmp)`：

```text
CHECK: pto.alloc_tile
CHECK: pto.trowexpandadd ins(%{{.*}}, %{{.*}}, %{{.*}}
CHECK-NOT: memref.alloc
```

同时检查 A5 下不自动生成 tmp。

#### lit：模式 2 不补 tmp

构造 RowMajor 扩展操作数的模式 2 用例，确认 pass 不自动补 tmp，并继续走 no-tmp overload。

#### lit：memplan 回写 addr

检查自动生成 tmp 在 plan memory 后带 `addr`：

```text
CHECK: pto.alloc_tile addr =
CHECK: pto.trowexpand{{.*}} ins(%{{.*}}, %{{.*}}, %{{.*}}
```

legacy 和 modern 都应覆盖。

#### lit：level3 负例

A2/A3 level3 + 模式 1 + no tmp 应报错：

```text
expected-error {{requires explicit tmp when compiling at level3}}
```

A5 level3 + no tmp 不应因为 tmp 缺失报错。

#### lit：verifier 负例

需要覆盖：

- A2/A3 显式 tmp 用在模式 2，报错。
- A2/A3 显式 tmp capacity 小于公式要求，报错。
- A2/A3 动态 `dst.validRow` 且显式 tmp 小于 8192 字节，按保守规则报错。
- A5 显式 tmp 不触发 A2/A3 容量校验。

## 后续扩展

后续新增其它 optional tmp op 时，需要补充一个 op-specific 小节，并明确：

- op 的 tmp 后端接口规格。
- 自动生成 tmp 的 canonical shape。
- 用户显式 tmp 的 verifier 规则。
- tmp 的 MemoryEffects。
- tmp 与 input/output 的 alias 约束。
- level3 下是否要求显式 tmp。
- lit 覆盖自动补 tmp、memplan 回写 addr、EmitC overload、level3 负例和 verifier 负例。
