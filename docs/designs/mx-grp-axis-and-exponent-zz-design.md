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

- 状态：设计提案，待评审。本 PR 当前只含设计文档，评审通过后在同一 PR 内实现。
- 关联 issue：[#1185](https://github.com/hw-native-sys/PTOAS/issues/1185)
- 设计复核基线：
  - PTOAS `988d50e24`
  - pto-isa 本地快照 `7af803bc4056af8b39a55751ac2f4b75cdb47fbd`（下称"快照"）
- 目标接口：`pto.tquant.mx`、exponent 布局转换 op

本文只定义 A5 MX 量化的分组轴（`grp_axis`）表达，以及 E8M0 exponent 从 ND/DN
到 ZZ 的布局转换在 PTO IR 与 EmitC 中的表达方式。不改变 INT8 量化、不改变
`pto.tmov` 现有的 ND-to-NZ / acc-to-vec / FP 语义。

## 2. 结论摘要

1. **不新增 `mx_alg` 属性。** PTOAS 已有的 `quant_type` × `quantScaleAlg` 与
   pto-isa `MxQuantAlg` 是精确的一一映射，`MxQuantAlg` 由 EmitC 合成。
2. **`pto.tquant.mx` 新增 `grp_axis` 与 `interleave` 属性**，默认
   `grp_axis = 1`、`interleave = false`，保持现有 IR 逐字兼容。
3. **verifier 的分组 shape 校验按轴分派**，替换当前"只比较总元素数"的写法。
4. **exponent X-to-ZZ 新增独立 op `pto.tmov.x2zz`**，不扩展 `pto.tmov`。
5. **现有 `exp_zz` + `storeMode` fused 路径标记为 deprecated**，理由见 §5.4：
   该 overload 在快照里只有 CPU-sim 实现，A5 设备头文件没有对应实现。

## 3. PTO-ISA 侧现状（已对快照逐条核对）

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

三条必须写进设计的事实：

1. **SFINAE 用 `Tmp::Loc != TileType::Scaling` 把 X-to-ZZ 与 FP 三参数 TMOV 区分开。**
   若 `tmp` 落在 scaling 地址空间，会静默命中 FP overload。
2. **ND 用 `dst` 的 valid shape 驱动，DN 用 `src` 的 valid shape 驱动。** 这是不对称的，
   verifier 与文档都必须显式说明，否则用户会把 valid 写在错误的一侧。
3. 默认模板参数 `grp_axis = 1`，所以不带模板参数的三参数 `TMOV` 即 ND-to-ZZ。

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
              = BLOCK_SIZE + rowBlockCount * validCol          // 化简，BLOCK_SIZE = 32
              = 32 + ceil(validRow / 16) * validCol
```

其中 `validRow` / `validCol` 取自 **`dst`**（见上面第 2 条）。

DN-to-ZZ 的 `tmp` 用量在 `GenerateB8IndicesDN2ZZToUB` 内部，快照里没有等价的闭式表达。
本设计**不在 verifier 里推导 DN 的 `tmp` 下界**，只做类型/布局/地址空间校验，容量由
调用方保证；见 §13 开放问题 1。

### 3.3 fused NZ store overload 的真实位置

`pto.tquant.mx` 现有的 `exp_zz` + `storeMode` 形态对应 pto-isa 的六 tile overload。
快照里：

- 公共 wrapper 在 `include/pto/common/pto_instr.hpp:2460`；
- **实现只在 `include/pto/cpu/TQuant.hpp:545`（CPU-sim）**；
- `include/pto/npu/a5/TQuant.hpp` 里 grep `VecStoreMode` / `store_mode` **无任何命中**。

即：该形态在 `__CPU_SIM` 下可用，A5 设备路径没有对应实现。这是 §5.4 决策的直接依据。
合入前需按仓库当前 pin 的 pto-isa 复核该结论（见 §13 开放问题 2）。

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

`pto.tmov` 同理：现在生成 `TMOV<DstT, SrcT, ...>(dst, src, ...)`，而 X-to-ZZ 需要
`TMOV(dst, src, tmp)` 或 `TMOV<0>(dst, src, tmp)`。

### 4.4 `pto.tmov` 现状

`def TMovOp`：`src`、`dst`、`Optional fp`、`Optional preQuantScalar`、`accToVecMode`、
`reluPreMode`。verifier 要求 `fp` 位于 scaling 地址空间。没有 `tmp`，没有 `grp_axis`。

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

### 5.3 决策三：X-to-ZZ 用独立 op `pto.tmov.x2zz`

**不扩展 `pto.tmov`**，理由三条：

1. **verifier 强度**：`pto.tmov` 的核心约束之一是 src/dst 同 shape；X-to-ZZ 恰恰是
   shape 变换（`[M, N/32]` → ZZ box）。塞进同一个 op 会迫使把这条规则改成按属性分派，
   削弱既有路径的检查强度。
2. **EmitC 形态不同**（§4.3）：一个发 `TMOV<DstT, SrcT, ...>`，一个发 `TMOV<0>`。
3. **前车之鉴**：#1122 把三个 op 合并成 `pto.tfillpad` + `mode` 属性后，`ExpandTileOp`
   漏读了该属性，导致 in-place 被静默当成 normal 展开，靠后续提交才补上。`pto.tmov`
   的消费者比 `tfillpad` 多得多（EmitC、ExpandTileOp、PlanMemory、InsertSync、ptobc），
   每一个都要正确分派新形态的风险不划算。

**代价**：新增一个 op 需要同步 ODS / verifier / EmitC / PTO-BC opcode / Python binding /
手册。§12 已按此排期。

### 5.4 决策四：`exp_zz` + `storeMode` 标记 deprecated

**做法**：本次不删除，行为不变，但：

- verifier 增加一条：`exp_zz` 形态与 `grp_axis = 0` **互斥**（该 fused 路径只有 ND 语义）；
- ODS `description` 与用户手册标注 deprecated，推荐改用 `pto.tquant.mx` + `pto.tmov.x2zz`；
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

### 6.2 新 op `pto.tmov.x2zz`

```tablegen
def TMovX2ZzOp : PTO_TOp<"tmov.x2zz", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Convert an E8M0 exponent tile from ND/DN grouping to ZZ box layout.";

  let arguments = (ins
    PTODpsType:$src,          // ND: [M, N/32]；DN: [M/32, N]
    PTODpsType:$dst,          // ZZ box 布局的目标 tile
    PTODpsType:$tmp,          // vec scratch，禁止 loc=scaling
    DefaultValuedAttr<PTO_MxGroupAxisAttr, "::mlir::pto::MxGroupAxis::Axis1">:$grpAxis
  );
  let results = (outs);
  let hasVerifier = 1;
  let assemblyFormat = [{
    `ins` `(` $src `,` $tmp `:` qualified(type($src)) `,` qualified(type($tmp)) `)`
    `outs` `(` $dst `:` qualified(type($dst)) `)` attr-dict
  }];
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() {
      return ::mlir::MutableOperandRange(getOperation(), 1, 1);
    }
  }];
}
```

IR 形态：

```mlir
pto.tmov.x2zz ins(%exp, %tmp : !pto.tile_buf<...>, !pto.tile_buf<...>)
              outs(%exp_zz : !pto.tile_buf<...>)
              {grpAxis = #pto<mx_group_axis axis0>}
```

`tmp` 放在 `ins` 而不是 `outs`：它是被覆写的 scratch，但不承载结果语义。DPS init 只
包含 `dst`；`tmp` 的写效应由 §9 的 memory effects 表达。

## 7. Verifier 设计

### 7.1 `TQuantMxOp::verify()` 改造

替换 §4.2 引用的那段轴无关校验，改为：

```cpp
const bool isDn = getGrpAxis() == MxGroupAxis::Axis0;

// (1) 分组轴上的整除约束
if (isDn) {
  if (srcRows != kDynamic && srcRows % 32 != 0)
    return emitOpError("expects src valid_shape[0] to be a multiple of 32 when grpAxis is axis0");
} else {
  if (srcCols != kDynamic && srcCols % 32 != 0)
    return emitOpError("expects src valid_shape[1] to be a multiple of 32 when grpAxis is axis1");
}

// (2) 分组 tile 的逐维 shape，而不再是总元素数
const int64_t expRows = isDn ? srcRows / 32 : srcRows;
const int64_t expCols = isDn ? srcCols      : srcCols / 32;
for (auto [name, ty] : {{"exp", expTy}, {"max", maxTy}, {"scaling", scalingTy}}) {
  auto v = getValidShapeVec(ty);
  if (v[0] != kDynamic && v[0] != expRows)
    return emitOpError() << "expects " << name << " valid_shape[0] to be " << expRows
                         << " for grpAxis=" << stringifyMxGroupAxis(getGrpAxis());
  if (v[1] != kDynamic && v[1] != expCols)
    return emitOpError() << "expects " << name << " valid_shape[1] to be " << expCols << " ...";
}
```

**注意**：这是对现有 ND IR 的**收紧**——原来只要总元素数对就通过，现在要求逐维匹配。
需要评估存量 IR 影响，处理方式见 §11.1 的 `ND-SHAPE-STRICT` 用例与 §13 开放问题 3。

`interleave` 相关：

```cpp
if (getInterleave()) {
  if (!isDn)
    return emitOpError("expects interleave to be used only with grpAxis=axis0");
  // interleave 输出 exponent 形状为 [ceil(M/32/2), 2N]，即 [ceil(srcRows/64), 2*srcCols]
  //   —— 只作用于 exp，dst/max/scaling 不受影响
  if (expValid[0] != kDynamic && expValid[0] != ceilDiv(srcRows, 64))
    return emitOpError("expects interleaved exp valid_shape[0] to be ceil(src rows / 64)");
  if (expValid[1] != kDynamic && expValid[1] != 2 * srcCols)
    return emitOpError("expects interleaved exp valid_shape[1] to be 2 * src cols");
}
```

`exp_zz` 互斥（§5.4）：

```cpp
if (getExpZz() && isDn)
  return emitOpError("expects the deprecated exp_zz form to use grpAxis=axis1; "
                     "use pto.tmov.x2zz for axis0 exponents");
```

### 7.2 `TMovX2ZzOp::verify()`

按 §3.2 的 pto-isa 约束逐条落地。A5-only，通过 `dispatchVerifierByArch` 在 A2/A3 上直接拒绝。

| # | 规则 | 诊断文案（草案） |
|---|---|---|
| 1 | `src`/`dst`/`tmp` 均为 vec tile | `expects src/dst/tmp to be vec tiles` |
| 2 | `tmp` 不得位于 scaling 地址空间 | `expects tmp not to be in the scaling address space; a scaling tmp selects the FP TMOV overload in PTO-ISA` |
| 3 | 三者元素类型相同 | `expects src, dst, and tmp to share one element type` |
| 4 | 元素类型 ∈ {ui8, i8, `!pto.f8E8M0`} | `expects an 8-bit exponent element type (ui8/i8/f8E8M0)` |
| 5 | `src` 为 `row_major` + `none_box` | `expects src to use blayout=row_major, slayout=none_box` |
| 6 | `dst` 为 `row_major` + `slayout=row_major` | `expects dst to use blayout=row_major, slayout=row_major (ZZ box)` |
| 7 | rank-2 valid shape | `expects rank-2 valid_shape for src/dst/tmp` |
| 8 | ND：`dstRows % 16 == 0 && dstCols % 64 == 0` | `expects ND-to-ZZ dst valid_shape to satisfy rows % 16 == 0 and cols % 64 == 0` |
| 9 | DN：`srcRows` 对应 `M/32`，要求 `M % 64 == 0`，即 `srcRows % 2 == 0`；`srcRows == 1` 为退化 identity | `expects DN-to-ZZ src valid_shape[0] to be even (M % 64 == 0), or 1 for the degenerate identity case` |
| 10 | DN：`srcCols % 16 == 0` | `expects DN-to-ZZ src valid_shape[1] to be a multiple of 16` |
| 11 | ND：`tmp` 物理容量 ≥ `32 + ceil(dstRows / 16) * dstCols` 字节（§3.2） | `expects tmp to provide at least <N> bytes for ND-to-ZZ (32 + ceil(dst rows / 16) * dst cols)` |
| 12 | src/dst 元素总数相等 | `expects src and dst to hold the same exponent count` |

规则 8/9/10 的驱动侧不同（ND 看 `dst`，DN 看 `src`），实现时**必须按轴选择被检查的 tile**，
这一点在 §3.2 已单独标注。

规则 11 只对 ND 施加；DN 的 `tmp` 下界见 §13 开放问题 1，先不做数值校验。

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

**兼容性取舍**：现有 ND IR（不写 `grpAxis`）会从形态 A/旧列表切到形态 B，生成的 C++
文本因此改变（模板参数从"类型列表"变成 `<1, MxQuantAlg::...>`）。语义等价，但会改动
既有 lit 期望值，见 §11.1 与 §12 步骤 5。若评审希望零文本变更，可改为"仅当显式写了
`grpAxis`/`interleave` 时才走形态 B"，代价是同一语义存在两种输出形态；本设计**倾向
统一到形态 B**，理由是长期只维护一条 grouped 路径。

### 8.2 `pto.tmov.x2zz`

新增 `PTOTMovX2ZzToEmitC`：

```cpp
// grpAxis = axis1（默认）
TMOV(dst, src, tmp);
// grpAxis = axis0
TMOV<0>(dst, src, tmp);
```

`axis1` **不写模板参数**（依赖 pto-isa 的 `grp_axis = 1` 默认值），与 issue 期望一致；
`axis0` 写 `<0>`。tile 类型全部由实参推导，不进模板参数列表。

## 9. 内存效应、内存规划与 liveness

`TMovX2ZzOp` 的 `MemoryEffectsOpInterface` 实现：

| operand | effect |
|---|---|
| `src` | `MemoryEffects::Read` |
| `dst` | `MemoryEffects::Write` |
| `tmp` | `MemoryEffects::Read` + `MemoryEffects::Write` |

`tmp` 必须同时声明读写，否则：

- PlanMemory 可能把 `tmp` 与仍然活跃的 buffer 复用；
- CSE / DCE 可能认为写 `tmp` 无副作用而删除或合并；
- InsertSync 的宏模型拿不到该 tile 的依赖边。

`pto.tquant.mx` 的效应不变（新增的是属性，不是 operand）。

`getPipe()` 返回 `PIPE_V`，与其他 vec 侧 tile op 一致，InsertSync 沿用现有 vec 路径。

## 10. 需要同步的层

按 `.claude/rules/cross-layer-sync.md` 逐层列出：

| 层 | 改动 |
|---|---|
| ODS | `PTOAttrs.td` 新增 `MxGroupAxis`；`PTOOps.td` 给 `TQuantMxOp` 加两个属性、新增 `TMovX2ZzOp` |
| IR / verifier | `PTO.cpp`：`TQuantMxOp::verify()` 按轴分派；新增 `TMovX2ZzOp::verify()`；`TQuantMxOp` 自定义 parser/printer 同步新属性 |
| EmitC | `PTOToEmitC.cpp`：`PTOQuantMxToEmitC` 二分；新增 `PTOTMovX2ZzToEmitC` 并注册进 `populatePTOToEmitCPatterns` |
| CAPI | `include/pto-c/Dialect/PTO.h` + `lib/CAPI/Dialect/PTO.cpp`：新枚举的 C 入口（参照 `QuantScaleAlg` 现有写法） |
| Python binding | `lib/Bindings/Python/PTOModule.cpp` 暴露 `MxGroupAxis`；`python/pto/dialects/pto.py` 同步 |
| PTO-BC | `tools/ptobc/generated/ptobc_opcodes_v0.h` 给 `pto.tmov.x2zz` 分配**新 opcode**（取当前未使用值，不复用空洞）；新属性走属性字典，不改 `pto.tquant.mx` 的既有 payload schema |
| 文档 | `docs/PTO_IR_manual.md`（两个 op 章节）、`docs/release/PTO-tile-Instruction-SPEC-v0.4.md`、本设计文档 |
| 测试 | 见 §11 |
| ReleaseNotes | 新增 `pto.tmov.x2zz`、`grpAxis`/`interleave`；标注 `exp_zz`/`storeMode` deprecated；说明 §8.1 的生成文本变化 |

**PTO-BC 注意事项**：`pto.tquant.mx` 的 opcode payload schema **不得改动**（参考 #1122 的
教训：同一 opcode 下改 payload 会造成旧字节码静默错位）。新属性通过属性字典编码，
`operand_mode` / `num_operands` 保持原值；新增 op 用新 opcode。
`tools/ptobc/tests/v0_fp_schema_compatibility_check.py` 的期望表需要相应扩展。

## 11. 测试方案

### 11.1 lit 正向：`test/lit/pto/tquant_mx_grp_axis_emitc.pto`

```
// RUN: ptoas --pto-arch=a5 --pto-level=level3 %s | FileCheck %s --check-prefix=EMITC
// RUN: not ptoas --pto-arch=a3 %s 2>&1 | FileCheck %s --check-prefix=A3-REJECT
```

用例矩阵（同一文件多个 `func.func`）：

| 函数 | `grpAxis` | `quant_type` × `quantScaleAlg` | src valid | exp/max/scaling valid | 期望模板参数 |
|---|---|---|---|---|---|
| `nd_fp8_ocp` | 默认（不写） | MXFP8 × OCP | `16x64` | `16x2` | `TQUANT<1, MxQuantAlg::OcpMxFp8E4M3>` |
| `nd_fp8_nv` | axis1 | MXFP8 × NV | `16x64` | `16x2` | `TQUANT<1, MxQuantAlg::NvMxFp8E4M3>` |
| `dn_fp8_ocp` | axis0 | MXFP8 × OCP | `64x16` | `2x16` | `TQUANT<0, MxQuantAlg::OcpMxFp8E4M3>` |
| `dn_fp4_ocp` | axis0 | MXFP4_E2M1 × OCP | `64x16` (f16) | `2x16` | `TQUANT<0, MxQuantAlg::OcpMxFp4E2M1>` |
| `dn_fp8_interleave` | axis0 | MXFP8 × OCP | `128x16` | exp `2x32`（`ceil(128/64) x 2*16`）；max/scaling `4x16` | `TQUANT<0, MxQuantAlg::OcpMxFp8E4M3, true>` |

CHECK 要点：
- 断言**完整的模板参数串**，而不只是 `TQUANT(`，否则 axis/alg 写错测不出来；
- 加 `// EMITC-NOT: VecStoreMode`，确认新路径不再走 fused 形态；
- `A3-REJECT` 断言 `tquant.mx is only supported on A5`。

### 11.2 lit 正向：`test/lit/pto/tmov_x2zz_emitc.pto`

| 函数 | `grpAxis` | 期望生成 |
|---|---|---|
| `nd_to_zz` | 默认 | `TMOV({{.*}}, {{.*}}, {{.*}});` 且 `CHECK-NOT: TMOV<` |
| `dn_to_zz` | axis0 | `TMOV<0>({{.*}}, {{.*}}, {{.*}});` |

同文件追加一条 `--mlir-print-ir-after=pto-plan-memory` 的 RUN，断言 `tmp` 参与内存规划
（不与其他活跃 buffer 复用），把 §9 的效应声明钉住。

### 11.3 lit 负向：`test/lit/pto/tquant_mx_grp_axis_invalid.pto`

用 `split-file` 拆分，逐条覆盖 §7.1：

| 子用例 | 期望诊断 |
|---|---|
| `dn_rows_not_aligned` | `expects src valid_shape[0] to be a multiple of 32 when grpAxis is axis0` |
| `dn_wrong_exp_shape`（给 `[M, N/32]`） | `expects exp valid_shape[0] to be ...` |
| `nd_wrong_exp_shape`（给 `[M/32, N]`） | 同上 ND 版本 |
| `interleave_on_nd` | `expects interleave to be used only with grpAxis=axis0` |
| `interleave_wrong_exp_shape` | `expects interleaved exp valid_shape[...]` |
| `exp_zz_with_axis0` | `expects the deprecated exp_zz form to use grpAxis=axis1` |

### 11.4 lit 负向：`test/lit/pto/tmov_x2zz_invalid.pto`

逐条覆盖 §7.2 的 12 条规则，重点包含：

- `tmp_in_scaling_space`（规则 2）——这条最容易被忽略且后果最隐蔽；
- `mismatched_elem_type`（规则 3）；
- `src_wrong_layout` / `dst_wrong_layout`（规则 5/6）；
- `nd_dst_cols_not_64`（规则 8）；
- `dn_src_rows_odd`（规则 9）；
- `tmp_too_small`（规则 11）；
- `on_a3`（arch 拒绝）。

### 11.5 精度用例：`test/samples/TquantMxDn/`

对照现有 `test/samples/TquantMx/` 的结构新建：

```
test/samples/TquantMxDn/
├── tquant_mx_dn.pto             # DN 量化 + pto.tmov.x2zz 完整链路
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

容差沿用 `tquant_mx_compare.py` 的做法：

| 输出 | dtype | 容差 |
|---|---|---|
| `dst` | int8（fp8 位型） | `atol = 0`（逐字节相等） |
| `exp` | uint8 | `atol = 0` |
| `exp_zz` | uint8 | `atol = 0` |
| `max` | float32 | `atol = 1e-5` |
| `scaling` | float32 | `atol = 1e-5` |

exponent 与量化结果都是**位精确**的，不允许容差；只有 `max`/`scaling` 走浮点容差。

同时给 `test/samples/TquantMx/` 追加一条 ND + `pto.tmov.x2zz` 的用例，覆盖 ND-to-ZZ。

`runop.sh` 的 `PTO_PTO_DIRS` 需要把 `TquantMxDn` 加入默认列表。

### 11.6 ST 用例：`test/tilelang_st/npu/a5/src/st/testcase/tquant_mx_dn/`

对照 `testcase/tmatmul_mx/` 的结构：`cases.py`（唯一真值源）、`gen_data.py`、
`compare.py`、`main.cpp`、`launch.cpp`、`CMakeLists.txt`、`tquant_mx_dn.pto`。

`cases.py` 建议的 case 维度：

```python
CASES = [
    {"name": "dn_fp8_ocp_64x64",    "m": 64,  "n": 64,  "grp_axis": 0, "alg": "ocp", "eps": 0.0},
    {"name": "dn_fp8_ocp_128x32",   "m": 128, "n": 32,  "grp_axis": 0, "alg": "ocp", "eps": 0.0},
    {"name": "dn_fp8_nv_64x64",     "m": 64,  "n": 64,  "grp_axis": 0, "alg": "nv",  "eps": 0.0},
    {"name": "dn_fp4_ocp_64x64",    "m": 64,  "n": 64,  "grp_axis": 0, "alg": "ocp", "fp4": True},
    {"name": "dn_identity_32x64",   "m": 32,  "n": 64,  "grp_axis": 0, "alg": "ocp"},  # 退化 identity
    {"name": "nd_fp8_ocp_16x128",   "m": 16,  "n": 128, "grp_axis": 1, "alg": "ocp"},
]
```

`m = 32`（DN-to-ZZ 退化为 identity）必须单独成 case，它是 §7.2 规则 9 的边界。

### 11.7 PTO-BC roundtrip

参照 `tools/ptobc/tests/fp_operand_forms_v0_encode.sh` 的范式新建
`tools/ptobc/tests/mx_grp_axis_v0_encode.sh` + testdata：

1. `ptobc encode` → `ptobc decode`；
2. grep 断言 `grpAxis = #pto<mx_group_axis axis0>`、`interleave = true`、`pto.tmov.x2zz` 都还在；
3. **把 roundtrip 结果重新喂给 `ptoas --emit-pto-ir` 验证**（光 grep 文本证明不了属性字典还自洽）；
4. 扩展 `v0_fp_schema_compatibility_check.py`，把 `pto.tquant.mx` 现有 opcode 的
   `(operand_mode, num_operands)` 钉进期望表，防止后续误改 payload。

### 11.8 运行方式

```bash
# 0) 构建（本地开发树；LLVM_BUILD_DIR 指向已有 LLVM 构建）
LLVM_BUILD_DIR=<llvm-build> ./quick_install.sh
#    或直接用 ninja
ninja -C build ptoas

# 1) 定向 lit（开发迭代用，最快）
ninja -C build ptoas
build/bin/llvm-lit -sv build/test/lit/pto/tquant_mx_grp_axis_emitc.pto \
                       build/test/lit/pto/tquant_mx_grp_axis_invalid.pto \
                       build/test/lit/pto/tmov_x2zz_emitc.pto \
                       build/test/lit/pto/tmov_x2zz_invalid.pto

# 2) 全量 lit
ninja -C build check-pto

# 3) ctest（含 PTO-BC roundtrip 与 schema 守卫）
ninja -C build check-ctest
ctest --test-dir build -R 'ptobc_mx_grp_axis_v0_encode|ptobc_v0_fp_schema_compatibility_check' --output-on-failure

# 4) 精度样例（py -> pto -> cpp，本地不含板卡）
PTOAS_BIN=build/tools/ptoas/ptoas ./test/samples/runop.sh -t TquantMxDn
PTOAS_BIN=build/tools/ptoas/ptoas ./test/samples/runop.sh -t TquantMx

# 5) 生成 C++ 的编译验证（需要匹配的 pto-isa/CANN）
#    确认 TQUANT<0, MxQuantAlg::...> 与 TMOV<0>(...) 能被 pto-isa 头文件解析
#    参考 docs/no_npu_compile_only_guide_zh.md

# 6) A5 ST（需要板卡或 CPU-sim 环境）
#    参考 test/tilelang_st/npu/a5 的既有流程运行 tquant_mx_dn
```

**验收前必须跑完 1–5；6 在具备环境时补充。** 第 5 步不能省：本特性的全部价值就是生成
能被 pto-isa 接受的调用，lit 只能证明文本形态。

## 12. 实施顺序

建议按可独立回滚的粒度切成 7 个提交：

| # | 内容 | 验证 |
|---|---|---|
| 1 | `MxGroupAxis` 枚举 + `TQuantMxOp` 两个属性（ODS/parser/printer/CAPI/Python） | 现有 lit 全绿（默认值保证零行为变化） |
| 2 | `TQuantMxOp::verify()` 按轴分派 + §11.3 负向用例 | 定向 lit |
| 3 | `PTOQuantMxToEmitC` 形态 B + §11.1 正向用例 + 更新受影响的既有 lit 期望 | 定向 lit + `check-pto` |
| 4 | 新增 `TMovX2ZzOp`（ODS + verifier + memory effects）+ §11.4 负向用例 | 定向 lit |
| 5 | `PTOTMovX2ZzToEmitC` + §11.2 正向用例 | 定向 lit + `check-pto` |
| 6 | PTO-BC opcode + roundtrip 用例 + schema 守卫扩展 | `check-ctest` |
| 7 | 精度样例 + ST 用例 + 手册 / SPEC / ReleaseNotes | `runop.sh` + 全量 lit |

提交 3 会改动既有 lit 的期望文本（§8.1），单独成一个提交便于 review 和回滚。

## 13. 风险与开放问题

1. **DN-to-ZZ 的 `tmp` 容量下界未知。** 快照里 ND 有闭式公式，DN 的用量在
   `GenerateB8IndicesDN2ZZToUB` 内部。本设计先不做数值校验（只校验类型/布局/地址空间）。
   → **行动**：实现前从 pin 的 pto-isa 读出 DN 的实际用量并补进 §7.2 规则 11；在 ST 里用
   一个刚好够大的 `tmp` 和一个偏小的 `tmp` 做对照，确认行为。
2. **pto-isa pin 是否包含所需接口。** issue 要求"至少 `f03c2454`"。仓库当前 pin 的是
   `ce3262e3`（`ci.yml` / `docker/Dockerfile` / `run_remote_npu_validation.sh` /
   `ci_sim.yml` / `no_npu_compile_only_guide_zh.md` 五处，由
   `.github/scripts/update_pto_isa_pin.py` 统一管理）。
   → **行动**：实现前先确认 `ce3262e3` 是否含 §3.1/§3.2 的接口；不含则本特性需连同
   pin bump 一起推进，且五处必须同步（用该脚本，勿手改）。
3. **ND 分组 shape 校验收紧的影响面。** §7.1 把"总元素数相等"改成"逐维相等"。
   存量 IR 里若有 `[1, M*N/32]` 这类扁平写法会被新 verifier 拒绝。
   → **行动**：实现前先跑一遍 `check-pto` + `runop.sh all` 摸底；若确有存量，考虑对
   `grpAxis = axis1` 保留"总元素数"宽松路径一个版本，并在 ReleaseNotes 中给出迁移说明。
4. **§8.1 的生成文本变化。** 现有 ND IR 的输出会从"类型列表模板参数"变成
   `<1, MxQuantAlg::...>`。语义等价但文本变化，属于对下游 golden/期望文件的影响面。
   → **行动**：提交 3 单独处理，并在 ReleaseNotes 说明。
5. **`exp_zz` fused 路径的最终去向。** 本次只标 deprecated。若确认其设备侧不可用，
   应单开 PR 删除，并按 #1122 的做法处理 PTO-BC 兼容（保留 wire alias 或明确不兼容）。
6. **`interleave` 的 exponent 形状** 取自 issue 的 `[ceil(M/64), 2N]`。
   → **行动**：实现前用 pin 的 pto-isa 复核 DN interleave 实现的实际写出形状，以代码为准。

## 14. 验收标准

- issue #1185 列出的四条契约（axis 0/1 TQUANT、RowMajor 量化数据 + raw ND/DN exponent、
  数据 ND-to-NZ、exponent ND/DN-to-ZZ）都能用 PTO IR 表达；
- `grpAxis` / `interleave` / `pto.tmov.x2zz` 生成的 C++ 与 §8 的模板参数表逐字一致；
- 不写新属性的现有 `pto.tquant.mx` IR **语义不变**，且 §11.1 有用例覆盖；
- §7 的每一条 verifier 规则都有对应的负向 lit 用例；
- DN 量化 + DN-to-ZZ 的精度用例在 exponent 与量化结果上**位精确**通过；
- PTO-BC roundtrip 保持 `pto.tquant.mx` 既有 payload schema 不变，新 op 使用新 opcode；
- `docs/PTO_IR_manual.md` 与 SPEC 中的 shape 表、约束表与 verifier 实现一致。
