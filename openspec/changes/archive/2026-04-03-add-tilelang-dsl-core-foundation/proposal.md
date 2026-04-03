# Proposal: 建立 `tilelang-dsl/` 独立前端的 v1 基础契约

## 概述

`docs/tilelang-dsl-guide.md` 已经描述了一套面向 Tile/TensorView authoring 的 Python DSL，但仓库中还没有与该文档对齐的 OpenSpec change，也没有把这套 surface 与现有 `python/pto/dialects/pto.py` 的实验性 VPTO Python DSL 做清晰切分。  
本 change 先落定 TileLang DSL v1 的基础契约：在 `tilelang-dsl/` 下建立独立前端目录、独立 package、独立测试与文档边界，并把首版 public surface、参数定型方式、Tile specialization 机制和 frontend diagnostics 固化为 OpenSpec。

## 背景与动机

当前存在三类直接问题：

1. `docs/tilelang-dsl-guide.md` 的 surface 尚未被 OpenSpec 约束

- 文档里已经暴露了 `@pto.vkernel`、`TensorView`、`Tile`、高层 `dma_load/dma_store`、typed-mask、element-indexing 等 surface。
- 这些 surface 没有对应 capability spec，后续实现边界、失败行为和测试覆盖都无法稳定收敛。

2. 现有 `python/pto/dialects/pto.py` 中的实验实现不是合适的 source of truth

- 当前文件把 PTO dialect Python bindings 和一个实验性的 `@pto.vkernel` parser 混在一起。
- 它使用的是另一套更接近 hand-written VPTO 的 surface，和 `docs/tilelang-dsl-guide.md` 的目标 DSL 并不等价。
- 用户已明确要求本特性不依赖现有其他 Python binding 实现，并要求把本特性相关工作集中在 `tilelang-dsl/` 目录。

3. `vpto-ir-legality` 的 vecscope OpenSpec 契约与真实实现/测试不一致

- 现有 `openspec/specs/vpto-ir-legality/spec.md` 仍把 `scf.for {llvm.loop.aivector_scope}` 记作 authoring-form carrier。
- 当前 verifier 与回归已经明确拒绝 legacy `scf.for {llvm.loop.aivector_scope}`，要求使用 dedicated `pto.vecscope/pto.strict_vecscope`。
- TileLang DSL 如果要稳定 lower 到 authoring-form VPTO，必须先把这一契约纠正到与真实实现一致。

## 目标

- 在 `tilelang-dsl/` 下定义独立的 TileLang DSL v1 package、源码边界、测试边界和文档边界。
- 固定 v1 `@pto.vkernel` 的最小 public surface：`a5` 单 target、monomorphic `dtypes`、bare `TensorView`/`Tile` 参数注解、descriptor API、Tile specialization。
- 固定 frontend diagnostics 契约，确保 unsupported matcher feature、unsupported Python syntax、Tile specialization 缺失、非法 shape profile 都能 fail-fast。
- 通过 OpenSpec 修正 `vpto-ir-legality` 中 vecscope 相关 requirement，使其与当前 verifier / lit 回归一致。

## 非目标

- 不在本 change 中实现 DSL -> VPTO lowering；该部分由后续 `add-tilelang-dsl-authoring-vpto-lowering` change 覆盖。
- 不在本 change 中引入 kernel matcher、`Any*` / `TypeVar`、多 signature `dtypes`、`constraints`、`priority`。
- 不要求复用或改造现有 `python/pto/dialects/pto.py` 的实验 `@pto.vkernel` 实现。
- 不为 `a5` 之外的 target 建模。
- 不在本 change 中扩展到 implicit vecscope inference、raw pointer authoring、advanced vector family。

## 变更内容

- 新增 `tilelang-dsl-surface` capability，定义独立 package、repo layout、v1 `@pto.vkernel` surface、参数定型方式、descriptor API 与 Tile specialization 契约。
- 新增 `tilelang-dsl-diagnostics` capability，定义 frontend 对 unsupported feature、unsupported syntax、specialization 缺失和 shape profile 错误的诊断义务。
- 修改 `vpto-ir-legality` capability，修正 authoring-form vecscope carrier 的 requirement：以 dedicated `pto.vecscope/pto.strict_vecscope` 为准，并继续拒绝 legacy `scf.for {llvm.loop.aivector_scope}`。

## Capabilities

### New Capabilities

- `tilelang-dsl-surface`: 定义 `tilelang-dsl/` 独立前端的 v1 public surface、package 入口、descriptor API、参数定型与 Tile specialization 契约。
- `tilelang-dsl-diagnostics`: 定义 TileLang DSL v1 frontend 的 fail-fast diagnostics、错误定位与错误分层契约。

### Modified Capabilities

- `vpto-ir-legality`: 修正 authoring-form VPTO vecscope carrier 契约，使 OpenSpec 与当前 verifier / regression 使用的 dedicated `pto.vecscope/pto.strict_vecscope` 语义一致。

## 预期结果

- `tilelang-dsl/` 成为本特性的唯一源码、样例、测试和局部文档承载目录；根目录只保留最小 build/install/test 接线。
- TileLang DSL v1 的 public surface 和 diagnostics 行为不再依赖 `docs/tilelang-dsl-guide.md` 的口头描述或现有实验实现，而是有明确 OpenSpec 契约。
- 后续 DSL -> VPTO lowering change 可以直接依附真实的 authoring-form VPTO legality contract，而不是继续踩在错误的 `llvm.loop.aivector_scope` requirement 上。

## 成功标准

- 新增 `openspec/changes/add-tilelang-dsl-core-foundation/`，包含 proposal、design、tasks。
- 新增 `specs/tilelang-dsl-surface/spec.md` 和 `specs/tilelang-dsl-diagnostics/spec.md`。
- 新增 `specs/vpto-ir-legality/spec.md` delta，明确 legacy `scf.for {llvm.loop.aivector_scope}` 不再是合法 authoring-form carrier。
- proposal/design/tasks 明确写清：
  - `tilelang-dsl/` 是本特性的唯一工作目录；
  - v1 只接受 `a5` 单 target 和单一 monomorphic `dtypes` signature；
  - bare `Tile` 参数必须先 specialization 再 materialize IR；
  - unsupported matcher feature 与 unsupported syntax 必须 fail-fast。

## 影响

- 受影响目录：
  - `tilelang-dsl/`
  - `openspec/specs/vpto-ir-legality/spec.md`
  - 必要的根级 CMake / 安装 / 测试入口接线
- 受影响 public API：
  - 新增独立 package `tilelang_dsl`
  - 新增 v1 descriptor API：`specialize()`, `mlir_text()`, `mlir_module()`, `verify()`, `emit(path)`
- 对现有 `python/pto/dialects/pto.py` 的要求：
  - 不再作为本特性的 source of truth
  - 如需接线，只允许最小兼容或安装 wiring，不允许把 TileLang DSL 核心逻辑继续堆叠在该文件内
