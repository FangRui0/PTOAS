## Context

### 范围

本 design 只覆盖 TileLang DSL v1 的基础前端契约，不覆盖 DSL -> VPTO lowering 本身。  
它回答四个问题：

1. TileLang DSL v1 的代码、测试、样例、文档应该放在哪里
2. v1 `@pto.vkernel` 的 public surface 到哪一层
3. bare `TensorView` / `Tile` 参数如何定型
4. frontend 必须在哪些点 fail-fast，而不是把非法输入拖到 lowering / verifier 阶段

### 当前状态

当前仓库里有三套彼此未完全对齐的事实：

1. `docs/tilelang-dsl-guide.md`

- 它描述了高层 TileLang DSL 的理想 surface，包括 `TensorView`、`Tile`、高层 DMA、mask inference、matcher、implicit vecscope inference 等。
- 该文档覆盖面很大，但没有被 OpenSpec capability 拆成可实现的 v1/v2 边界。

2. `python/pto/dialects/pto.py`

- 该文件当前主要服务于 PTO dialect Python bindings。
- 其中夹带了一个实验性的 `@pto.vkernel` parser，surface 更接近直接 author VPTO，不等于 TileLang DSL guide。
- 继续在这个文件里叠加 TileLang DSL，会把两套不同 DSL 和两套不同约束混在一起。

3. 真实的 VPTO legality contract

- 当前 verifier 与 `test/vpto_validate/` 明确要求 dedicated `pto.vecscope/pto.strict_vecscope`。
- `openspec/specs/vpto-ir-legality/spec.md` 仍残留 `llvm.loop.aivector_scope` 版本的旧 requirement。
- 如果不先修正 spec，后续 lowering 会天然踩在错误契约上。

### 实现约束

- 本特性相关源码、样例、测试、局部文档必须集中在 `tilelang-dsl/`。
- 根目录其他位置只允许做最小 build/install/test wiring，不得把核心实现重新塞回 `python/` 或 `test/` 现有目录树。
- v1 只支持 `a5`，并且只支持单个 monomorphic `dtypes` signature。
- bare `Tile` 参数不能依赖 Python runtime 值自动推导 physical shape；必须由显式 specialization 提供。
- diagnostics 需要在 frontend 分层给出，不得把“unsupported feature”伪装成底层 verifier failure。

## Goals / Non-Goals

**Goals:**

- 建立独立 `tilelang_dsl` package 和 `tilelang-dsl/` 目录边界。
- 固定 v1 `@pto.vkernel` 的 descriptor API 与参数定型规则。
- 固定 v1 bare `Tile` specialization 机制。
- 固定 v1 frontend diagnostics 的分层和失败行为。
- 修正 `vpto-ir-legality` 的 vecscope requirement，使后续 lowering change 可依附真实 contract。

**Non-Goals:**

- 不在本 change 中设计具体 lowering pass、builder 或 codegen pipeline。
- 不在本 change 中给 `constraints`、`priority`、`Any*`、`TypeVar` 定义运行语义。
- 不引入公开的 TileLang 中间 IR。
- 不要求现有 `python/pto/dialects/pto.py` 与新 package 共用内部实现。

## Decisions

### 1. 采用独立 package `tilelang_dsl`，而不是扩展现有 `python/pto`

决策：

- TileLang DSL v1 实现放在 `tilelang-dsl/python/tilelang_dsl/`
- 示例统一使用 `import tilelang_dsl as pto`
- `tilelang-dsl/tests/`、`tilelang-dsl/examples/`、`tilelang-dsl/docs/` 一并承载本特性工件

原因：

- 用户已经明确要求不考虑现有其他 Python binding 实现。
- 现有 `python/pto/dialects/pto.py` 同时承担 dialect binding 和实验 DSL，继续叠加会扩大耦合面。
- 独立 package 让 OpenSpec、测试和后续实现边界都更清晰。

备选方案：

- 直接扩展 `python/pto/dialects/pto.py`
  - 放弃原因：会把 TileLang DSL 和现有实验 VPTO DSL 混在同一入口，难以隔离行为和文档口径。

### 2. v1 decorator surface 固定为 `a5` 单 target + 单一 monomorphic `dtypes`

决策：

- v1 `@pto.vkernel` 仅接受 `target="a5"`
- `op` 为必填 metadata
- `dtypes` 必须是仅含一个 tuple 的 monomorphic signature
- `name`、`verify` 保留
- `constraints`、`priority`、多 signature `dtypes`、`Any*`、`TypeVar` 一律在 frontend reject，并由 diagnostics 明确指向 follow-up change

原因：

- 这是当前最小且可实现的契约，能支撑后续 v1 lowering，不会把 matcher 语义混入基础 change。
- 这让参数定型规则稳定：每个参数位置只存在一个最终类型绑定结果。

备选方案：

- 直接支持完整 matcher surface
  - 放弃原因：会把 kernel registry、constraint evaluation、tie-breaking、wildcard typing 一并引入，超出 v1 基础 change。

### 3. bare `TensorView` / `Tile` 注解继续沿用 guide 风格，元素类型通过 `dtypes` 绑定

决策：

- `TensorView` / `Tile` 参数在函数签名中使用 bare annotation
- 单个 `dtypes` signature 按参数位置绑定元素类型
- 标量参数仍使用显式标量注解，并在 `dtypes` 的同位置写出同类型

原因：

- 这与 `docs/tilelang-dsl-guide.md` 的核心书写方式一致。
- 对后续 matcher change 友好，不会先人为引入另一套“参数注解写满全部类型”的平行 surface。

备选方案：

- 要求每个参数都在注解中写完整类型/shape
  - 放弃原因：与 guide 差异过大，也会把 Tile specialization 和动态 TensorView profile 搅进签名层。

### 4. bare `Tile` 参数采用 descriptor-level specialization，而不是 Python runtime 推导

决策：

- bare `Tile` 参数的 physical shape / memory space / config 不在函数定义时写死
- `descriptor.specialize(**bindings)` 是唯一合法的补全入口
- 只有所有 bare `Tile` 参数都 specialization 完成后，才能调用 `mlir_text()` / `mlir_module()` / `verify()`

原因：

- Tile physical shape 必须静态，但内核定义时未必能知道具体实例。
- 显式 specialization 比隐式 runtime 推导更稳定，也更适合后续 matcher / registry 场景。

备选方案：

- 让 `Tile` 参数从 runtime Python object 自动推导
  - 放弃原因：会把编译期 contract 和运行期 object 混在一起，难以保证 deterministic IR surface。

### 5. diagnostics 在 frontend 分层 fail-fast，不把 unsupported feature 甩给后端 verifier

决策：

- decorator-level unsupported feature
- syntax-level unsupported Python construct
- type/profile-level illegal shape / missing specialization
- lowering前非法 vector-scope 前提

以上都必须在 TileLang frontend 直接报错，并附带源码位置。

原因：

- 这些错误是 TileLang surface 语义问题，不应该等到底层 VPTO verifier 再以“语义不合法 IR”形式暴露。
- fail-fast diagnostics 才能稳定区分“DSL 不支持”与“lowering 出 bug”。

### 6. 先在 OpenSpec 中修正 `vpto-ir-legality` 的 vecscope contract

决策：

- 本 change 直接带一个 `vpto-ir-legality` delta
- 明确 authoring-form 只接受 dedicated `pto.vecscope/pto.strict_vecscope`
- 继续拒绝 legacy `scf.for {llvm.loop.aivector_scope}`

原因：

- 这是后续 lowering change 的前置条件。
- 继续保留错误 spec 会让实现与契约长期漂移。

## Risks / Trade-offs

- [Risk] `tilelang_dsl` 与现有 `pto` 相关命名空间容易混淆  
  Mitigation：独立 package 名固定为 `tilelang_dsl`，示例中只通过 `import tilelang_dsl as pto` 复用书写风格。

- [Risk] v1 对 matcher feature 的 reject 可能被误解为“设计缺失”  
  Mitigation：在 proposal、diagnostics 和 follow-up change 中明确这些能力被延期到 `extend-tilelang-dsl-matcher-and-advanced-surface`。

- [Risk] 先修 spec 再做实现可能暴露与现有 verifier 更多不一致点  
  Mitigation：本 change 只修正已经被 `lib/PTO/Transforms/PTOValidateVPTOIR.cpp` 与 `test/vpto_validate/` 明确证明的 vecscope 冲突，不额外扩张。

- [Risk] bare `Tile` specialization API 若定义不清，会把 shape/profile 责任拖到后续 change  
  Mitigation：在本 change 中直接固定 `specialize()` 是唯一入口，并把缺失 specialization 归类为 frontend error。
