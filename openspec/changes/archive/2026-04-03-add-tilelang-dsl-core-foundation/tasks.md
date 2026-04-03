## 1. OpenSpec 基础契约

- [x] 1.1 新增 `openspec/changes/add-tilelang-dsl-core-foundation/specs/tilelang-dsl-surface/spec.md`，固定 v1 package、decorator surface、参数定型和 Tile specialization 契约。
- [x] 1.2 新增 `openspec/changes/add-tilelang-dsl-core-foundation/specs/tilelang-dsl-diagnostics/spec.md`，固定 fail-fast diagnostics、错误定位和 unsupported feature 行为。
- [x] 1.3 新增 `openspec/changes/add-tilelang-dsl-core-foundation/specs/vpto-ir-legality/spec.md` delta，修正 vecscope carrier requirement 与真实 verifier 一致。

## 2. 目录与接线

- [x] 2.1 在 `tilelang-dsl/` 下创建 `python/`, `tests/`, `examples/`, `docs/` 基础布局，并保持本特性源码集中在该目录。
- [x] 2.2 增加最小根级 build/install/test wiring，让 `tilelang_dsl` package 能被本仓库构建和测试系统发现，但不把核心逻辑迁回 `python/` 现有目录。
- [x] 2.3 明确 `tilelang-dsl/` 与现有 `python/pto/dialects/pto.py` 的边界，避免在旧文件中继续堆叠 TileLang DSL 核心实现。

## 3. Surface 与 descriptor API

- [x] 3.1 实现 v1 `@pto.vkernel` descriptor skeleton，固定 `target/op/dtypes/name/verify` 字段和 `a5` 单 target 约束。
- [x] 3.2 实现 bare `TensorView` / `Tile` 参数的单一 monomorphic `dtypes` 绑定规则。
- [x] 3.3 实现 bare `Tile` 参数的 `specialize(**bindings)` 机制，并把 `mlir_text()`, `mlir_module()`, `verify()`, `emit(path)` 挂到 descriptor 上。

## 4. Frontend diagnostics

- [x] 4.1 为 `constraints`、`priority`、多 signature `dtypes`、`Any*`、`TypeVar` 提供 fail-fast diagnostics，并在消息中指向 follow-up change。
- [x] 4.2 为 unsupported Python syntax / arbitrary call / unsupported op surface 提供 source-located diagnostics。
- [x] 4.3 为缺失 Tile specialization、dynamic physical tile shape、非法 shape profile 提供前端错误，而不是把错误拖到 lowering 或 verifier。

## 5. 测试与文档

- [x] 5.1 在 `tilelang-dsl/tests/` 增加 package/import、descriptor API、specialization 和 diagnostics 的正反向测试。
- [x] 5.2 在 `tilelang-dsl/docs/` 写明 v1 surface 与延期 feature，明确现有 `python/pto/dialects/pto.py` 不是本特性的 source of truth。
- [x] 5.3 运行与记录最小验证命令，确认 `tilelang_dsl` package 可被构建/导入，diagnostics 能稳定定位到 DSL 源位置。
