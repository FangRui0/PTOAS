# vpto-ir-legality Specification

## MODIFIED Requirements

### Requirement: VPTO vector and predicate structure MUST stay inside a single dedicated vecscope carrier

legacy `scf.for {llvm.loop.aivector_scope}` authoring form MUST NOT be accepted any longer.  
在 authoring-form VPTO 中，所有消费或产生 `!pto.vreg`、`!pto.mask<...>`、`!pto.align` 的 VPTO op MUST 位于 dedicated `pto.vecscope` 或 `pto.strict_vecscope` 作用域内。  
同时，dedicated `pto.vecscope/pto.strict_vecscope` carrier MUST NOT 相互嵌套。

#### Scenario: vector or predicate VPTO op outside dedicated vecscope is rejected

- **WHEN** authoring-form VPTO IR 中出现消费或产生 `!pto.vreg`、`!pto.mask<...>`、`!pto.align` 的 VPTO op，且该 op 不在任何 `pto.vecscope` 或 `pto.strict_vecscope` 内
- **THEN** authoring-stage verifier MUST 拒绝该 IR
- **AND** 诊断 MUST 明确指出违规 op 缺少 enclosing dedicated vecscope

#### Scenario: legacy `scf.for {llvm.loop.aivector_scope}` carrier is rejected

- **WHEN** authoring-form VPTO IR 试图继续使用带 `llvm.loop.aivector_scope` attr 的 `scf.for` 作为 vector carrier
- **THEN** authoring-stage verifier MUST 拒绝该 IR
- **AND** 诊断 MUST 明确指出该 form 已是 legacy authoring surface
- **AND** 诊断 MUST 要求改用 dedicated `pto.vecscope/pto.strict_vecscope`

#### Scenario: nested dedicated vecscope carriers are rejected

- **WHEN** 某个 `pto.vecscope` 或 `pto.strict_vecscope` 作用域内再次出现 dedicated `pto.vecscope` 或 `pto.strict_vecscope`
- **THEN** authoring-stage verifier MUST 拒绝该 IR
- **AND** 诊断 MUST 明确指出存在 nested dedicated vecscope

#### Scenario: shared scalar and control-flow surface is still allowed outside dedicated vecscope

- **WHEN** `arith`、`scf`、pointer-building、copy programming 或 sync programming 相关 op 本身不产生也不消费 `!pto.vreg`、`!pto.mask<...>`、`!pto.align`
- **THEN** authoring-stage verifier MUST NOT 仅因为这些 op 位于 dedicated vecscope 外就拒绝 IR
- **AND** vecscope 约束 MUST 只针对 VPTO vector / predicate / align surface 生效
