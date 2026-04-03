TileLang DSL v1 lives under this directory.

This subtree is the source of truth for the new frontend introduced by
`add-tilelang-dsl-core-foundation`.

Boundary with the existing `python/pto/dialects/pto.py` module:
- `tilelang-dsl/` owns new TileLang DSL v1 core implementation work
- `python/pto/dialects/pto.py` keeps PTO dialect bindings and the legacy
  experimental VPTO Python DSL surface
- Root-level wiring into build/install/test is allowed, but TileLang DSL core
  logic must not move back into `python/pto/dialects/pto.py`

Layout:
- `python/tilelang_dsl/`: package sources
- `tests/`: TileLang DSL focused tests
- `examples/`: self-contained examples
- `docs/`: local documentation for this frontend

Root-level wiring belongs to follow-up tasks and must stay minimal.
