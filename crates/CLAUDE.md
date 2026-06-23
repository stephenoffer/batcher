# Rust Data Plane — quick rules

You are editing the `bc-*` engine crates. Full detail:
`.claude/rules/rust-engine.md`. The non-negotiables:

- **Respect the DAG.** `bc-arrow → bc-expr → bc-ir → {bc-runtime, bc-codegen} →
  bc-interp → bc-py`. Dependencies point one way only — no upward/sideways edges.
- **Only `bc-py` links PyO3.** Every other crate must `cargo test` without Python.
  Don't add `pyo3` anywhere else.
- **Arrow `RecordBatch` is the only columnar contract.** No bespoke row formats.
- **One `Expr`, one `RelOp`, shared across tiers.** The interpreter
  (`bc-interp::execute`) is the correctness oracle; the parallel path and the
  Cranelift JIT MUST agree with it bit-for-bit (JIT falls back on its unsupported
  subset, never diverges).
- **Mergeable algebra for stateful ops.** Build `partial → combine → finalize` in
  `bc-runtime` so one implementation serves single-node, multi-core, and
  distributed. Keep the `combine_finalize(partition(partial(x))) == single-node`
  invariant test green.
- **IR tags are a wire contract** with Python `to_ir()`. Change one side → change
  both in the same commit + add a round-trip/differential test.
- **Errors via `thiserror`** (`#[from]` / `transparent`); no `unwrap`/`panic` on
  data paths.
- **Bounded files.** ≤800 code lines per `.rs` (the trailing `#[cfg(test)]` module is
  excluded). When a file outgrows it, split into a `foo/mod.rs` + submodules along
  responsibility seams — but the `Expr`/`RelOp` enums and their `serde` tags stay in
  the crate's `lib.rs` (the wire contract is one seam never cut across).
  (`.claude/rules/maintainability.md`; `just lint-structure`)

Before done: `just check` → `just test-rust` → `cargo clippy --workspace
--exclude bc-py -- -D warnings` → `cargo fmt --all`. If you touched the FFI
surface or IR tags, also `just build` + `just test-py`. Tests:
`.claude/rules/testing.md`. Full recipe: the `add-relational-operator` skill.
