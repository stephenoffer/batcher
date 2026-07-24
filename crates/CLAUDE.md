# Rust Data Plane

You are editing the `bc-*` engine crates: pure Rust over Apache Arrow, ruthlessly tested
against an oracle. This file carries what you need for almost any Rust change; the deep
reference is `.claude/rules/rust-engine.md` (+ `maintainability.md`, `testing.md`,
`performance.md`).

## The crate DAG — one direction only

```
                    ┌→ bc-ir → bc-runtime ┐
bc-arrow → bc-expr →┤                     ├→ bc-interp → bc-py
                    └→ bc-codegen ────────┘
leaves (no bc-* deps), consumed higher up:
  bc-sketches → bc-runtime, bc-py      bc-resource → bc-interp, bc-py
  bc-transport → bc-py                 bc-io       → bc-py
  bc-secrets  → bc-expr (secret-reference resolution; NO cloud SDK by design)
  bc-udf → (nothing depends on it — not on a live path)
```

Depend only downward — never an upward or sideways edge. A type needed in two places belongs
in the lowest crate both can see. Two things this corrects, which you should not "fix" back:
**`bc-codegen` does not depend on `bc-ir`** (it compiles scalar `Expr`, so it sits beside it),
and **`bc-py` is not merely downstream of `bc-interp`** — it depends directly on four leaves,
making it a second assembly point. `MAP.md` prints each crate's live deps from its manifest.

**Only `bc-py` links PyO3.** Every other crate must `cargo test` with no Python interpreter
(`just check` / `just test-rust` run `--workspace --exclude bc-py`). Never add `pyo3` elsewhere.

## Arrow is the only columnar contract

Every operator, both tiers, and the FFI boundary speak Arrow `RecordBatch`
(`bc_arrow::Morsel`, 16,384 rows). No bespoke row structs or alternative columnar formats.
The Python boundary is **zero-copy** via the Arrow C Data Interface — never serialize batches
across FFI. `bc-arrow` is the single place the workspace pins its arrow version.

## One `Expr`, one `RelOp` — and the seam that is never cut

There is exactly one scalar expression type (`bc_expr::Expr`) and one relational plan type
(`bc_ir::RelOp`). Both tiers consume the *same* `Expr`; that shared source is what guarantees
parity. Never fork a second representation for a new backend.

**The enums and their `serde` tags stay in their crate's `lib.rs`** — that is the wire
contract, and it lives in exactly one place. This is why `bc-expr/src/lib.rs` is allowlisted
over the size limit rather than split: the size limits are **subordinate** to the invariants.
Evaluation bodies are already extracted to `eval/`; extract those, not the enum.

Those tags are the wire contract with Python's `to_ir()`. **Change one side → change the
other in the same commit**, and add a round-trip/differential test. A Rust-only change to a
`serde` tag is a silent correctness bug, because nothing in Rust will fail.

## Execution tiers — the interpreter is the oracle

- **Tier-0 `bc_interp::execute`** — the sequential reference and correctness oracle. Keep it
  simple, deterministic, obviously correct. Read *this* to learn what an operator means, not
  `par.rs` (2,600 lines of scheduling that computes the same thing).
- **Tier-0 parallel `bc_interp::par`** — same operator semantics (`ops` is shared), different
  *scheduling* only. MUST compute exactly what the sequential path does.
- **Tier-1 `bc-codegen`** — Cranelift JIT for the supported `Expr` subset. MUST be
  **bit-for-bit identical** to `bc_expr::Expr::eval` there, and **silently fall back**
  (`compile_expr → None`, or per-batch eval error → interpreter) on anything else. A JIT path
  that disagrees produces wrong answers *fast*, with no error. Compile once per operator and
  reuse across morsels — a per-morsel compile loses to the interpreter.

Extending `Expr`: handle it in the interpreter first, then either teach the JIT **and prove
parity**, or leave it to fall back. Never ship a JIT path that diverges.

## Mergeable algebra — single-node == distributed

Stateful operators in `bc-runtime` MUST be `partial(batch) → state`, `combine(states) → state`
(associative **and** commutative), `finalize(state) → rows`. This one implementation serves
one core, many cores (`par` morselizes + merges), and many machines
(`dist::{partial_aggregate, partition_batches, combine_finalize}`). There is **no** separate
distributed operator with its own semantics.

Keep green: `combine_finalize(partition(partial(pₖ)))` over all partitions == single-node.
An operator without a mergeable form works perfectly single-node, passes every local test,
and silently caps at one machine — surfacing at cluster scale as wrong results, not an error.

`bc-interp` orchestrates; `bc-runtime` owns the state, so compiled pipelines own no relational
state and a SIMD/NUMA/spill rewrite lands without touching callers.

## Where things actually are

- **Sorting is in `bc-interp::ops`**, not `bc-runtime` (radix, sample-sort, stable string,
  external merge). A sort carries no state *between* morsels the way agg/join build does.
- **Window** straddles: kernels in `bc-runtime/window*.rs`, out-of-core in
  `bc-interp::window_spill`.
- **Filter/project/limit** have no file of their own — they are arms in `bc-interp/ops/mod.rs`.
- **`keys.rs`** is the ONE canonical grouping/partitioning key form; every hash path (assign /
  combine / shuffle / join / window) derives key identity from it so they cannot disagree.
  `bc-arrow/float_ident.rs` is the matching NaN/±0 contract.
- **`bc-transport`** is the Flight shuffle with credit-based flow control (1 credit = 1 batch
  slot; producer blocks at 0). The data plane bypasses the Ray object store — never route
  bulk batches through Ray.

## Conventions

`thiserror` enums per crate, composed with `#[from]`/`#[error(transparent)]`; return `Result`
— never `unwrap()`/`panic!` on a path that can see user data. Avoid `unsafe`; when
unavoidable honor `unsafe_op_in_unsafe_fn` and document the safety contract (FFI lives in
`bc-py`). Module-level `//!` states the file's single responsibility; item `///` explains the
*why*. Match the existing density — these crates are heavily, purposefully documented.

**Size:** ≤800 code lines per `.rs`, **excluding** the trailing `#[cfg(test)]` module (Rust
co-locates tests; counting them would punish good test density). `foo.rs` → `foo/mod.rs` +
submodules when it outgrows that — unless splitting would cut a seam above, in which case
allowlist it with a reason.

## Before done

`just check` → `just test-rust` → `cargo clippy --workspace --exclude bc-py --all-targets -- -D warnings`
→ `cargo fmt --all`. Touched the FFI surface or IR tags? Also `just build` + `just test-py`,
and prove the boundary is unchanged with `just surface-diff`. New `bc-runtime` primitive?
Add a `#[cfg(test)]` unit test **and** the mergeability invariant test; if stateful, test it
spilled and partitioned too.
