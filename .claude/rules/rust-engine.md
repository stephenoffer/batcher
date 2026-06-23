# Rule: The Rust Data Plane

The `bc-*` crates are the data plane. They are pure Rust over Apache Arrow, fast,
and ruthlessly tested against an oracle. Every rule here exists to keep the engine
correct as it grows multiple execution tiers and scales from one core to a cluster.

## Crate DAG — dependencies point one way only

```
bc-arrow → bc-expr → bc-ir → ┬→ bc-runtime ┐
                             └→ bc-codegen  ┴→ bc-interp → bc-py
bc-sketches   (depends only on bc-arrow / arrow)
bc-transport  (depends only on bc-arrow / arrow + flight stack)
```

- A crate MUST depend only on crates strictly below it. Never add an upward edge
  (e.g. `bc-runtime` importing `bc-interp`) or a sideways one. If you need a type
  in two places, it belongs in the lowest crate that both can see (often `bc-ir`
  or `bc-arrow`).
- **`bc-py` is the only crate that links PyO3.** Every other crate MUST build and
  `cargo test` with no Python interpreter (`just check` / `just test-rust` run
  `--workspace --exclude bc-py`). Never add `pyo3` to another crate.

## Arrow is the only columnar contract

- Every operator, the interpreter, the JIT, and the FFI boundary speak Arrow
  `RecordBatch` (`bc_arrow::Morsel`, default 16,384 rows). Do not invent row
  structs, bespoke buffers, or alternative columnar formats.
- The Python boundary is **zero-copy** via the Arrow C Data Interface
  (`arrow-pyarrow`). Don't serialize batches across FFI.
- `bc-arrow` is the single place the workspace pins its arrow version. Use
  `bc_arrow`'s re-exports rather than depending on `arrow` ad-hoc where practical.

## One `Expr`, one `RelOp` — shared across every tier

- There is exactly one scalar expression type (`bc_expr::Expr`) and one relational
  plan type (`bc_ir::RelOp`). Both the interpreter and the JIT consume the *same*
  `Expr`; that shared source is what guarantees semantic parity. Do not fork a
  second representation for a new backend.
- Their `serde` tags are the **wire contract** with Python. `RelOp` is deserialized
  from JSON whose `op` tags (`scan`, `filter`, `project`, `aggregate`, `sort`,
  `hash_join`, `distinct`, `window`, `limit`, `union`) and `snake_case` operators
  MUST match Python's `to_ir()` output exactly. Change one side → change the other
  in the same commit, and add a round-trip/differential test.

## Execution tiers — the interpreter is the oracle

- **Tier-0, `bc-interp::execute`** — the sequential reference. It is the
  correctness oracle. Keep it simple, deterministic, and obviously correct; every
  other path is checked against it.
- **Tier-0 parallel, `bc-interp::par`** — same operator semantics (the `ops`
  module is shared), different *scheduling* only (morselize + rayon + hash
  shuffle). The parallel path MUST compute exactly what the sequential path does.
- **Tier-1, `bc-codegen`** — Cranelift JIT for the supported `Expr` subset
  (numeric, no nulls, arith/compare). It MUST be **bit-for-bit identical** to
  `bc_expr::Expr::eval` on that subset, and MUST silently fall back (`compile_expr`
  → `None`, or a per-batch eval error → interpreter) on anything unsupported.
  Compile once per operator and reuse across morsels — a per-morsel compile loses
  to the interpreter.

When you extend `Expr`: handle it in the interpreter first (the oracle), then
either teach the JIT *and* prove parity, or leave the JIT to fall back. Never ship
a JIT path that disagrees with the interpreter.

## Mergeable algebra — single-node == distributed

Stateful operators (aggregation, join, distinct, window, top-N) MUST be built as
mergeable primitives in `bc-runtime`:

- `partial(batch) → state`, `combine(states) → state`, `finalize(state) → rows`.
- `combine` MUST be associative+commutative so partials merge in any order.
- This one implementation serves: one core (interp), many cores
  (`bc-interp::par` morselizes + merges), and many machines
  (`bc-interp::dist::{partial_aggregate, partition_batches, combine_finalize}`
  composed by the Python orchestrator over Ray). There is **no** separate
  distributed operator with its own semantics.
- The invariant `combine_finalize(partition(partial(pₖ)))` over all partitions ==
  single-node result is a test you MUST keep green when touching these.

`bc-interp` orchestrates; `bc-runtime` owns the state. Compiled/parallel pipelines
own no relational state — it lives in `bc-runtime` structures behind stable
signatures, so SIMD/NUMA/spillable rewrites can land without touching callers.

## Sketches and transport

- `bc-sketches` (HLL / KLL / Count-Min / ColumnStats) are all `Mergeable` with a
  fixed seed so partition-built sketches merge identically. Kyber consumes them for
  cardinality/quantile estimates. Keep them deterministic and mergeable.
- `bc-transport` is the Arrow Flight data-plane shuffle with **credit-based flow
  control** (Carbonite model: 1 credit = 1 batch slot; producer blocks at 0). The
  data plane bypasses the Ray object store entirely — do not route bulk batches
  through Ray.

## Code conventions

- **Errors**: `thiserror` enums per crate; compose with `#[from]` /
  `#[error(transparent)]`. Return `Result`; never `unwrap()`/`panic!` on a path
  that can see user data (panics are for genuine invariant violations only).
- **Unsafe**: avoid it; when unavoidable, honor `unsafe_op_in_unsafe_fn` and
  document the safety contract. FFI lives in `bc-py`.
- **Naming**: `snake_case` fns, `PascalCase` types, `SCREAMING_SNAKE` consts.
- **Docs**: module-level `//!` states the crate's single responsibility and
  contract; item `///` explains the *why*, not the *what*. Match the existing
  density — these crates are heavily, purposefully documented.
- **Lint clean**: `cargo clippy --workspace --exclude bc-py -- -D warnings` MUST
  pass. `cargo fmt --all` before done.

## Gate before "done"

`just check` → `just test-rust` → `cargo clippy ... -D warnings`. If you touched
the FFI surface or IR tags, also `just build` + `just test-py`. See `/run-quality-gate`.
