---
name: add-relational-operator
description: End-to-end recipe to add or extend a relational operator (or scalar expression) in Batcher across every layer — Rust IR, interpreter, runtime, parallel + distributed paths, the Python plan/api surface, the JSON IR contract, and the required differential + benchmark coverage. Invoke when adding a new operator/aggregate/expression or extending an existing one.
---

# Add a relational operator (across all layers)

Batcher operators span the Rust data plane and the Python control plane and must
behave identically single-node, multi-core, and distributed. Do the layers in this
order — each step is checked against the one before it. Use an existing operator of
the same shape as your template: **`Filter`/`Project`** (stateless, per-batch),
**`Aggregate`/`Distinct`** (stateful, mergeable), **`HashJoin`** (two inputs),
**`Sort`/`Window`** (breaker).

Read `.claude/rules/rust-engine.md`, `.claude/rules/python-control-plane.md`, and
`.claude/rules/testing.md` first.

## 1. Define the IR (the shared contract)

- **Scalar expression?** Add a variant to `bc_expr::Expr` in
  `crates/bc-expr/src/lib.rs` (and the matching `StrFunc`/`MathFunc`/etc. enum).
  Pick the `serde` tag / `rename_all = "snake_case"` name deliberately — it is the
  wire contract.
- **Relational operator?** Add a variant to `bc_ir::RelOp` in
  `crates/bc-ir/src/lib.rs` with boxed child(ren) and an `#[serde]` `op` tag (e.g.
  `"hash_join"`). Reuse existing pieces (`ProjectionItem`, `AggregateItem`,
  `AggFunc`, `SortKey`, `JoinType`) rather than inventing parallel structures.

This tag is now law for both sides. Don't proceed until the name is final.

## 2. Implement in the interpreter — the correctness oracle

- Add the operator logic to `crates/bc-interp/src/ops.rs` (the shared operator
  module both executors call) and wire it into the sequential walk in
  `crates/bc-interp/src/lib.rs` (`execute`).
- For an expression, implement `Expr::eval` in `bc-expr` using arrow compute
  kernels. Make it obviously correct; this is the oracle everything else is graded
  against. Handle nulls, empty input, and type edges here.

## 3. If stateful, build the mergeable primitive in `bc-runtime`

Stateful operators (aggregation, join, distinct, window, top-N) MUST be mergeable:

- Implement `partial(batch) → state`, `combine(states) → state`,
  `finalize(state) → cols` in the relevant `crates/bc-runtime/src/{agg,join,
  shuffle,window}.rs` module. `combine` must be associative + commutative.
- Add a Rust unit test asserting the invariant:
  `finalize(combine(partials))` over any partition split == the single-node result.
- Keep state in `bc-runtime` structures behind stable signatures so the interp,
  parallel, and dist paths all reuse them (and future spill/SIMD rewrites don't
  touch callers).

## 4. Parallel path (`bc-interp::par`)

- Wire the operator into `crates/bc-interp/src/par.rs`. Stateless ops map over
  morsels; stateful ops do partial-per-morsel → combine → finalize; joins
  hash-partition both sides into buckets and run per bucket.
- Reuse the same `ops`/`bc-runtime` code as the sequential path — the two
  executors differ only in *scheduling*, never in semantics. A Rust test must show
  `par::execute_parallel` == `execute` on the new operator.
- If the expression is JIT-eligible (numeric, no nulls), it flows through
  `ops::try_compile` automatically — verify the JIT result matches the interpreter,
  or that it correctly falls back.

## 5. Distributed path (`bc-interp::dist`)

- For mergeable ops, expose/compose the map-reduce primitives in
  `crates/bc-interp/src/dist.rs`: `partial_aggregate` (map), `partition_batches`
  (hash shuffle), `combine_finalize` (reduce). These are exactly the `bc-runtime`
  pieces surfaced at orchestrator granularity — don't write new semantics.
- See the dedicated `add-distributed-operator` skill for the Flight transport +
  Carbonite credit wiring and the single-node==distributed equivalence test.

## 6. Surface it in Python (`plan` + `api`)

- Add the `LogicalPlan` node in `python/batcher/plan/logical.py` with a `to_ir()`
  returning `{"op": "<tag>", ...}` whose tag and field names **exactly match** the
  Rust `serde` definition from step 1. (For expressions, extend
  `python/batcher/plan/expr_ir.py`.)
- Add the fluent, lazy, immutable method to `python/batcher/api/dataset.py`
  (returns a new `Dataset`; expression-first; one obvious way to do it). Curate
  `__all__`, type hints, a docstring, and typed errors (`PlanError`) for bad input.
- Keep lowering in the neutral `plan` layer; don't leak it into kyber/carbonite/core.

## 7. Optimizer (only if relevant)

If the operator enables a rewrite (pushdown, fusion, pruning), that's a separate
Kyber pass — use the `add-kyber-optimizer-pass` skill. Don't bake optimization into
the operator itself.

## 8. Test — the hard gate

- **Differential vs DuckDB**: add cases to `tests/differential/` (next to
  `test_diff_*.py`) using `assert_same`. Cover nulls, empty input, single row,
  multiple batches, type boundaries.
- **Rust**: unit test in the relevant crate; seq == par (== JIT) agreement; the
  mergeability invariant for stateful ops.
- **Plan shape** (if it changes optimization): assertion in `tests/unit/`.
- A query exercising the operator must round-trip through the JSON IR and execute.

## 9. Benchmark

If it's on a hot path, add/extend a query shape in `benchmarks/harness.py` and run
`python benchmarks/run.py`. Correctness vs DuckDB/Polars must pass before any
timing; no regression on existing ratios. See `.claude/rules/performance.md`.

## Done checklist

- [ ] One `Expr`/`RelOp` variant, serde tag matches Python `to_ir()`
- [ ] Interpreter (oracle) implemented; nulls/empty/edges handled
- [ ] Stateful → `bc-runtime` mergeable primitive + invariant test
- [ ] `par` path == sequential; JIT matches or falls back
- [ ] `dist` path composes the mergeable primitives (if distributable)
- [ ] Python `plan` node + `api` method; `__all__`, types, docstring, typed errors
- [ ] Differential test vs DuckDB; Rust unit tests; plan-shape test if applicable
- [ ] Benchmark run, no regression (if hot path)
- [ ] `/run-quality-gate` green
