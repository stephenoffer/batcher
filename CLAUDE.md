# Batcher v2 — Engineering Contract

Batcher is a native, JIT-compiling, **adaptive** data engine: a Python control
plane over a Rust data plane on Apache Arrow. The goal is to beat DuckDB, Spark,
Ray Data, and Polars across the whole range — **sub-second small queries to
PB-scale**, **batch and streaming**, **single-node and distributed** — for
SQL-style, DataFrame, and ML/multimodal workloads. The moat is an adaptive
control layer that re-optimizes *during* a query (which DuckDB's static optimizer
and Spark AQE's stage-boundary adaptation cannot match).

This file is the always-loaded contract. It is law. The `@import`ed rule files
carry the detail; read the one for the layer you are touching before you edit.

## The non-negotiable invariants (hard gates)

These are MUSTs. A change that breaks one is wrong even if it compiles and the
tests you wrote pass.

1. **Three independent subsystems.** `kyber` (optimizer), `carbonite` (resources),
   and `core` (executor) MUST NOT import one another. Only `api` (the conductor)
   imports all three; `plan` is the neutral contract layer and imports none of
   them. Enforced by `just lint-layers`. → `.claude/rules/architecture.md`
2. **Control plane vs data plane.** Python builds/optimizes a plan and ships it as
   JSON IR; it MUST NOT touch a tuple/row in the hot path. Per-row and per-batch
   work lives in Rust. → `.claude/rules/architecture.md`
3. **Arrow is the only columnar contract.** Every operator, the interpreter, the
   JIT, and the FFI boundary speak Arrow `RecordBatch`. No bespoke row formats.
   The boundary is zero-copy (Arrow C Data Interface). → `.claude/rules/rust-engine.md`
4. **Only `bc-py` links PyO3.** Every other crate is pure Rust, `cargo test`-able
   without a Python interpreter. → `.claude/rules/rust-engine.md`
5. **The crate DAG points one way.** Dependencies flow `bc-arrow → bc-expr →
   bc-ir → {bc-runtime, bc-codegen} → bc-interp → bc-py`. Never add an upward or
   sideways edge. → `.claude/rules/rust-engine.md`
6. **One `Expr`, one `RelOp`, shared across tiers.** The Tier-0 interpreter is the
   correctness oracle; the Tier-1 Cranelift JIT MUST be bit-for-bit identical on
   its supported subset, and silently fall back otherwise. → `.claude/rules/rust-engine.md`
7. **Single-node == distributed via mergeable algebra.** Stateful operators are
   built as `partial → combine → finalize`; one implementation serves one core,
   many cores, and many machines. No separate distributed code path with its own
   semantics. → `.claude/rules/rust-engine.md`, `.claude/rules/performance.md`
8. **The JSON IR is a stable wire contract.** Python `to_ir()` tags and Rust
   `serde` tags MUST stay in lockstep. → `.claude/rules/python-control-plane.md`
9. **Everything is tested, correctness before speed.** Every relational/operator
   change MUST add a **differential test vs DuckDB**; new Rust primitives MUST
   have a unit test and preserve the seq == par == JIT oracle. No timing claim
   without a passing correctness check. → `.claude/rules/testing.md`
10. **No performance regressions.** Perf-relevant changes are benchmarked against
    DuckDB/Polars via `benchmarks/`, and reasoned about vs Spark/Ray Data.
    → `.claude/rules/performance.md`
11. **Python stays clean.** `ruff check` + `ruff format` clean, fully typed, no
    duplication, no dead code, and a small curated public API with docstrings.
    → `.claude/rules/python-quality.md`
12. **Structure stays bounded.** File/dir/class size limits (Python ≤500 lines, Rust
    ≤800 excl. tests, ≤12 files/dir, ≤5 levels deep, `__init__` ≤120 re-export-only);
    "many small things" grow as grouped-by-family modules + a registry, never a god
    file or one-file-per-rule; no mixin god-objects — fluent builders + namespace
    accessors instead. This is how v2 avoids v1's collapse. Enforced by
    `just lint-structure` + the pre-commit hook. → `.claude/rules/maintainability.md`

## Repository map

```
python/batcher/          Control plane — never touches a tuple in the hot path
  api/        conductor: the only layer that imports kyber+carbonite+core
  kyber/      optimizer: an ordered list of passes (plan → plan)
  carbonite/  resource manager: buffer pool, spill, credit-based flow control
  core/       executor: drives the engine, adaptive re-optimization loop
  plan/       NEUTRAL: LogicalPlan/PhysicalPlan, expr_ir, schema, JSON IR (to_ir)
  metadata/   learned stats (MetadataHub) — Core measures, Kyber consumes
  config/  io/  dist/  _sql/  _internal/

crates/                  Data plane — pure Rust + Arrow (only bc-py links PyO3)
  bc-arrow      Arrow re-exports + Morsel (RecordBatch, 16,384 rows)
  bc-expr       the one scalar Expr + vectorized eval (interpreter oracle)
  bc-ir         the one relational RelOp DAG (JSON wire contract)
  bc-runtime    mergeable stateful primitives: agg, join, shuffle, window
  bc-codegen    Cranelift JIT for Expr (Tier-1; bit-for-bit parity w/ bc-expr)
  bc-interp     Tier-0 executor: execute (seq oracle), par, dist
  bc-sketches   mergeable HLL / KLL / Count-Min for cardinality/quantiles
  bc-transport  Arrow Flight shuffle (data plane bypasses the Ray object store)
  bc-py         the ONLY PyO3 crate; thin, zero-copy FFI boundary

architecture.txt, docs/  Design + the mathematical foundations (source of truth)
benchmarks/              Correctness-gated benchmarks vs DuckDB / Polars
tests/{unit,differential,integration}/   The test pyramid
```

## Dev workflow

Use the `just` recipes — they encode the exact build/test invocations:

```
just build        # maturin develop — build the Rust engine into the venv
just build-release
just check        # cargo check --workspace --exclude bc-py  (fast, no PyO3 link)
just test-rust    # cargo test  --workspace --exclude bc-py
just test-py      # pytest  (requires `just build` first)
just test         # CI: check → test-rust → build → test-py
just fmt          # cargo fmt + clippy -D warnings
just lint-py      # ruff check + ruff format --check  (Python quality gate)
just lint-layers  # import-linter — enforces the three-subsystem independence
just docs         # build the docs site (warnings = errors: orphans, broken refs)
just bench        # operator-mix benchmark vs DuckDB/Polars (also bench-tpch, bench-dist)
```

**Nothing is "done" until the quality gate is green.** Before you claim a change
works: `just check`, `just test-rust`, `just build`, `just test-py`,
`just lint-layers`, and `clippy -D warnings`. Doc changes also run `just docs`
(the doc code examples execute under `just test-py` via
`tests/docs/test_doc_examples.py`). For perf-relevant work, also run `just bench`
(and `just bench-tpch` / `just bench-dist` where relevant). The
`/run-quality-gate` skill walks this and how to triage failures.

## Skills (invoke when the task matches)

- **`add-relational-operator`** — extend the engine with a new operator across
  every layer (IR → interp → runtime → par/dist → Python → tests → benchmark).
- **`add-distributed-operator`** — wire an operator through the distributed path
  (partial/combine/finalize, shuffle, Arrow Flight, Carbonite credits).
- **`add-kyber-optimizer-pass`** — append an optimization pass to the Kyber
  pipeline, with sketch-based cardinality and the tests that prove it preserves
  results.
- **`run-quality-gate`** — the full verification sequence and failure triage.

## Source of truth

`architecture.txt` and `docs/internals/{kyber,carbonite,mathematical_foundations}.md`
are the authoritative design + math (contracts, control theory, sketch error
bounds, regret/stability proofs). When a design question has a real answer there,
read it — do not re-derive or guess. This contract summarizes; those documents
decide.

@.claude/rules/architecture.md
@.claude/rules/rust-engine.md
@.claude/rules/python-control-plane.md
@.claude/rules/python-quality.md
@.claude/rules/maintainability.md
@.claude/rules/testing.md
@.claude/rules/performance.md
