# Testing strategy

Batcher's claim is to be faster than DuckDB, Spark, Polars, and Ray Data *and*
correct. That is only credible if correctness is proven mechanically against a
reference on every change. So the rule is blunt: correctness before speed, and a
fast wrong answer is a bug. The benchmark harness enforces it literally — it refuses
to time a query whose result does not match the oracle.

## Two oracles

Tests do not hand-roll expected values. They check against one of two references.

**DuckDB, for relational behavior.** Any operator, expression, SQL form, or
optimizer rewrite must produce the same result DuckDB does on the same input. The
harness is `tests/differential/conftest.py::assert_same` — an order-independent,
type-tolerant comparison (it accepts int-versus-float, Decimal-versus-float, and
float rounding). Cases live in the `test_diff_*.py` files next to it. If Batcher and
DuckDB legitimately differ, that is a decision to surface and document, never to
hide by weakening a test.

**The Tier-0 interpreter, for the Rust engine.** `bc-interp::execute` (sequential)
is the reference. The parallel executor and the Cranelift JIT must agree with it
bit-for-bit on every supported input. A new `bc-runtime` primitive also gets the
mergeability test: `combine_finalize(partition(partial(pₖ)))` over all partitions
must equal the single-node result, which is what guarantees one core, many cores,
and many machines compute the same thing.

## Test layout

```
tests/
├── unit/           fast, no native engine — optimizer passes, IR validation, cost
├── differential/   cross-check results against DuckDB/Polars (the correctness spine)
├── integration/    end-to-end — I/O, adaptive re-optimization, distributed, spilling
├── io/             source and sink formats
├── property/       Hypothesis invariants — merge associativity, IR round-trips, idempotence
└── docs/           executes the code examples in the docs, and the examples/ scripts
```

The `docs/` directory runs two harnesses: `test_doc_examples.py` executes the fenced
`python` blocks embedded in this documentation, and `test_examples.py` runs every
standalone script under the top-level `examples/` directory. Both fail the suite if a
demonstrated API is removed or renamed, so usage coverage cannot rot.

Markers are declared in `pyproject.toml`: `unit`, `differential`, `integration`,
and `property`. Property tests (Hypothesis) are encouraged for algebraic invariants
— merge associativity, encode/decode round-trips, optimizer idempotence — where one
law covers a space no enumerated case can.

## What each change must prove

The gate scales with what you touched.

- A new or changed operator or expression adds a differential test against DuckDB
  covering nulls, empties, and type edges, and keeps the Rust sequential, parallel,
  and JIT paths in agreement. Touching the JSON IR adds a round-trip test that the
  Python `to_ir()` shape deserializes in Rust.
- A new `bc-runtime` primitive gets a unit test and the mergeability invariant; if it
  is stateful, it is tested spilled and partitioned too.
- A new Kyber pass gets a unit test proving the rewrite is semantics-preserving (the
  plan changes, the result does not) plus a differential test that the optimized
  query still matches DuckDB.
- A distributed change gets an equivalence test: single-node output equals
  multi-partition output.
- A bug fix lands with a regression test that fails before the fix.

## Running the tests

```bash
just test          # the CI sequence: check → test-rust → build → test-py
just test-rust     # cargo test (the Rust oracle, parallel and JIT parity)
just test-py       # pytest, including the differential suite and doc examples
```

`just test-py` requires a built engine (`just build` first), because the
differential and integration suites run real queries. The documentation's code
examples are executed under `tests/docs/test_doc_examples.py`, so a doc snippet that
stops working fails the build rather than rotting silently.

## Coverage philosophy

Cover the contract, not the implementation: every operator against empty input,
nulls, a single row, multiple batches, and type boundaries. A wide enumerated suite
of these edges catches more than chasing a coverage percentage, because the edges are
where engines actually disagree.

## Coverage measurement

Coverage is measured on both planes and gated as a ratchet — the floor sits just
below the achieved baseline so it blocks regressions, and is raised as coverage grows.
It is a backstop against untested code creeping in, not a target to game; the edge
suite above is what actually proves correctness.

```bash
just cov-py        # Python control plane (pytest-cov, branch coverage)
just cov-rust      # Rust data plane (cargo-llvm-cov; one-time: cargo install cargo-llvm-cov)
just cov-gate      # the CI gate — runs the suite under coverage, fails below the floor
```

`just test` runs the full correctness suite (`test-py`) and then `cov-gate`. The
settings live in `[tool.coverage.*]` in `pyproject.toml` (the compiled `_native`
extension is omitted — it is exercised through the data plane, not the Python suite, so
counting it would mislead).

The gate measures a **deterministic subset** — `tests/{unit,differential,property,io,docs}`
— and deliberately excludes `tests/integration`. Those Ray / adaptive-learning /
distributed tests are stable on their own (and run for correctness under `test-py`),
but coverage instrumentation perturbs their timing enough to make them flaky, which
would make an enforced gate non-deterministic. The current floor is **62%** branch
coverage of `python/batcher`, set just below the measured subset baseline of ~64%.
Raise the `--cov-fail-under` value in the `cov-gate` recipe whenever a round of new
tests lifts the baseline. (`just cov-py` reports the same subset with line-by-line
misses and an HTML drill-down.)

## See also

- [Execution engine](execution.md) — the sequential, parallel, and JIT paths under test
- [Kyber optimizer](kyber.md) — the passes the differential tests guard
