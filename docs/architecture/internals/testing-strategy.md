# Testing strategy

Batcher's claim is to be faster than DuckDB, Spark, and Polars *and*
correct. That is only credible if correctness is proven mechanically against a
reference on every change. So the rule is blunt: correctness before speed, and a
fast wrong answer is a bug. The benchmark harness enforces that literally: it refuses
to time a query whose result does not match the oracle.

## Two oracles

Tests do not hand-roll expected values. They check against one of two references.

**DuckDB, for relational behavior.** Any operator, expression, SQL form, or
optimizer rewrite must produce the same result DuckDB does on the same input. The
harness is `tests/differential/conftest.py::assert_same`, an order-independent,
type-tolerant comparison that accepts int-versus-float, Decimal-versus-float, and
float rounding. Cases live in the `test_diff_*.py` files next to it. If Batcher and
DuckDB legitimately differ, that is a decision to surface and document, never to
hide by weakening a test.

**The Tier-0 interpreter, for the Rust engine.** `bc-interp::execute` (sequential)
is the reference. The parallel executor and the Cranelift JIT must agree with it
bit-for-bit on every supported input. A new `bc-runtime` primitive also gets the
mergeability test: `combine_finalize(partition(partial(pₖ)))` over all partitions
must equal the single-node result, which is what guarantees one core, many cores,
and many machines compute the same thing.

## Property-based behavior testing

The two oracles say *what* the right answer is; property-based tests decide *where to look
for a wrong one*. They are a first-class layer alongside the oracles, not a substitute:
instead of one enumerated input, Hypothesis generates thousands of random tables and
pipelines and asserts an invariant holds on every draw, shrinking any failure to a minimal
counterexample. This is the layer that guards work whose correctness lives in a
*combination*: the full optimizer rule set, the metadata shortcuts, and the adaptive and
self-tuning paths. In all three, the bug an example misses is exactly the one that matters.
The suite lives in `tests/property/`, and each file pins one invariant and drives it through
an oracle.

**Optimizer result-invariance** (`test_prop_optimizer_result_invariance.py`). Kyber's full
154-rule set must change the *plan* and never the *answer*. Hypothesis builds a random typed
table and a random valid pipeline (filters carrying redundant and absorbing boolean shapes,
derived columns, group-by aggregates, distinct, sort, limit, union) and asserts

```text
result(FULL 154 rules)  ==  result(NO rules)  ==  ds.collect()
```

on an order-independent multiset compare, and on row order too when the pipeline is totally
ordered. Any rule, or any rule *interaction*, that alters a result falls out as a
counterexample. That is a real correctness bug to minimize and fix, never to weaken away.

**Confluence, termination, and determinism** (same file). Result-invariance says the rules
are individually sound; confluence says they behave in *combination*. The test asserts the
optimized IR at the production `fixpoint_iterations` cap equals the IR at a far larger cap.
That proves the combined set reaches its plan fixpoint within the budget the engine actually
runs, so no plan is silently *under*-optimized by being truncated mid-convergence as more
rules are added. It also asserts that re-running is byte-identical, so the rule system is
confluent with a unique normal form. Re-optimizing an already-optimized plan is a fixpoint
and re-executes to the identical rows. This is the guarantee that a growing,
grouped-by-family rule set does not regress into the interference and oscillation a large
uncoordinated pass list would.

**Metadata fidelity and the EXACT firewall** (`test_prop_metadata_fidelity.py`). The
metadata shortcuts answer `count` / `is_empty` / `min` / `max` / `n_unique` / `n_null` from
Parquet footer statistics without scanning a row. Two things must hold on random typed data
written to Parquet and read back. The first is **fidelity**: whenever a shortcut fires, its
answer equals both the engine-executed answer and DuckDB, because a wrong footer-derived
answer is a silent bug that never scans a row to get caught. The second is the **EXACT
firewall**: past a filter the source's bounds are no longer exact for the result, so a
shortcut that cannot *prove* its answer must decline and fall back to execution rather than
return a stale pre-filter value.
The test asserts both the decline on a genuinely partial filter and that the executed
fallback still matches DuckDB.

**Adaptive equals non-adaptive** (`test_prop_adaptive_equivalence.py`). Intra-query
re-optimization re-plans each pipeline breaker on its *measured* cardinality, so a join's
build side or broadcast choice can flip mid-query. The moat is that it plans *better*, never
*differently*. On a random selective-filter-into-join, the shape where the measured count
most diverges from the estimate, `adaptive=True`, `False`, and `"auto"` must produce the
same rows, and for the inner join the shared result is cross-checked against DuckDB.

**Adaptive-tuning result-invariance** (`test_prop_tuning_invariance.py`). Every self-tuning
lever is contractually result-invariant: morsel size, spill, shuffle partition count,
adaptive morsel sizing, and the learned strategy choices behind them all change *how* a
query runs, never *what* it returns. Hypothesis runs the same aggregate or distinct at
opposite settings of each knob (a 1-row morsel against 64k, spill forced against in-memory,
one partition against seven) and asserts byte-identical results. This is the contract every
adaptive-tuning optimization must hold, and it is checked rather than trusted.

**The mergeable algebra** (`test_prop_mergeable_invariant.py`,
`test_prop_partition_invariant.py`). Stateful operators are `partial → combine → finalize`
with an associative-commutative `combine`; that is the single invariant that lets one
implementation serve one core, many cores, and many machines. The tests assert an
aggregate / distinct / sort-limit over one morsel equals the same over any random chunking
of the input, and equals DuckDB. Because the native distributed primitives are directly
callable, `test_prop_mergeable_invariant.py` also drives
`combine_finalize(partition(partial(pₖ)))` over the raw Rust kernels and asserts it equals
the single-node result. Partition-independence is therefore proven at the primitive, not
only through the Python path.

**Why this layer earns its place.** These invariants caught real bugs during development that
no example test would have surfaced: a transient crash under a concurrent plan edit, and an
adaptive-vs-non-adaptive divergence that traced to a pre-existing empty-input engine
limitation, where the adaptive path short-circuits an empty subtree while the one-shot path
hits the gap. Both were surfaced honestly and isolated rather than papered over. A property
that fails is a decision to surface, exactly like a differential mismatch, and weakening it
to go green is not an option.

## Test layout

Each directory holds one kind of test, and the kind decides what it is allowed to depend
on:

```text
tests/
├── unit/           fast, no native engine: optimizer passes, IR validation, cost
├── differential/   cross-check results against DuckDB/Polars (the correctness spine)
├── integration/    end to end: I/O, adaptive re-optimization, distributed, spilling
├── io/             source and sink formats
├── property/       Hypothesis invariants: optimizer, metadata, adaptive, mergeable
└── docs/           executes the code examples in the docs, and the examples/ scripts
```

The `docs/` directory runs two harnesses: `test_doc_examples.py` executes the fenced
`python` blocks embedded in this documentation, and `test_examples.py` runs every
standalone script under the top-level `examples/` directory. Both fail the suite if a
demonstrated API is removed or renamed, so usage coverage cannot rot.

Markers are declared in `pyproject.toml`: `unit`, `differential`, `integration`,
`property`, and `docs`. Property tests (Hypothesis) are encouraged for algebraic
invariants such as merge associativity, encode/decode round-trips, and optimizer
idempotence, where one law covers a space no enumerated case can. The behavioral suite
above (`tests/property/`) is where they guard the optimizer, metadata, adaptive, and tuning
work. Those files carry both `property` and `integration` because they drive the native
engine, and they run under `just test-py`. Unlike the rest of the integration suite they
are counted in the coverage gate, because `tests/property` is in `cov-gate`'s measured set
(described below), so the random-search coverage of those paths is part of the ratchet.

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
just test          # the CI sequence: check, test-rust, build, test-py
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

Coverage is measured on both planes and gated as a ratchet. The floor sits just
below the achieved baseline so it blocks regressions, and it is raised as coverage grows.
It is a backstop against untested code creeping in, not a target to game. The edge
suite above is what actually proves correctness.

```bash
just cov-py        # Python control plane (pytest-cov, branch coverage)
just cov-rust      # Rust data plane (cargo-llvm-cov; one-time: cargo install cargo-llvm-cov)
just cov-gate      # the CI gate: runs the suite under coverage, fails below the floor
```

`just test` runs the full correctness suite (`test-py`) and then `cov-gate`. The
settings live in `[tool.coverage.*]` in `pyproject.toml`. The compiled `_native`
extension is omitted, because it is exercised through the data plane rather than the
Python suite, so counting it would mislead.

The gate measures a **deterministic subset**, `tests/{unit,differential,property,io,docs}`,
and deliberately excludes `tests/integration`. Those Ray, adaptive-learning, and
distributed tests are stable on their own, and they run for correctness under `test-py`,
but coverage instrumentation perturbs their timing enough to make them flaky, which
would make an enforced gate non-deterministic. The floor is **62%** branch
coverage of `python/batcher`, set just below the measured subset baseline of about 64%.
Raise the `--cov-fail-under` value in the `cov-gate` recipe whenever a round of new
tests lifts the baseline. (`just cov-py` reports the same subset with line-by-line
misses and an HTML drill-down.)

## See also

- {doc}`/architecture/internals/execution`: the sequential, parallel, and JIT paths under test.
- {doc}`/architecture/internals/kyber`: the passes the differential tests guard.
- {doc}`/architecture/internals/extending`: where a new operator, rule, or format adds its own coverage.
