# Testing strategy

This page describes how Batcher proves a change correct: the two reference oracles, the property-based layer that searches for what examples miss, the test layout, and the gates each kind of change must clear.

Batcher aims to beat DuckDB, Spark and Polars and to be correct while doing it, and the first aim is worth nothing without the second. So the rule is blunt: correctness comes before speed, and a fast wrong answer is a bug. The benchmark harness enforces that literally, refusing to time a query whose result doesn't match the oracle.

## Two oracles

Tests don't hand-roll expected values. They check against one of two references.

**DuckDB, for relational behavior.** Any operator, expression, SQL form or optimizer rewrite must produce the result DuckDB produces on the same input. The comparison helpers live in [`tests/_harness.py`](https://github.com/stephenoffer/batcher/blob/main/tests/_harness.py) and are re-exported from [`tests/differential/conftest.py`](https://github.com/stephenoffer/batcher/blob/main/tests/differential/conftest.py), beside the `duck` fixture. `assert_same` is an order-independent, type-tolerant multiset comparison that accepts int against float, Decimal against float, and float rounding. `assert_same_ordered` compares row by row, and `assert_same_for_query` picks between the two by whether the query ends in `ORDER BY`. Cases live in the `test_diff_*.py` files. If Batcher and DuckDB legitimately differ, that is a decision to surface and document, never one to hide by weakening a test.

:::{warning}
Never check a sort with `assert_same`. It sorts both sides before comparing, so an engine that returned unsorted rows would pass. A spilled descending sort once returned wrong data with every gate green for exactly this reason. Use `assert_same_ordered` for any result whose order is part of the answer.
:::

**The Tier-0 interpreter, for the Rust engine.** `bc-interp::execute`, the sequential path, is the reference. The parallel executor and the Cranelift JIT must agree with it bit for bit on every supported input. A new `bc-runtime` primitive also gets the mergeability test: `combine_finalize(partition(partial(p_k)))` over all partitions must equal the single-node result. That equality is the guarantee that one core, many cores and many machines compute the same thing.

## Property-based testing

The two oracles say what the right answer is. Property-based tests decide where to look for a wrong one. Instead of one enumerated input, Hypothesis generates many random tables and pipelines, asserts an invariant on every draw, and shrinks any failure to a minimal counterexample. This layer guards work whose correctness lives in a combination: the full optimizer rule set, the metadata shortcuts, and the adaptive and self-tuning paths. In each of them, the bug an example misses is the one that matters. The suite lives in [`tests/property/`](https://github.com/stephenoffer/batcher/tree/main/tests/property), and each file pins one invariant and drives it through an oracle. The following files are the core of it.

**Optimizer result-invariance** (`test_prop_optimizer_result_invariance.py`). Kyber's rule set must change the plan and never the answer. Hypothesis builds a random typed table and a random valid pipeline, with filters carrying redundant and absorbing boolean shapes, derived columns, group-by aggregates, distinct, sort, limit and union, and asserts the following:

```text
result(full rule set)  ==  result(no rules)  ==  ds.collect()
```

The comparison is an order-independent multiset, plus row order when the pipeline is totally ordered. Any rule, or any interaction between rules, that alters a result falls out as a counterexample to minimize and fix.

**Confluence, termination and determinism**, in the same file. Result-invariance says the rules are sound one at a time. Confluence says they behave together. The test asserts that the optimized IR at the production `optimizer.fixpoint_iterations` cap equals the IR at a far larger cap, so the combined set reaches its fixpoint within the budget the engine runs and no plan is silently truncated mid-convergence as rules are added. It also asserts that re-running is byte-identical and that re-optimizing an optimized plan is a fixpoint. That is the guarantee that a growing rule set doesn't regress into interference and oscillation.

**Metadata fidelity and the exact-only firewall** (`test_prop_metadata_fidelity.py`). The metadata shortcuts answer `count`, `is_empty`, `min`, `max`, `count_distinct` and null counts from Parquet footer statistics without scanning a row. Two properties must hold on random typed data written to Parquet and read back. Whenever a shortcut fires, its answer equals both the executed answer and DuckDB's, because a wrong footer-derived answer never scans a row that could catch it. And past a filter, where the source's bounds no longer describe the result exactly, a shortcut that can't prove its answer must decline and fall back to execution. The test asserts the decline on a partial filter and that the executed fallback still matches DuckDB.

**Adaptive equals non-adaptive** (`test_prop_adaptive_equivalence.py`). Stage-boundary re-optimization re-plans on measured cardinalities, so a join's build side or broadcast choice can flip mid-query. It must plan better, never differently. On a random selective filter feeding a join, the shape where the measured count diverges most from the estimate, `adaptive=True`, `False` and `"auto"` must produce the same rows, and the inner-join result is cross-checked against DuckDB.

**Tuning invariance** (`test_prop_tuning_invariance.py`). Every self-tuning lever is contractually result-invariant: morsel size, spill, shuffle partition count, adaptive morsel sizing, and the learned strategy choices behind them change how a query runs and never what it returns. Hypothesis runs the same aggregate or distinct at opposite settings of each knob, such as a 1-row morsel against 64k, spill forced against in-memory, and one partition against seven, and asserts byte-identical results.

**The mergeable algebra** (`test_prop_mergeable_invariant.py` and `test_prop_partition_invariant.py`). Stateful operators are `partial`, `combine` and `finalize`, with an associative, commutative `combine`, and that single invariant lets one implementation serve one core, many cores and many machines. The tests assert that an aggregate, a distinct or a sort-limit over one morsel equals the same over any random chunking of the input, and equals DuckDB. Because the native distributed primitives are callable directly, `test_prop_mergeable_invariant.py` also drives `combine_finalize(partition(partial(p_k)))` over the raw Rust kernels, so partition-independence is proven at the primitive and not only through the Python path.

The suite holds more than these, including streaming, watermark and path-equivalence properties. These invariants have caught bugs no example test surfaced, such as a crash under a concurrent plan edit and an adaptive divergence traced to an empty-input limitation in the one-shot path. A failing property is a decision to surface, exactly like a differential mismatch, and weakening it to go green isn't an option.

## Test layout

Each directory holds one kind of test, and the kind decides what it may depend on:

| Directory | What it holds |
|---|---|
| [`tests/unit/`](https://github.com/stephenoffer/batcher/tree/main/tests/unit) | Fast tests with no native engine: optimizer passes, IR validation, cost |
| [`tests/differential/`](https://github.com/stephenoffer/batcher/tree/main/tests/differential) | Results cross-checked against DuckDB and Polars, the correctness spine |
| [`tests/integration/`](https://github.com/stephenoffer/batcher/tree/main/tests/integration) | End to end: I/O, adaptive re-optimization, distributed, spilling |
| [`tests/io/`](https://github.com/stephenoffer/batcher/tree/main/tests/io) | Source and sink formats and lakehouse round-trips |
| [`tests/property/`](https://github.com/stephenoffer/batcher/tree/main/tests/property) | Hypothesis invariants: optimizer, metadata, adaptive, mergeable |
| [`tests/docs/`](https://github.com/stephenoffer/batcher/tree/main/tests/docs) | The code examples in these docs, the `examples/` scripts, and the docs structure |

[`tests/docs/`](https://github.com/stephenoffer/batcher/tree/main/tests/docs) runs two example harnesses. `test_doc_examples.py` executes the fenced `python` blocks in this documentation, and `test_examples.py` runs every script under the top-level `examples/` directory. Both fail the suite when a demonstrated API is removed or renamed.

Markers are declared in `pyproject.toml`: `unit`, `differential`, `integration`, `property`, `docs` and `io`. The property files carry both `property` and `integration`, because they drive the native engine, and they run under `just test-py`.

Coverage of the cross-product matters as much as coverage of each operator. [`tests/differential/test_diff_operator_matrix.py`](https://github.com/stephenoffer/batcher/blob/main/tests/differential/test_diff_operator_matrix.py) runs every relational operator through `collect()`, `collect(spill=True)` and `iter_batches()` on one input loaded with nulls, empty input, a single row, `-0.0` and NaN float keys, every ordering flag, and enough rows to cross a morsel boundary. It exists because four wrong-answer bugs lived where an operator met a non-default path, such as a spilled descending sort that emitted nulls mid-result.

## What each change must prove

The gate scales with what you touched:

- A new or changed operator or expression adds a differential test against DuckDB covering nulls, empties and type edges, and keeps the Rust sequential, parallel and JIT paths in agreement. Touching the JSON IR adds a round-trip test that the Python `to_ir()` shape deserializes in Rust.
- A new `bc-runtime` primitive gets a unit test and the mergeability invariant. If it is stateful, it is tested spilled and partitioned too.
- A new Kyber pass gets a unit test proving the rewrite is semantics-preserving, where the plan changes and the result doesn't, plus a differential test that the optimized query still matches DuckDB.
- A distributed change gets an equivalence test showing single-node output equals multi-worker output. CI installs no Ray, so that test is skipped there, and a recorded cluster run is the evidence.
- A bug fix lands with a regression test that fails before the fix.
- Any test change runs `just lint-tests`, which catches a test that can't fail, and `just lint-methodology`, which catches one arranged so that it doesn't.

## Running the tests

The following recipes run the suites:

```bash
just test          # the CI sequence: check, test-rust, build, test-py, cov-gate
just test-rust     # cargo test: the Rust oracle, parallel and JIT parity
just test-py       # pytest, including the differential suite and doc examples
```

`just test-py` needs a built engine, so run `just build` first, because the differential and integration suites run real queries.

## Coverage

Cover the contract rather than the implementation: every operator against empty input, nulls, a single row, multiple batches and type boundaries. A wide suite of those edges catches more than chasing a percentage, because the edges are where engines disagree.

Coverage is still measured on both planes and gated as a ratchet, as a backstop against untested code creeping in:

```bash
just cov-py        # Python control plane (pytest-cov, branch coverage)
just cov-rust      # Rust data plane (cargo-llvm-cov; one-time: cargo install cargo-llvm-cov)
just cov-gate      # the CI gate: runs the suite under coverage, fails below the floor
```

The gate measures a deterministic subset, [`tests/unit`](https://github.com/stephenoffer/batcher/tree/main/tests/unit), [`tests/differential`](https://github.com/stephenoffer/batcher/tree/main/tests/differential), [`tests/property`](https://github.com/stephenoffer/batcher/tree/main/tests/property), [`tests/io`](https://github.com/stephenoffer/batcher/tree/main/tests/io) and [`tests/docs`](https://github.com/stephenoffer/batcher/tree/main/tests/docs), and fails below 85% branch coverage of [`python/batcher`](https://github.com/stephenoffer/batcher/tree/main/python/batcher). It excludes [`tests/integration`](https://github.com/stephenoffer/batcher/tree/main/tests/integration), whose Ray, adaptive-learning and distributed tests run for correctness under `test-py` but turn flaky under coverage instrumentation. The compiled `_native` extension is omitted, because the data plane exercises it rather than the Python suite. Settings live in `[tool.coverage.*]` in `pyproject.toml`. Raise `--cov-fail-under` in the `cov-gate` recipe whenever new tests lift the baseline, because a floor nobody tightens isn't a ratchet.

## See also

- {doc}`/architecture/internals/execution`: the sequential, parallel and JIT paths under test.
- {doc}`/architecture/internals/kyber`: the rules the differential and property tests guard.
- {doc}`/architecture/internals/extending`: where a new operator, rule or format adds its own tests.
- {doc}`/benchmarks/methodology`: the same correctness gate applied to every benchmark number.
