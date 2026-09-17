# Execution engine

Once Kyber has optimized a plan, the execution engine runs it. The Python side
(`core`) does no per-row work: it lowers the physical plan to JSON IR, hands it to
the Rust data plane through one FFI call, and gets Arrow batches back. Everything
that touches a row happens in Rust.

```python
out, metrics_json = _native.execute_plan_metered(plan.to_json(), sources, engine_cfg, query_id)
```

The plan crosses the boundary as JSON, and the data crosses as zero-copy Arrow
`RecordBatch`es through the Arrow C Data Interface. Nothing else moves between the two
languages.

This page is the contributor's view: the tiers, the crates each path lives in, the
thresholds with their config names, and the metadata layer that answers a terminal
without a scan. {doc}`/architecture/execution` covers the shape of the model itself,
the pipeline-and-breaker structure and the lazy API, and is the better place to start.

## Execution tiers

There is one set of operator semantics, exercised by three execution paths. The
sequential interpreter is the oracle, and the other two must agree with it.

![One shared Expr and RelOp feeding three execution tiers. The Tier-0 sequential interpreter is the correctness oracle. The Tier-0 parallel path changes only scheduling and must equal the oracle. The Tier-1 Cranelift JIT must be bit-for-bit identical on its supported subset, and an unsupported expression falls back to the interpreter rather than diverging.](/_static/diagrams/execution_tiers.svg)

- **Tier-0 sequential** (`bc-interp`, `execute`) is the reference. It is simple,
  deterministic, and obviously correct, and every other path is tested against it.
- **Tier-0 parallel** (`bc-interp::par`) reuses the same operator code and changes
  only the scheduling: morselize, run on a rayon thread pool, and hash-shuffle into
  the breakers. It computes exactly what the sequential path does.
- **Tier-1 JIT** (`bc-codegen`) compiles the supported subset of column
  expressions to machine code with Cranelift. Each distinct expression compiles once into
  a process-wide cache (`bc-codegen/src/cache.rs`), keyed on the expression and the types
  of the columns it reads, and is reused across every batch, operator and query that
  shares that key. On anything it does not support, and on a batch its compiled code
  cannot evaluate, it falls back to the interpreter rather than diverge. The JIT is
  bit-for-bit identical to the interpreter on its subset.

The JIT compiles scalar expressions, not whole pipelines, so there is no compiled pipeline
for a re-plan to throw away. That is how adaptivity and compilation coexist: re-planning at
a breaker changes the operators, the relational state lives in the runtime library rather
than in generated code, and the new operators evaluate their expressions through the same
cache.

## Which crate runs which scale

The mergeable primitives, `partial(batch) -> state`, `combine(states) -> state` and
`finalize(state) -> rows`, are written once in `bc-runtime`, and three callers compose
them. That mapping is the thing to know before touching a stateful operator:

- On one core, `bc-interp`'s sequential `execute` runs it.
- On many cores, `bc-interp::par` morselizes, builds partials in parallel, and combines
  them.
- On many machines, `bc-interp::dist` composes the same `partial / combine / finalize`
  across Ray workers, with batches moving over `bc-transport`.

So a new stateful operator that has no mergeable form is not merely un-distributed. It
is capped at the sequential path, and the failure surfaces at cluster scale as wrong
results rather than as an error. `tests/integration/test_distributed.py` asserts the
invariant directly, operator by operator: single-node output must equal multi-worker
output. It calls `pytest.importorskip("ray")`, and CI installs no Ray, so a green PR gate
says nothing about that arm. A recorded cluster run is the evidence. See
{doc}`/architecture/execution` for why the algebra is shaped this way, and
{doc}`/architecture/deep-dives/operators/mergeable-algebra` for a worked example.

## The thresholds, and what they are called

The architecture page describes these behaviors. The exact gates and their config names
live here, because these are the values you change or cite in code.

Adaptive re-optimization triggers when an estimate was wrong by more than
`optimizer.reoptimize_error` (default 2x). It engages only on a query that contains a
join and whose total scan input clears 5M rows, or roughly 320 MB, for each pipeline
breaker the loop would cut at (`api/adaptive/gating.py`). That is about 10M rows for the
simplest joined shape and more for a many-join one, because each cut is what costs. A
query with no join is out at any size, which excludes more queries than the row floor
does. Most small queries never reach the loop at all.

One path skips the floor. A distributed plan that `dist.executors.plan_analysis.requires_staging`
flags, such as a breaker beneath another breaker, is staged at any size and with no join,
because the one-shot dispatcher would otherwise apply the inner breaker per partition and
return a wrong answer. There staging is a correctness requirement rather than an
optimization.

A query that clears the gate gets stage-boundary re-optimization, the same mechanism and
granularity as Spark AQE. Two things about it reach further than AQE: it runs single-node
as well as distributed, and what it measured is recorded to the `MetadataHub` and read by
the next run. See {doc}`/architecture/internals/kyber` for that
cross-query half.

Carbonite's memory envelope throttles new allocations at `memory.soft_limit` (0.85 of
the budget) and begins spilling at `memory.hard_limit` (0.90). Spilling is a property of
the runtime primitive rather than a separate operator, so the plan does not change when
a query goes out of core.

A morsel is a `RecordBatch` of 16,384 rows (`execution.morsel_rows`) or 1 MiB
(`execution.morsel_bytes`), whichever bound trips first, so a batch of wide rows splits on
bytes long before it reaches the row count. `execution.parallelism` pins the worker-thread
count, and its default of `0` lets the engine size the pool from the host: every physical
core plus a third of the SMT siblings. All three are shipped to the Rust data plane as part
of the engine config, so the Python and Rust sides never disagree about them. {doc}`/configuration/options` is the full reference.

## Answering from metadata (no scan)

The fastest execution is none at all. Before a terminal runs the engine, the
conductor asks Kyber whether the answer is *provably* derivable from the sources'
declared statistics, meaning a Parquet or ORC footer, a lakehouse manifest, or a SQL
catalog. The IO layer already opens all of those for schema and split planning. When
the answer is derivable, the terminal returns it without touching a row.

The layer covers the terminals whose result a footer can carry:

- `count()` and `is_empty()`, from the relation's exact row count (an ORC `nrows`, a
  summed Parquet footer), through row-preserving projections, and across the
  mergeable operators that keep an exact count (an empty-side join, {py:meth}`limit(0) <batcher.Dataset.limit>`, a
  UNION of exact counts).
- Global (keyless) aggregates: `min` and `max` from footer bounds, `count(*)`,
  `count(col)` from `rows - null_count`, `sum` from a catalog's recorded total,
  `count_distinct` from an exact distinct count, and `bool_and` and
  `bool_or` from a boolean column's exact min/max.
- Per-column existence and null facets: {py:meth}`null_count <batcher.Dataset.null_count>`, {py:meth}`has_nulls <batcher.Dataset.has_nulls>`, {py:meth}`all_null <batcher.Dataset.all_null>`.
- Filtered counts. `WHERE col IS NULL` is exactly the recorded null count,
  `col IS NOT NULL` is `rows - null_count`, and a provably out-of-range predicate
  (`col > max`, or `col = v` outside `[min, max]` or absent from the column's
  membership bloom) is exactly `0`. A predicate that only *partially* overlaps the
  column's range needs a histogram, so it is **not** answered and falls back.
- {py:meth}`describe() <batcher.Dataset.describe>` and `ds.meta.col(name).summary()`, as a per-column snapshot assembled from whichever
  facets are exact, omitting the rest so the caller runs the real describe for what
  is missing.
- A provably empty plan. Collection short-circuits a contradiction filter,
  `limit(0)`, an always-false predicate, or an empty-side join to a correct-schema,
  zero-row table with no scan.

### The firewall: exact or fall back

A wrong metadata answer is not a slow query. It is *silent corruption*. So the
whole layer is gated on one rule: an exact answer is produced only from statistics
that are `Provenance.EXACT` end to end. Footer min/max (on numeric, temporal,
boolean, and decimal columns), null counts, and exact row counts are EXACT; a
byte-truncated string bound, the min and max of a filtered or limited column, which
survive only as bounds, a sketch-derived distinct count, and a learned prior are not. An
inexact statistic never answers an exact terminal. It may only inform cost or back
an explicitly-named `approx_*` terminal (`approx_count_distinct`, an approximate
quantile). Provenance can only ever be *weakened* as stats propagate through the
plan through the single `weakest` and `downgrade` combiners in `plan/stats.py`, so
nothing can silently over-claim. Every shortcut returns `None`, meaning "execute normally", the
moment it cannot prove exactness. A metadata answer is therefore an optimization
that can never change a result.

This is proven, not asserted: property tests generate random data and assert the
metadata answer equals the executed answer equals DuckDB across the covered
terminals. One correctness fix the discipline caught: Parquet's `distinct_count` is
only an *estimate*, but it had been tagged EXACT, which would have let it answer an
exact {py:meth}`count_distinct <batcher.plan.expr_ir.core.Expr.count_distinct>` wrongly. It is now `SKETCH`, kept only on already-inexact
columns to inform cost and {py:meth}`approx_count_distinct <batcher.plan.expr_ir.core.Expr.approx_count_distinct>`.

```python
# docs: run
import batcher as bt

ds = bt.from_pydict({"x": [1, 2, 3, 4, 5]})
# Answered from the source's exact row count, with no scan.
assert ds.count() == 5
# Provably empty: the predicate is out of range, so is_empty() short-circuits.
assert ds.filter(bt.col("x") > 100).is_empty()
```

## See also

- {doc}`/architecture/execution`: the pipeline-and-breaker model and the lazy API,
  which this page assumes.
- {doc}`/architecture/internals/kyber`: query planning and the re-optimization loop.
- {doc}`/architecture/internals/carbonite`: memory, spill, and flow control.
- {doc}`/user-guide/analyze/metadata-shortcuts`: the metadata layer above, from the caller's
  side, with every terminal it covers.
- {doc}`/configuration/options`: every execution knob.
- {doc}`/architecture/deep-dives/query/query-lifecycle`: the same journey traced call by call.
- {doc}`/architecture/deep-dives/operators/morsel-parallelism`: how a morsel becomes a unit of scheduling.
