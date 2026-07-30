# Execution engine

Once Kyber has optimized a plan, the execution engine runs it. The Python side
(`core`) does no per-row work: it lowers the physical plan to JSON IR, hands it to
the Rust data plane through one FFI call, and gets Arrow batches back. Everything
that touches a row happens in Rust.

```python
out, metrics = _native.execute_plan_metered(plan.to_json(), sources, cfg.engine_config_json())
```

The plan crosses the boundary as JSON; the data crosses as zero-copy Arrow
`RecordBatch`es (the Arrow C Data Interface). Nothing else moves between the two
languages.

This page is the contributor's view: the tiers, the crates each path lives in, the
thresholds with their config names, and the metadata layer that answers a terminal
without a scan. {doc}`../architecture/execution` covers the shape of the model itself,
the pipeline-and-breaker structure and the lazy API, and is the better place to start.

## Execution tiers

There is one set of operator semantics, exercised by three execution paths. The
sequential interpreter is the oracle; the other two must agree with it.

![One shared Expr and RelOp feeding three execution tiers. The Tier-0 sequential interpreter is the correctness oracle. The Tier-0 parallel path changes only scheduling and must equal the oracle. The Tier-1 Cranelift JIT must be bit-for-bit identical on its supported subset, and an unsupported expression falls back to the interpreter rather than diverging.](../_static/diagrams/execution_tiers.svg)

- **Tier-0 sequential** (`bc-interp`, `execute`) is the reference. It is simple,
  deterministic, and obviously correct, and every other path is tested against it.
- **Tier-0 parallel** (`bc-interp::par`) reuses the same operator code and changes
  only the scheduling: morselize, run on a rayon thread pool, and hash-shuffle into
  the breakers. It computes exactly what the sequential path does.
- **Tier-1 JIT** (`bc-codegen`) compiles the supported subset of column
  expressions to machine code with Cranelift, compiling once per operator and reusing
  that across every morsel. On anything it does not support it falls back to the interpreter
  rather than diverge. The JIT is bit-for-bit identical to the interpreter on its
  subset.

A query can drop from a compiled pipeline back to the interpreter at any breaker.
That is what lets adaptivity and compilation coexist: an artifact can be thrown
away and the relational state, which lives in the runtime library rather than in
generated code, survives.

## Which crate runs which scale

The mergeable primitives (`partial(batch) → state`, `combine(states) → state`,
`finalize(state) → rows`) are written once in `bc-runtime`, and three callers compose
them. That mapping is the thing to know before touching a stateful operator:

- On one core, `bc-interp`'s sequential `execute` runs it.
- On many cores, `bc-interp::par` morselizes, builds partials in parallel, and combines
  them.
- On many machines, `bc-interp::dist` composes the same `partial / combine / finalize`
  across Ray workers, with batches moving over `bc-transport`.

So a new stateful operator that has no mergeable form is not merely un-distributed. It
is capped at the sequential path, and the failure surfaces at cluster scale as wrong
results rather than as an error. CI asserts the invariant directly: single-node output
must equal multi-worker output for every stateful operator. See
{doc}`../architecture/execution` for why the algebra is shaped this way, and
{doc}`../deep-dives/mergeable-algebra` for a worked example.

## The thresholds, and what they are called

The architecture page describes these behaviors; the exact gates and their config names
live here, because these are the values you change or cite in code.

Adaptive re-optimization triggers when an estimate was wrong by more than
`optimizer.reoptimize_error` (default 2x). It engages only on a query that contains a
join and whose total scan input clears 20M rows or roughly 1.3 GB
(`api/adaptive/gating.py`), so most small queries never reach it. Be precise about what
that buys: this is stage-boundary re-optimization, the same mechanism and granularity as
Spark AQE, not something finer. The two places the loop reaches further than AQE are
that it runs single-node as well as distributed, and that what it measured is recorded
to the MetadataHub and read by the *next* run. See {doc}`kyber` for that cross-query
half.

Carbonite's memory envelope throttles new allocations at `memory.soft_limit` (0.85 of
the budget) and begins spilling at `memory.hard_limit` (0.90). Spilling is a property of
the runtime primitive rather than a separate operator, so the plan does not change when
a query goes out of core.

The morsel is a `RecordBatch` of 16,384 rows by default (`execution.morsel_rows`), and
`execution.parallelism` sets the worker-thread count (`0` uses every core). Both are
shipped to the Rust data plane as part of the engine config, so the Python and Rust
sides never disagree about them. {doc}`../configuration/options` is the full reference.

## Answering from metadata (no scan)

The fastest execution is none at all. Before a terminal runs the engine, the
conductor asks Kyber whether the answer is *provably* derivable from the sources'
declared statistics, meaning a Parquet or ORC footer, a lakehouse manifest, or a SQL
catalog. The IO layer already opens all of those for schema and split planning. When
the answer is derivable, the terminal returns it without touching a row.

The layer covers the terminals whose result a footer can carry:

- `count()` and `is_empty()`, from the relation's exact row count (an ORC `nrows`, a
  summed Parquet footer), through row-preserving projections, and across the
  mergeable operators that keep an exact count (an empty-side join, `limit(0)`, a
  UNION of exact counts).
- Global (keyless) aggregates: `min` and `max` from footer bounds, `count(*)`,
  `count(col)` from `rows − null_count`, `sum` from a catalog's recorded total,
  `n_unique` and `count_distinct` from an exact distinct count, and `bool_and` and
  `bool_or` from a boolean column's exact min/max.
- Per-column existence and null facets: `null_count`, `has_nulls`, `all_null`.
- Filtered counts. `WHERE col IS NULL` is exactly the recorded null count,
  `col IS NOT NULL` is `rows − null_count`, and a provably out-of-range predicate
  (`col > max`, or `col = v` outside `[min, max]` or absent from the column's
  membership bloom) is exactly `0`. A predicate that only *partially* overlaps the
  column's range needs a histogram, so it is **not** answered and falls back.
- `describe()` and `summary()`, as a per-column snapshot assembled from whichever
  facets are exact, omitting the rest so the caller runs the real describe for what
  is missing.
- A provably-empty plan. `_collect` short-circuits a contradiction filter,
  `limit(0)`, an always-false predicate, or an empty-side join to a correct-schema,
  zero-row table with no scan.

### The firewall: exact or fall back

A wrong metadata answer is not a slow query. It is *silent corruption*. So the
whole layer is gated on one rule: an exact answer is produced only from statistics
that are `Provenance.EXACT` **end to end**. Footer min/max (on numeric, temporal,
boolean, and decimal columns), null counts, and exact row counts are EXACT; a
byte-truncated string bound, a filtered/limited column (whose min/max survive only
as *bounds*), a sketch-derived distinct count, and a learned prior are **not**. An
inexact statistic never answers an exact terminal. It may only inform cost or back
an explicitly-named `approx_*` terminal (`approx_n_unique`, an approximate
quantile). Provenance can only ever be *weakened* as stats propagate through the
plan (via the single `weakest`/`downgrade` combiner), so nothing can silently
over-claim. Every shortcut returns `None`, meaning "execute normally", the
moment it cannot prove exactness. A metadata answer is therefore an optimization
that can never change a result.

This is proven, not asserted: property tests generate random data and assert the
metadata answer equals the executed answer equals DuckDB across the covered
terminals. One correctness fix the discipline caught: Parquet's `distinct_count` is
only an *estimate*, but it had been tagged EXACT, which would have let it answer an
exact `count_distinct` wrongly. It is now `SKETCH`, kept only on already-inexact
columns to inform cost and `approx_count_distinct`.

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

- {doc}`../architecture/execution`: the pipeline-and-breaker model and the lazy API,
  which this page assumes.
- {doc}`kyber`: query planning and the re-optimization loop.
- {doc}`carbonite`: memory, spill, and flow control.
- {doc}`../user-guide/metadata-shortcuts`: the metadata layer above, from the caller's
  side, with every terminal it covers.
- {doc}`../configuration/options`: every execution knob.
- {doc}`../deep-dives/query-lifecycle`: the same journey traced call by call.
- {doc}`../deep-dives/morsel-parallelism`: how a morsel becomes a unit of scheduling.
