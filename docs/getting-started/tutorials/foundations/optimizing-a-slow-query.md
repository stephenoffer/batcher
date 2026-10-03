# Optimizing a slow query

Batcher tells you exactly what it planned and what it measured, so tuning a query is a short loop: read the plan, measure, fix the operator that dominated. This tutorial runs that loop on the most common case there is, a Python callback doing work an expression could do.

![The loop has three steps. First, read the plan with explain(), which runs nothing. Then run the query under stats() to measure it operator by operator and find the hot spot. Then fix that one operator, and run explain() and stats() again. In this tutorial, the before state is a map_batches callback, with no row estimates in the plan and the fee arithmetic running on 200,000 rows. The after state is the same arithmetic as an expression, with the filter pushed into the scan and the fee computed on 20,000 rows. Both return the same answer, with ten times less arithmetic after the fix.](/_static/diagrams/slow_query_loop.svg)

## Where this ends up

You'll take a 200,000-row query with a `map_batches` in the middle and rewrite it in one line. The following table compares the two versions:

| | Naive (`map_batches`) | Rewritten as an expression |
|---|---|---|
| Rows the fee arithmetic touches | 200,000 | 20,000 |
| Predicate pushdown | Blocked, because the UDF is opaque | The filter runs below the projection |
| Row estimates in `explain()` | `est≈?` on every operator | An estimate and its source on every operator |
| Execution tier for the arithmetic | A Python call per batch | The Rust data plane, eligible for the Cranelift JIT |
| The answer | Correct | Identical |

## 1. The data and the query

You have 200,000 events and want the total fee on the failed ones, by country. The fee is three percent of the amount, so someone reached for `map_batches` and a PyArrow multiply.

```python
import batcher as bt
import pyarrow.compute as pc

n = 200_000
events = bt.from_pydict(
    {
        "user_id": [i % 5000 for i in range(n)],
        "country": [["us", "de", "fr", "jp"][i % 4] for i in range(n)],
        "status": ["error" if i % 10 == 0 else "ok" for i in range(n)],
        "amount": [float(i % 97) for i in range(n)],
    }
)


def add_fee(batch):
    return batch.append_column("fee", pc.multiply(batch.column("amount"), 0.03))


slow = (
    events.map_batches(add_fee, output_columns=[*events.columns, "fee"])
    .filter(bt.col("status") == "error")
    .group_by("country")
    .agg(fees=bt.col("fee").sum())
    .sort("country")
)
result = slow.to_pydict()
print(result["country"], [round(v, 2) for v in result["fees"]])
# ['fr', 'us'] [14399.7, 14397.0]
```

The output is rounded because float sums carry a binary tail. Now find out what it cost.

## 2. Profile it, and read what is missing

{py:meth}`ds.stats() <batcher.Dataset.stats>` runs the query and reports what the engine measured for each operator. A `map_batches` stage is measured too, so it shows up by name:

```python
stats = slow.stats()
print("MapBatches" in [op.kind for op in stats.ops])
# True
```

The row counts and timings are real. What's missing is the plan: every operator in `explain()` reads `est≈?`.

```python
print("est≈?" in slow.explain())
# True
```

That is the diagnosis. The optimizer can't see inside a Python callback, so it can't push the filter below it. The UDF computes a `fee` on all 200,000 rows, and 180,000 of them are thrown away immediately. Every batch also round-trips through Python.

## 3. Say it as an expression instead

The UDF multiplies a column by a constant. An expression does that in Rust, in plain sight of the optimizer:

```python
fast = (
    events.with_columns(fee=bt.col("amount") * 0.03)
    .filter(bt.col("status") == "error")
    .group_by("country")
    .agg(fees=bt.col("fee").sum())
    .sort("country")
)
```

Nothing has run yet, and this time the whole plan is visible.

## 4. Read the plan

`explain()` runs the optimizer and renders the plan without executing it. Each operator carries its estimated row count and where the estimate came from.

```python
print(fast.explain())
```

:::{dropdown} The plan, on a session that has never run this query
```text
query plan (planned)                                                5 operators
───────────────────────────────────────────────────────────────────────────────
OPERATOR                              ESTIMATE  NOTES
sort  [country]                          est≈4  (default)
└─ aggregate  [by country · sum]         est≈4  (default)
   └─ project                       est≈20,000  (default)
      └─ filter  [status = error]   est≈20,000  (default)
         └─ scan  [source 0]       est≈200,000  (exact)  pushed[status = error]
```
:::

Read it bottom-up. You wrote the projection before the filter, and the optimizer pushed the predicate into the scan, so the arithmetic runs on 20,000 rows instead of 200,000:

```python
print("pushed[status = error]" in fast.explain())
# True
```

`(exact)` marks a known count; `(default)` is a prior, used until the optimizer has statistics.

## 5. Measure it

{py:meth}`stats() <batcher.Dataset.stats>` executes the query and reports what the engine measured, operator by operator.

```python
run = fast.stats()
print(run)
```

:::{dropdown} The per-operator report
```text
OP  KIND       ROWS IN  ROWS OUT   TIME  OP SHARE           OUT  BACKEND
────────────────────────────────────────────────────────────────────────
 0  sort             2         2   24µs  ░░░░░░  <1%       28 B  interp
 1  aggregate   20,000         2  623µs  █░░░░░  16%       28 B  interp
 2  project     20,000    20,000  656µs  █░░░░░  17%  273.4 KiB  interp
 3  filter     200,000    20,000  2.6ms  ████░░  67%  449.2 KiB  interp
 4  scan       200,000   200,000    2µs  ░░░░░░  <1%    3.9 MiB  interp
────────────────────────────────────────────────────────────────────────
total: 86.11 ms, 2 rows out
```

The full report continues with a `bottleneck:` line and an `operators:` line.
:::

Times vary from run to run. The row counts don't. You can read the same numbers from code:

```python
by_kind = {op.kind: op for op in run.ops}
print(by_kind["project"].rows_in)
# 20000
print(by_kind["filter"].rows_in, "->", by_kind["filter"].rows_out)
# 200000 -> 20000
print(run.rows, round(by_kind["filter"].selectivity, 2))
# 2 0.1
```

The projection sees 20,000 rows because the filter ran first. The `BACKEND` column names the tier that ran each operator: `interp`, `jit`, or `interp+jit` when part of an expression compiled with Cranelift.

`run.bottleneck` names the operator that dominated, and `run.bottleneck_summary()` says whether the run was I/O-bound or compute-bound.

:::{dropdown} Reading the wall-clock line
The `operators:` line splits the total into engine work and everything else: planning, optimization, admission, and building the Arrow result. On a small query the operators are often a few percent of the clock, and the fix is fewer, larger calls or a cached plan. `RunStats.wall_clock_summary()` prints the same line for a script.
:::

## 6. Check the estimate against the measurement

Every `OpStat` carries the optimizer's estimate beside what happened, in `est_rows` and `est_error`:

```python
agg = by_kind["aggregate"]
print(round(agg.est_rows), "->", agg.rows_out)
# 4 -> 2
```

Batcher acts on the gap between estimate and measurement in two ways:

- **Within a query**, the adaptive loop re-plans at pipeline breakers (a sort, an aggregate, a join build) when the relative error exceeds `optimizer.reoptimize_error` (2.0 by default). It engages on joined queries of 5,000,000 rows or about 320 MB per breaker, so this small query runs in one pass.
- **Across runs**, measured row counts land in the `MetadataHub`, and the optimizer plans the same shape from evidence the next time.

## 7. Stop recomputing a shared upstream

If several queries hang off one expensive intermediate, `cache()` stores its result once. The cache is bounded by `memory.result_cache_max_bytes` (256 MiB by default) and yields memory to running queries under pressure.

```python
failures = events.filter(bt.col("status") == "error").cache()
failures.collect()  # fills the cache

print(failures.count())
# 20000
print(failures.group_by("country").agg(n=bt.count()).sort("country").to_pydict())
# {'country': ['fr', 'us'], 'n': [10000, 10000]}
```

:::{tip}
`collect` or `to_pydict` fills the cache; `count()` reads a warm cache but never fills one. Transforms on a cached dataset run from the cached input.
:::

## 8. When it is memory, not CPU

Set `memory.max_memory_bytes` and the engine keeps to it. Aggregation, distinct, sort, join build, and partitioned windows all spill to disk rather than exceed the budget, with the same result:

```python
from batcher.config import Config, MemoryConfig, config_context

budget = Config().replace(memory=MemoryConfig(max_memory_bytes=64 * 1024 * 1024))
with config_context(budget):
    bounded = fast.to_pydict()

print(bounded == fast.to_pydict())
# True
```

:::{important}
In production, set `max_memory_bytes` to the real ceiling, such as the container or cgroup limit. Without a budget there is nothing to spill against.
:::

## The loop, in short

To diagnose any slow query, complete the following steps:

1. Run `explain()`. Check that the predicate reached the scan, that the projection sits above the filter, and that the join picked the build side you expected.
1. Run `stats()`. Find the operator that took the time, and check whether anything spilled.
1. Fix that operator. A Python `map_batches` in a relational pipeline is the first suspect.
1. Cache a shared upstream with `cache()`, and set a memory budget so a big query slows down instead of failing.

## Where to go next

Pick a direction by what the measurement pointed at:

::::{grid} 1 3 3 3
:gutter: 3

:::{grid-item-card} {octicon}`meter;1.1em` Performance and memory
:link: /user-guide/operate/tuning/performance
:link-type: doc
Every tuning lever, with its default.
:::

:::{grid-item-card} {octicon}`code;1.1em` Expressions
:link: /user-guide/transform/columns/expressions
:link-type: doc
What you can say without reaching for a UDF.
:::

:::{grid-item-card} {octicon}`graph;1.1em` Benchmarks
:link: /benchmarks/index
:link-type: doc
Measured, correctness-gated results, workload by workload.
:::
::::

## See also

- {doc}`Explain plans </user-guide/operate/tuning/explain-plans>`: every field in the output you just read.
- {doc}`UDFs </user-guide/transform/columns/udfs>`: when a `map_batches` is the right answer, and how to make it cost less.
- {doc}`Caching </user-guide/operate/tuning/caching>`: what `cache()` stores, and when it is evicted.
- {doc}`Adaptive re-optimization </architecture/deep-dives/adaptive/adaptive-reoptimization>`: the pipeline-breaker re-plan from step 6.
- {doc}`JIT compilation </architecture/deep-dives/query/jit-compilation>`: what `jit` and `interp+jit` in the backend column mean.
- {doc}`Spilling </architecture/deep-dives/memory/spilling>`: what happens when the budget in step 8 binds.
- {doc}`Troubleshooting </user-guide/operate/running/troubleshooting>`: the other failure modes.
