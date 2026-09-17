# Optimizing a slow query

This tutorial teaches the loop you run when a query is slow: read the plan, run the query under measurement, find the operator that dominated, and fix that one. Batcher tells you what it planned and what it measured, so you never have to guess.

The bug on this page is the most common one there is: a Python callback doing work an expression could do.

The loop has three steps, and you go around it again after every fix:

![The loop has three steps. First, read the plan with explain(), which runs nothing. Then run the query under stats() to measure it operator by operator and find the hot spot. Then fix that one operator, and run explain() and stats() again. In this tutorial, the before state is a map_batches callback, with no row estimates in the plan and the fee arithmetic running on 200,000 rows. The after state is the same arithmetic as an expression, with the filter pushed into the scan and the fee computed on 20,000 rows. Both return the same answer, with ten times less arithmetic after the fix.](/_static/diagrams/slow_query_loop.svg)

:::{note}
**What you'll build.** A 200,000-row query with a `map_batches` in the middle of it, a diagnosis, and a one-line rewrite. You need `pip install batcher-engine` and nothing else. Everything runs as written, in memory, in under a second.
:::

## Where this ends up

The naive version puts a Python callback between the scan and the filter. The rewrite says the same arithmetic as an expression. The following table compares the two:

| | Naive (`map_batches`) | Rewritten as an expression |
|---|---|---|
| Rows the fee arithmetic touches | 200,000 | 20,000 |
| Predicate pushdown | Blocked, because the UDF is opaque | The filter runs below the projection |
| Row estimates in `explain()` | `est≈?` on every operator | An estimate and its source on every operator |
| Execution tier for the arithmetic | A Python call per batch | The Rust data plane, eligible for the Cranelift JIT |
| The answer | Correct | Identical |

Both versions are right. One of them does ten times the arithmetic to get there.

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

Float sums carry the usual binary floating-point tail, which is why the output is rounded. The answer is correct. Now find out what it cost.

## 2. Profile it, and read what is missing

{py:meth}`ds.stats() <batcher.Dataset.stats>` runs the query and reports what the engine measured for each operator. A `map_batches` stage is measured too, so it shows up by name:

```python
stats = slow.stats()
print("MapBatches" in [op.kind for op in stats.ops])
# True
```

The row counts and timings are real. What's missing is the plan. Run `explain()` on `slow` and every operator reads `est≈?`, with a note that the plan contains a UDF stage shown un-lowered, so there is no optimized tree and no row estimate anywhere in it.

:::{warning}
That absence is the diagnosis. A `map_batches` is a Python callback. The optimizer can't see inside it, can't know which columns it reads, and can't know what it does to the row count. The engine can time it, but it can't plan around it.
:::

Two costs follow. First, the filter can't move below the callback, because a predicate is only pushed past an operator the optimizer understands. The UDF runs on all 200,000 rows to produce a `fee` column that 180,000 of them immediately throw away.

Second, every batch round-trips through Python. A per-batch Python UDF in the middle of a native pipeline costs about half its throughput even when it does nothing. That was measured on image ingest, where removing one re-typing `map_batches` from the read path took decode from 2,000 to 4,600 images per second. The {doc}`multimodal ingest benchmark </benchmarks/results/multimodal-ingest>` has the full account.

## 3. Say it as an expression instead

The UDF multiplies a column by a constant. Expressions do that in Rust, and the optimizer can see through them.

```python
fast = (
    events.with_columns(fee=bt.col("amount") * 0.03)
    .filter(bt.col("status") == "error")
    .group_by("country")
    .agg(fees=bt.col("fee").sum())
    .sort("country")
)
```

Nothing has run. This is a lazy plan, and this time a fully relational one, so you can look at it before you pay for it.

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

Read it bottom-up and notice what moved. You wrote the projection before the filter. The optimizer put the filter underneath it, and pushed the predicate into the scan, because it can now see that `fee` isn't needed to evaluate `status == 'error'`. The arithmetic runs on 20,000 rows, not 200,000.

That is the whole fix. It was unavailable while the UDF stood in the way.

The notes column matters too. `(exact)` on the scan is a known count. `(default)` on everything above it is a prior, because the optimizer has no statistics on these columns yet.

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

The projection sees 20,000 rows because the filter ran first. The `BACKEND` column names the tier that ran each operator's expressions: `interp`, `jit`, or `interp+jit` when some sub-expressions compiled with Cranelift and others fell back. A Python callback never appears there, because it never reaches either tier.

`run.bottleneck` names the operator that dominated the engine's own time, and `run.bottleneck_summary()` says whether the run was I/O-bound or compute-bound. That's where you look next.

Read the `operators:` line first, though. The total covers the whole terminal call, and the operators cover only the engine work inside it. The rest is planning, optimization, admission, and building the Arrow result. When operators are a large share of the clock, the table is worth reading. On a small query they are often a few percent, and then nothing in the table is what you were waiting for: the fix is fewer, larger calls or a cached plan. `RunStats.wall_clock_summary()` prints the same line for a script.

## 6. Check the estimate against the measurement

Every `OpStat` carries the optimizer's estimate beside what happened, in `est_rows` and `est_error`:

```python
agg = by_kind["aggregate"]
print(round(agg.est_rows), "->", agg.rows_out)
# 4 -> 2
```

Here the prior was close. On a join it can be badly wrong, and a bad estimate is what steers join order into huge intermediates. In the TPC-H record, a cold q5 with no learned distinct counts was steered into 12M to 18M-row intermediates. Comparing `est_rows` with `rows_out` is how you catch that.

Batcher acts on the gap in two ways. Within a query, the adaptive loop re-plans at pipeline breakers, such as a sort, an aggregate, or a join build, when the relative error `|actual - estimate| / estimate` exceeds `optimizer.reoptimize_error` (2.0 by default). That loop engages on joined queries large enough to justify staging, at 5,000,000 rows or about 320 MB for each breaker it would cut at, so the small query on this page runs in one pass.

Across runs, Core records each operator's measured row count alongside Kyber's prediction into the `MetadataHub`, and Kyber reads it the next time it plans that shape. Core measures and Kyber decides. That TPC-H q5 ran 7,115 ms cold and 300 ms warm.

## 7. Stop recomputing a shared upstream

If several queries hang off one expensive intermediate, `cache()` stores its result once. The cache is bounded by `memory.result_cache_max_bytes`, 256 MiB by default, and yields memory back to running queries under pressure, so it can't grow the process without bound.

```python
failures = events.filter(bt.col("status") == "error").cache()
failures.collect()  # fills the cache

print(failures.count())
# 20000
print(failures.group_by("country").agg(n=bt.count()).sort("country").to_pydict())
# {'country': ['fr', 'us'], 'n': [10000, 10000]}
```

:::{tip}
A terminal that materializes the result, such as `collect` or `to_pydict`, is what fills the cache. `count()` is served from a warm cache but never fills one. A transform on the cached dataset, such as the `group_by` above, is a new result that runs from the cached input rather than recomputing the filter. If you cache and see no speedup, check that something filled it.
:::

## 8. When it is memory, not CPU

A query that dies is slower than a query that is slow. Setting `memory.max_memory_bytes` opts the engine into spilling. Aggregation, distinct, sort, join build, and partitioned windows all spill to disk rather than exceed the envelope, and the spilled result is bit-identical to the in-memory one.

```python
from batcher.config import Config, MemoryConfig, config_context

budget = Config().replace(memory=MemoryConfig(max_memory_bytes=64 * 1024 * 1024))
with config_context(budget):
    bounded = fast.to_pydict()

print(bounded == fast.to_pydict())
# True
```

:::{important}
In production, set `max_memory_bytes` to the real ceiling, such as the container or cgroup limit, and let Carbonite decide when to spill. You don't ask an operator to spill. You give the engine a budget and it keeps to it. Leave the budget unset and the engine has no envelope to respect, so a query that outgrows RAM fails instead of slowing down.
:::

## The loop, in short

To diagnose any slow query, complete the following steps:

1. Run `explain()`. Check that the predicate reached the scan, that the projection sits above the filter, and that the join picked the build side you expected.
1. Run `stats()`. Find the operator that took the time, and check whether anything spilled.
1. Fix that operator. A Python `map_batches` in the middle of a relational pipeline is the first thing to suspect, because it blocks pushdown and keeps the arithmetic out of the Rust tiers.
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
