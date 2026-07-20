# Optimizing a slow query

A query is slow. This tutorial is the loop you run to find out why: look at the plan, run it
under measurement, find the operator that dominated, fix *that*. Guessing is the thing to avoid.

The bug in this page is a real one, and it is the most common one there is.

:::{note}
**What you'll build.** A 200,000-row query with a `map_batches` in the middle of it, a
diagnosis, and a one-line rewrite. You need `pip install batcher-engine` and nothing else.
Everything runs as written, in memory, in under a second.
:::

## Where this ends up

The whole tutorial is this table. The naive version puts a Python callback between the scan
and the filter, and the optimizer cannot see through it.

| | Naive (`map_batches`) | Rewritten as an expression |
|---|---|---|
| Rows the fee arithmetic touches | **200,000** | **20,000** |
| Predicate pushdown | Blocked: the UDF is opaque | The filter runs *below* the projection |
| `project` backend | No per-operator metrics at all | **`interp+jit`**, compiled once, reused per morsel |
| `stats()` / `explain(analyze=True)` | Raises `BackendError` | A full per-operator report |
| The answer | Correct | Identical |

Both versions are right. One of them does ten times the arithmetic to get there, and will
not tell you that it did.

## 1. The data and the query

200,000 events. You want the total fee on the failed ones, by country. The fee is three
percent of the amount, so someone reached for `map_batches` and a NumPy multiply.

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

(Float sums do not land on round numbers, hence the rounding: the raw values carry the usual
binary-floating-point tail.)

Correct answer. Now find out what it cost.

## 2. Ask the engine to profile it, and read the refusal

`ds.stats()` runs the query and reports what the engine *measured* per operator. Here it
refuses:

```python
from batcher._internal.errors import BackendError

try:
    slow.stats()
except BackendError as exc:
    print(type(exc).__name__)
    print("map_batches" in str(exc))
# BackendError
# True
```

> `explain(analyze=True)/stats() is not available for map_batches/ML pipelines (the opaque
> UDF path emits no per-operator metrics); profile the relational portion instead.`

:::{warning}
That message is the diagnosis, not an obstacle. A `map_batches` is a Python callback: the
optimizer cannot see inside it, cannot know what columns it reads, and cannot know what it
does to the row count. It is a wall in the middle of the plan, and the engine cannot profile
across it.
:::

Two things follow, and both are costing you time:

- **The filter cannot move below it.** A predicate can only be pushed past an operator the
  optimizer understands. So the UDF runs on all 200,000 rows to produce a `fee` column that
  180,000 of them will immediately throw away.
- **Every batch round-trips through Python.** Even an *identity* `map_batches` roughly halves
  throughput on a native pipeline. That is not a figure of speech. It is the measured
  effect that made image ingest 2× slower until the re-type UDF was removed from the read
  path, and you can read the whole story on the [multimodal ingest
  benchmark](../benchmarks/multimodal-ingest.md).

## 3. Say it as an expression instead

The UDF multiplies a column by a constant. Expressions do that, in Rust, and the optimizer
can see through them.

```python
fast = (
    events.with_columns(fee=bt.col("amount") * 0.03)
    .filter(bt.col("status") == "error")
    .group_by("country")
    .agg(fees=bt.col("fee").sum())
    .sort("country")
)
```

Nothing has run. It is a lazy plan, and this time a fully relational one, so you can look at
it before you pay for it.

## 4. Read the plan

`explain()` runs the optimizer and renders the plan **without executing**, annotated with
each operator's estimated row count and where the estimate came from.

```python
print(fast.explain())
```

:::{dropdown} The plan, on a session that has never run this query
```text
sort                            est≈2,000 (default)
  aggregate                     est≈2,000 (default)
    project                     est≈20,000 (default)
      filter                    est≈20,000 (default)
        scan                    est≈200,000 (exact)
```
:::

Read it bottom-up, and notice what moved. You *wrote* the projection before the filter. The
optimizer put the filter **underneath** it, because it can now see that `fee` is not needed
to evaluate `status == 'error'`. The arithmetic will run on 20,000 rows, not 200,000.

That is the whole fix, and it was unavailable while the UDF stood in the way.

Now look at the estimates. `est≈2,000 (default)` on the aggregate is a prior, not a
fact: the optimizer has no statistics on the cardinality of `country` yet, so it guesses. It is
about to be wrong by three orders of magnitude. Hold that thought.

## 5. Measure it

`stats()` executes the query and reports what the engine measured, operator by operator.

```python
run = fast.stats()
print(run)
```

:::{dropdown} The full per-operator report
```text
 op  kind             rows_in    rows_out        ms      out_kb  backend
------------------------------------------------------------------------
  0  sort                   2           2      0.02           0  interp
  1  aggregate          20000           2      0.07           0  interp
  2  project            20000       20000      2.03           0  interp+jit
  3  filter            200000       20000      2.03           0  interp
  4  scan              200000      200000      0.09        3964  interp
------------------------------------------------------------------------
total: 12.58 ms, 2 rows out
bottleneck: project (op 2), 16% of wall time — compute-bound (project)
```
:::

Times vary run to run; the row counts do not. Three things in that table are worth pulling
out:

```python
by_kind = {op.kind: op for op in run.ops}
print(by_kind["project"].rows_in)
# 20000
print(by_kind["filter"].rows_in, "->", by_kind["filter"].rows_out)
# 200000 -> 20000
print(by_kind["project"].backend)
# interp+jit
print(run.rows, round(by_kind["filter"].selectivity, 2))
# 2 0.1
```

The projection sees 20,000 rows because the filter ran first. And its backend is
`interp+jit`: the expression was compiled by Cranelift once and reused across every morsel,
which is a thing that cannot happen to a Python callback.

`run.bottleneck` names the operator that dominated wall time and `run.bottleneck_summary()`
says whether the run was I/O-bound or compute-bound. That is where you look next.

## 6. The estimate was wrong, and the engine noticed

The plan guessed 2,000 rows out of the aggregate. The measurement says 2. `OpStat.est_rows`
and `OpStat.est_error` are how you catch the estimate that is lying to your join order,
because an estimate off by that much is exactly what steers a join into a 12M-row
intermediate.

Two things protect you, and both are why that measurement was worth taking.

*During* the query, the engine re-optimizes at pipeline breakers. A sort, an aggregate, a
join build is a point where it has just measured the true size of what it processed, and when
the estimate was off by more than `optimizer.reoptimize_error` (2× by default) it re-plans
the rest of the query on real numbers. That is the part a static optimizer cannot do.

*After* the query, Core records the measurement into the MetadataHub, and Kyber reads it on
the next run. The same `explain()` call now says so:

```python
print("learned" in fast.explain())
# True
```

Core measures, Kyber decides. Run a query twice and the second plan is built on facts.

## 7. Stop recomputing a shared upstream

If several queries hang off one expensive intermediate, `cache()` materializes it once. It is
memory-bounded (256 MiB by default, LRU, released under pressure), so it cannot grow the
process without bound.

```python
failures = events.filter(bt.col("status") == "error").cache()

print(failures.count())
# 20000
print(failures.group_by("country").agg(n=bt.count()).sort("country").to_pydict())
# {'country': ['fr', 'us'], 'n': [10000, 10000]}
```

:::{tip}
The second terminal is served from the stored result. `cache()` marks *that*
dataset, so a further transform on it is a new, uncached dataset. If you cache and see no
speedup, check that you are re-running the cached handle rather than something derived from
it.
:::

## 8. When it is memory, not CPU

A query that dies is slower than a query that is slow. Setting `memory.max_memory_bytes` is
what opts the engine into spilling: aggregation, distinct, sort, join build and partitioned
windows all spill to disk rather than exceeding the envelope, and the spilled result is
bit-identical to the in-memory one.

```python
from batcher.config import Config, MemoryConfig, config_context

budget = Config().replace(memory=MemoryConfig(max_memory_bytes=64 * 1024 * 1024))
with config_context(budget):
    bounded = fast.to_pydict()

print(bounded == fast.to_pydict())
# True
```

:::{important}
In production you set that to the real ceiling, the container or cgroup limit, and let
Carbonite decide when to spill. You do not ask an operator to spill; you give the engine a
budget and it obeys it. Leave `max_memory_bytes` unset and the engine has no envelope to
respect, so a query that outgrows RAM dies instead of degrading. The spilled result is
bit-identical to the in-memory one, so there is nothing to trade away.
:::

## The loop, in short

To diagnose any slow query, complete the following steps:

1. Run `explain()`. Is the predicate at the scan? Is the projection above the filter? Did the
   join pick the build side you expected?
1. Run `stats()`. Which operator actually ate the wall time, and did anything spill?
1. Fix that operator. A Python `map_batches` in the middle of a relational pipeline is the
   first thing to suspect, because it blocks both pushdown and the JIT.
1. Cache a shared upstream with `cache()`, and set a memory budget so a big query degrades
   instead of dying.

## What you learned

::::{grid} 1 3 3 3
:gutter: 3

:::{grid-item-card} {octicon}`meter;1.1em` Performance and memory
:link: ../user-guide/performance
:link-type: doc
Every tuning lever, with its default.
:::

:::{grid-item-card} {octicon}`code;1.1em` Expressions
:link: ../user-guide/expressions
:link-type: doc
What you can say without reaching for a UDF.
:::

:::{grid-item-card} {octicon}`graph;1.1em` Benchmarks
:link: ../benchmarks/index
:link-type: doc
Where the engine is fast, and where it is not.
:::
::::

## See also

- [Explain plans](../user-guide/explain-plans.md): every field in the output you just read.
- [UDFs](../user-guide/udfs.md): when a `map_batches` *is* the right answer, and how to make
  it cost less.
- [Caching](../user-guide/caching.md): what `cache()` stores, and when it is evicted.
- [Adaptive re-optimization](../deep-dives/adaptive-reoptimization.md): the pipeline-breaker
  re-plan that step 6 relies on.
- [JIT compilation](../deep-dives/jit-compilation.md): what `interp+jit` in the backend
  column actually means.
- [Spilling](../deep-dives/spilling.md): what happens when the budget in step 8 binds.
- [Troubleshooting](../user-guide/troubleshooting.md): the other failure modes.
