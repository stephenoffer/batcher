# Performance and memory

Batcher is built to stay fast on a laptop and survive on a cluster. This page
covers the levers a user actually reaches for: caching a result you reuse, the
morsel-driven execution model and its adaptive sizing, out-of-core spilling under a
bounded memory budget, and how to read what a query did. Every knob lives on one
frozen `Config`, applied process-wide with `set_config` or scoped to a block with
`config_context`.

## Setup

```python
import batcher as bt

events = bt.from_pydict(
    {
        "region": ["us", "eu", "us", "eu", "us", "eu"],
        "status": ["active", "active", "churned", "active", "active", "churned"],
        "amount": [10.0, 5.0, 99.0, 7.0, 3.0, 8.0],
    }
)
```

## Result caching

`cache()` marks a dataset's result to be stored in memory the first time it is
computed. A later terminal on the *same* cached dataset returns the stored result
instead of re-running the plan — the Spark/Polars `cache` pattern. Use it when an
expensive upstream (a filter, a join, an aggregation) feeds several downstream
queries.

```python
hot = events.filter(bt.col("status") == "active").cache()

first = hot.to_pydict()   # computed once, then stored
second = hot.to_pydict()  # cache hit — no re-execution
print(first == second)
# True
print(sorted(first["region"]))
# ['eu', 'eu', 'us', 'us']
```

The cache is process-wide and memory-bounded by `memory.result_cache_max_bytes`
(256 MiB by default), holding results LRU and yielding their memory back to running
queries under pressure — so caching never grows the process without bound. It marks
*this* result: a further transform on a cached dataset is a new, uncached result.

## Reusing a cached dataset

Because a cached dataset is a reusable handle, run as many terminals on it as you
like. Each is served from the one materialized result.

```python
print(hot.count())
# 4
print(hot.group_by("region").agg(total=bt.col("amount").sum()).sort("region").to_pydict())
# {'region': ['eu', 'us'], 'total': [12.0, 13.0]}
```

## Morsel-driven execution

The engine's unit of work is a *morsel* — a small Arrow `RecordBatch`, 16,384 rows
by default — sized to fit cache so scheduling stays granular and parallelism stays
even across cores. You rarely change it, but `execution.morsel_rows` and
`execution.morsel_bytes` are the levers: a morsel splits at whichever bound trips
first, so wide rows (large strings, embeddings, blobs) stay memory-bounded even at a
fixed row count. The setting is result-invariant — a morsel only batches data, it
never changes the output.

```python
from batcher.config import Config, ExecutionConfig, config_context

small_morsels = Config().replace(execution=ExecutionConfig(morsel_rows=4096))
with config_context(small_morsels):
    out = events.group_by("region").agg(total=bt.col("amount").sum()).sort("region").to_pydict()
print(out)
# {'region': ['eu', 'us'], 'total': [20.0, 112.0]}
```

`execution.adaptive_morsel_sizing` (on by default) shrinks the per-morsel target
under memory pressure so the streaming working set stays bounded when memory is
tight, and leaves it at the configured target otherwise. For per-batch UDF and
inference workloads, the `pid` section tunes the controller that grows or shrinks
the batch toward a target latency.

## Adaptive re-optimization

Every cost estimate is a guess until the query runs. At a pipeline breaker — a sort,
an aggregate, a join build — the engine has *measured* the real size of what it just
processed, and when an estimate was off by more than `optimizer.reoptimize_error`
(2x by default), it re-plans the rest of the query on the measured numbers before
continuing. This is the part static optimizers cannot match. `collect(adaptive=...)`
controls it: `"auto"` (the default) turns it on only when a join's input size is a
pure estimate, and `True`/`False` force it. The result is identical whichever way it
runs.

```python
dim = bt.from_pydict({"region": ["us", "eu"], "tier": ["gold", "silver"]})
joined = (
    events.join(dim, on="region")
    .group_by("tier")
    .agg(total=bt.col("amount").sum())
    .sort("tier")
)
print(joined.collect(adaptive=True).to_pydict())
# {'tier': ['gold', 'silver'], 'total': [112.0, 20.0]}
```

## Out-of-core spilling

Stateful operators — aggregation, distinct, sort, join build, windowed-by-partition
— spill to disk when they would exceed the memory envelope, so a query that does not
fit in memory slows down rather than dying. Spilling is a property of the runtime
primitive, not a separate plan: the result is bit-identical to the in-memory run.

You do not ask an operator to spill; you set a memory budget and the engine decides.
Setting `memory.max_memory_bytes` is what opts the in-memory engine into spilling —
the data plane receives a per-operator budget of `max_memory_bytes x hard_limit`.
A deliberately tiny budget forces the out-of-core path here so the example runs
anywhere; in production you set it to the real ceiling (honoring a container/cgroup
limit).

```python
from batcher.config import MemoryConfig

big = bt.from_pydict({"k": [i % 50 for i in range(2000)], "v": list(range(2000))})


def totals(ds: bt.Dataset) -> dict:
    return ds.group_by("k").agg(total=bt.col("v").sum()).sort("k").to_pydict()


in_memory = totals(big)

tiny_budget = Config().replace(memory=MemoryConfig(max_memory_bytes=1))
with config_context(tiny_budget):
    spilled = totals(big)

print(in_memory == spilled)
# True — the out-of-core result is identical to the in-memory one
print(len(spilled["k"]))
# 50
```

At scale the local (NVMe) spill tier overflows to `memory.spill_remote_uri` (any
fsspec URL) when local disk fills, and a skewed aggregate bucket that overflows
`memory.spill_bucket_max_bytes` is re-partitioned and reduced one piece at a time —
so a large or skewed query degrades gracefully instead of running out of memory.

## Reading a query plan

`explain()` runs the optimizer and renders the optimized plan with per-operator
cardinality estimates, without executing — the way to confirm a predicate landed at
the scan or a join was reordered the way you expected.

```python
print(events.filter(bt.col("status") == "active").select("region", "amount").explain())
# A text rendering of the optimized plan, annotated with estimated row counts.
```

Where `explain()` shows the *planned* shape, `stats()` runs the query and reports
what the engine *measured* — rows in/out, wall time, peak bytes, spill, and the
operator that dominated wall time.

```python
run = events.group_by("region").agg(total=bt.col("amount").sum()).stats()
print(run.rows)
# 2
print(run.bottleneck is not None)
# True — the operator that took the most wall time
```

For a quick per-column read of the data itself (counts, null fraction, approximate
distinct count) before a load, `profile()` executes a one-row-per-column summary.

```python
print(events.profile().columns)
# ['column', 'count', 'null_count', 'null_fraction', 'approx_distinct']
```

## Tuning checklist

Reach for these in order; most workloads need none of them.

- **A result reused across queries** — `cache()` the shared upstream.
- **Bounded or container memory** — set `memory.max_memory_bytes` to the real
  ceiling so stateful operators spill instead of OOMing.
- **Wide rows (blobs, embeddings)** — lower `execution.morsel_bytes` to keep the
  working set bounded; leave `morsel_rows` alone.
- **A query slower than expected** — `explain()` to check the plan, then `stats()`
  to find the operator that dominated wall time.
- **A cluster shuffle under memory pressure** — the credit-based backpressure in
  `flow_control` and `distributed`; see [Fault tolerance](../architecture/fault-tolerance.md).

Every field, with its default and meaning, is in
[Configuration options](../configuration/options.md).

## Next steps

- [Configuration options](../configuration/options.md): the full `Config` reference.
- [Fault tolerance](../architecture/fault-tolerance.md): how a distributed query
  survives task, worker, and node failures.
- [Aggregations](aggregations.md): the breakers that spill and re-optimize.
