# Caching results

A `Dataset` is a plan, not a result. Call `collect()` twice and the plan runs twice.
That surprises people coming from pandas, where a DataFrame *is* the data. Laziness is
what lets the optimizer push filters into the scan and fuse projections, so the answer
is not to make datasets eager. It is to say, once, which result you intend to reuse.

## Setup

```python
import batcher as bt

events = bt.from_pydict(
    {
        "region": ["us", "eu", "us", "eu", "us"],
        "status": ["active", "active", "churned", "active", "active"],
        "amount": [10.0, 3.0, 99.0, 9.0, 4.0],
    }
)
```

## The problem: a reused subquery runs twice

```python
active = events.filter(bt.col("status") == "active")

by_region = active.group_by("region").agg(total=bt.col("amount").sum())
rows = active.count()  # ← re-runs the scan and the filter

print(by_region.sort("region").to_pydict(), rows)
# {'region': ['eu', 'us'], 'total': [12.0, 14.0]} 4
```

Two terminals, two executions. On this data nobody notices. On a filtered 500 GB scan
feeding five downstream reports, it is five scans.

## cache()

:::{tip}
One rule covers most of this page: if a result has more than one consumer, call
`cache()` on it. Once. On the node they share.
:::

`cache()` marks a dataset's result to be kept in memory after it is first computed. The
first terminal executes normally and stores the Arrow result; later terminals on the
same cached dataset return it without re-running the plan. It is the Spark/Polars
`cache` pattern, and it is a *marker*: nothing runs when you call it.

```python
hot = events.filter(bt.col("status") == "active").cache()

first = hot.count()          # executes the plan, stores the result
second = hot.count()         # cache hit, no re-execution
totals = hot.group_by("region").agg(total=bt.col("amount").sum())

print(first, second)
# 4 4
print(totals.sort("region").to_pydict())
# {'region': ['eu', 'us'], 'total': [12.0, 14.0]}
```

The speedup is the whole plan, not a constant factor. A cached aggregate over 200,000
rows drops from hundreds of milliseconds to well under one:

```python
import time

big = bt.range(0, 200_000).with_columns(g=bt.col("value") % 7)
counts = big.group_by("g").agg(n=bt.count()).cache()

start = time.perf_counter()
counts.collect()
cold = time.perf_counter() - start

start = time.perf_counter()
counts.collect()
warm = time.perf_counter() - start

print(warm < cold)
# True
```

## It caches *this* result, not the branch

A transform on a cached dataset is a new, uncached dataset. This trips people up:

```python
base = events.filter(bt.col("amount") > 3.0).cache()
derived = base.select("region")  # ← NOT cached; a different plan

print(base.count(), derived.count())
# 4 4
```

`base` is cached. `derived` re-runs, though it re-runs *from* the cached `base`, so it is
not paying for the filter again. Cache the node every consumer shares, which is usually
the expensive join or aggregate right before the branch point, not the leaf.

## What it costs and when it gives it back

The cache is process-wide, keyed by the plan and its inputs, and bounded by
`memory.result_cache_max_bytes` (256 MB by default). It is an LRU, and it yields memory
back to running queries under pressure, so caching cannot grow the process without bound
and cannot OOM a query that needs the memory more.

```python
from batcher.config import Config

print(Config().memory.result_cache_max_bytes)
# 268435456
```

Raise it if your working set is genuinely larger and you have the headroom:

```python
# docs: skip
from batcher.config import Config, MemoryConfig, config_context

with config_context(Config().replace(memory=MemoryConfig(result_cache_max_bytes=4 << 30))):
    hot.collect()
```

:::{warning}
Two things it does not do. It does not survive the process, so it is not a checkpoint: a
crash, a new worker, or a fresh Python session loses it. And it covers single-node
relational results only, so a `map_batches` / ML pipeline or a distributed collect is not
cached at all, and no error tells you so.
:::

## When to write instead

If the result is bigger than the budget, is needed by another process, or is expensive
enough that you do not want to recompute it after a crash, the answer is not a bigger
cache. Write it and read it back.

::::{tab-set}
:::{tab-item} cache()

```python
warm = events.filter(bt.col("status") == "active").cache()
print(warm.count(), warm.group_by("region").agg(n=bt.count()).count())
# 4 2
```

In-process, in memory, bounded by the budget, gone when the process exits.

:::

:::{tab-item} Checkpoint to Parquet

```python
# docs: skip
active.write.parquet("s3://bucket/staging/active/")
active = bt.read.parquet("s3://bucket/staging/active/")
```

Durable, shareable, and the next query gets a fresh scan with statistics and predicate
pushdown.

:::
::::

That second one is a checkpoint, and it is the right call more often than people expect.
Here is the whole decision:

| Situation | Reach for |
| --- | --- |
| A result several downstream queries share, inside one process | `cache()` |
| A result larger than `result_cache_max_bytes` | write it |
| A result another process, job, or person needs | write it |
| A stage you do not want to recompute after a crash | write it |
| A `map_batches` / ML pipeline result | write it, since `cache()` does not cover it |

## See also

- {doc}`Performance <performance>`: morsel sizing, spilling, and the memory budget.
- {doc}`Explain plans <explain-plans>`: confirm the plan you cached is the plan you meant.
- {doc}`Writing data <writing-data>`: the checkpoint alternative.
- {doc}`Query lifecycle <../deep-dives/query-lifecycle>`: what "the plan runs twice" means,
  stage by stage.
- {doc}`Buffer pool <../deep-dives/buffer-pool>`: the memory the cache is yielding back
  when a running query needs it more.
- {doc}`Configuration options <../configuration/options>`: `memory.result_cache_max_bytes`
  and the rest of the memory envelope.
- {doc}`Optimizing a slow query <../tutorials/optimizing-a-slow-query>`: caching in its
  place, among the other fixes.
