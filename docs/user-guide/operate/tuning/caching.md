# Caching results

A {py:class}`Dataset <batcher.Dataset>` is a plan, not a result. Call {py:meth}`collect() <batcher.Dataset.collect>` twice and the plan runs twice.
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

`cache()` marks a dataset's result to be kept after it is first computed. The first
terminal executes normally and stores the Arrow result; later terminals on the same cached
dataset return it without re-running the plan. It is the Spark and Polars `cache` pattern,
and it is a *marker*: nothing runs when you call it. Where the stored result lives is a
separate question, answered by the storage level below.

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

### Which terminals fill it, and which only read it

A terminal that materializes the result fills the cache: `collect`, `to_arrow`,
`to_pydict`, `to_pylist`, and the pandas and Polars conversions.

`count()`, `is_empty()` and `iter_batches()` are served from a warm cache but never fill a
cold one. That is deliberate rather than an omission: all three are built to answer without
materializing the result, so filling the cache from one would materialize exactly what the
caller asked not to. Warm the cache with one `collect()` and every later call to any of them
is a hit.

```python
warm = events.filter(bt.col("status") == "active").cache()
warm.collect()                 # fills the cache
print(warm.count())            # served from it, no second scan
# 4
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
`memory.result_cache_max_bytes` (256 MB by default). It yields memory back to running
queries under pressure, so caching cannot grow the process without bound and cannot OOM a
query that needs the memory more.

Eviction is cost-aware rather than purely recent. Each entry is ranked by how long it took
to compute, how often it has been served, and how much memory it holds, so an expensive,
small, frequently-read result outlives a cheap, large, cold one. Plain LRU gets that
backwards whenever cached results differ by orders of magnitude in either dimension, which
they usually do.

The ranking also ages. A result that was read heavily during a warmup and is then never
asked for again does not hold its place forever: entries are scored against a floor that
rises as the cache evicts, so a working set that moves on can displace the one before it.
You do not have to clear the cache between phases of a job to stop an early query from
crowding out a later one.

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

## Storage levels

What the memory budget evicts is not necessarily lost. By default an evicted result is
written to a local disk tier and read back on the next recall, so a working set larger
than the memory budget costs a decompress rather than a full recompute. The
{py:class}`StorageLevel <batcher.StorageLevel>` argument to `cache()` chooses how far that
goes, using the same three names Spark uses:

| Level | Where the result may live | Reach for it when |
| --- | --- | --- |
| `MEMORY_AND_DISK` | Memory, demoting to local disk on eviction | The default. Almost always right. |
| `MEMORY_ONLY` | Memory only; an evicted result is recomputed | Recompute is genuinely cheaper than a disk read, or the scratch volume is precious |
| `DISK_ONLY` | Disk only; never charged against the memory budget | A result far larger than the memory budget that you still do not want to recompute |

```python
hot = events.filter(bt.col("status") == "active").cache("memory_and_disk")
wide = events.group_by("region").agg(total=bt.col("amount").sum()).cache(bt.StorageLevel.DISK_ONLY)

print(hot.count(), wide.count())
# 4 2
```

The disk tier is bounded by `memory.result_cache_disk_max_bytes` (4 GB by default) and
lives beside the spill files, on `memory.spill_dir` or whichever fast local volume the node
reports. When local disk fills it overflows to `memory.spill_remote_uri`, so a cache on a
small-NVMe node degrades to object storage instead of failing. Set the budget to `0` to
turn the tier off, which makes every level behave as `MEMORY_ONLY`.

Spark's serialized (`_SER`) and replicated (`_2`) variants have no counterpart here. The
in-memory form is already Arrow, which is the serialized form, so there is no second
representation to pick between.

## Dropping a cached result

Caching is self-managing: the budget evicts, and memory pressure reclaims. Reach for
{py:meth}`uncache() <batcher.Dataset.uncache>` only when you know a result is stale or
finished with and you want its memory and disk back at a known moment.

```python
hot.uncache()  # spelled unpersist() if you are coming from Spark
print(hot.count())  # recomputed, same answer
# 4
```

{py:func}`bt.clear_cache() <batcher.clear_cache>` does the same for every cached result at
once, which is what you want between benchmark runs.

## Checking whether the cache is earning its budget

{py:func}`bt.cache_stats() <batcher.cache_stats>` reports both tiers. Read two numbers
together, because on their own they point the wrong way:

```python
measured = events.group_by("status").agg(n=bt.count()).cache()

before = bt.cache_stats()
measured.collect()  # miss: computed and stored
measured.collect()  # hit: served from the cache
after = bt.cache_stats()

print(after["hits"] - before["hits"], after["misses"] - before["misses"])
# 1 1
```

Take a difference across two readings, as above. The counters are lifetime figures for the
process, so they are not reset by `clear_cache()`: a number that reset whenever the cache
was emptied could not tell you whether the cache was worth its budget.

A low hit rate **with** evictions means the budget is too small: the results were dropped
before anything read them again. A low hit rate with **no** evictions means the cache is
not useful for this workload, and a bigger budget will not change that. The same reading
applies to the second tier: `demotions` with no `disk_hits` means results are being
written down that nothing reads back, and the disk budget is being spent for nothing.

:::{warning}
Two things the process cache does not do. It does not survive the process, so it is not a
checkpoint: a crash, a new worker, or a fresh Python session loses it, disk tier included.
And it covers single-node relational results only, so a `map_batches` or ML pipeline result
is not cached at all, and neither is a `write`, which streams to its sink rather than
materializing. No error tells you so in either case.
:::

## Sharing results across processes

Both tiers above die with the process, and its keys are each input's *object* identity,
which is exact inside one process and meaningless outside it. That leaves out the workload
most likely to repeat itself: a scheduled job, a dashboard, or a notebook that re-issues
the same query from a fresh interpreter every time.

Point `memory.shared_cache_uri` at a store and those runs share results. The scheme picks
the backend:

| URI | Backend | Shares across |
| --- | --- | --- |
| `redis://host:6379/0` | Redis (needs the `redis` extra) | Processes and nodes |
| `rocksdb:///var/lib/batcher/cache` | Embedded RocksDB (needs the `rocksdb` extra) | Runs on one node |

```python
# docs: skip
from batcher.config import Config, MemoryConfig, config_context

shared = Config().replace(memory=MemoryConfig(shared_cache_uri="redis://localhost:6379/0"))
with config_context(shared):
    daily = bt.read.parquet("s3://bucket/events/").group_by("region").agg(n=bt.count()).cache()
    daily.collect()  # the next process to run this reads the result instead of the table
```

Reach for RocksDB on a single machine and Redis when more than one process needs the same
entries. RocksDB locks its directory, so exactly one process may hold it open, and it says
so rather than quietly disagreeing.

### The one rule that makes this safe

A shared entry outlives the run that wrote it, which is the one way a cache can be *wrong*
rather than merely cold. So a result is written to the shared store only when **every**
input has both a durable identity and a content version. The path says which table this
is; the version token says which state of it. Rewrite the file and the key changes, so the
next run recomputes rather than serving the previous run's rows.

Three consequences follow, and they are all silent by design:

- **In-memory data never shares.** `bt.from_pydict` has no cross-run identity, so a query
  reading it uses the process cache only.
- **A source that cannot version itself declines.** Unversioned is indistinguishable from
  unchanged, and this is the boundary where that distinction has to be made.
- **A tenant or a governed viewer never shares with another.** Both are in the key, because
  a shared store makes that failure cross-process.

Declining costs a recompute, which is why nothing here guesses.

### What is too large to share

A result over 256 MB is not written to the shared store. Sharing it would serialize a
second full copy of it into the process that just computed it, then push those bytes at
the network on every run of the query, and Redis refuses a value over 512 MB in any case.
The process cache and its disk tier still hold results of any size, so a large result is
cached locally and simply not shared.

Declined writes are counted, so a workload whose results are all too large can tell that
the shared store is doing nothing for it rather than assuming it is merely cold:

```python
# docs: skip
stats = bt.cache_stats()
print(stats["shared_writes"], stats["shared_declined"])
```

### When the store is down

A shared cache degrades to recompute. Every read and write is contained: an unreachable
server, a slow one, or an entry this version cannot decode all become a miss, and the query
runs. That is deliberate, and it means an outage is invisible in the timings alone. Watch
`shared_errors`, and read `shared_hit_rate` alongside it: the rate is taken over every
lookup the store was asked for, so a store failing half its reads reports 0.5 rather than
looking healthy on the half it answered.

```python
# docs: skip
stats = bt.cache_stats()
print(stats["shared_hits"], stats["shared_misses"], stats["shared_errors"])
```

`bt.clear_cache()` empties this process's tiers and leaves the shared store alone, because
it belongs to every process reading it.

## Caching remote file bytes

The caches above hold query *results*. A separate one holds the remote *files* a query
reads. Point `memory.file_cache_dir` at a local disk and the first read of an object-store
file streams it there, so later reads of the same file come off local disk instead of
crossing the network again. It is transparent and result-invariant, since a miss just
re-fetches.

Use `"auto"` on a cluster. The right directory is a per-node fact, so naming a literal path
in a shared config names the wrong one everywhere but the machine it was written for. Each
node resolves its own fast local volume, and a node with no fast local disk gets no cache
rather than competing for the container overlay:

```python
# docs: skip
from batcher.config import Config, MemoryConfig, config_context

cached = Config().replace(memory=MemoryConfig(file_cache_dir="auto"))
with config_context(cached):
    totals = bt.read.parquet("s3://bucket/events/").group_by("region").agg(n=bt.count())
    print(totals.collect().num_rows)
```

This tier is the only one that saves network rather than compute, so it is measured
differently. `file_coalesced` counts fetches that waited on another thread rather than
downloading their own copy of the same file, which is what a scan whose workers all open
one dimension table saves. `file_declined` counts files larger than the whole budget: those
are read remotely every time, so a non-zero count with a low hit rate means the budget is
too small for a single file rather than too small for the working set.

```python
# docs: skip
stats = bt.cache_stats()
print(stats["file_hit_rate"], stats["file_coalesced"], stats["file_declined"])
```

Local paths are never cached, and `memory.file_cache_max_bytes` bounds what the directory
holds.

## When to write instead

If the result is needed by another process, or is expensive enough that you do not want
to recompute it after a crash, the answer is not a bigger cache. Write it and read it
back. Size alone is a weaker argument than it used to be, since the disk tier already
absorbs a result the memory budget cannot hold, but durability and sharing are things no
tier here provides.

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
| A result larger than `result_cache_max_bytes`, still inside one process | `cache()`, which demotes it to the disk tier |
| A result larger than `result_cache_disk_max_bytes` | write it |
| A result the same query needs on its next run, from a fresh process | a shared cache |
| A result another job or person needs, as data | write it |
| A stage you do not want to recompute after a crash | write it |
| A `map_batches` or ML pipeline result | write it, since `cache()` does not cover it |

## See also

- {doc}`Performance </user-guide/operate/tuning/performance>`: morsel sizing, spilling, and the memory budget.
- {doc}`Explain plans </user-guide/operate/tuning/explain-plans>`: confirm the plan you cached is the plan you meant.
- {doc}`Writing data </user-guide/moving-data/writing-data>`: the checkpoint alternative.
- {doc}`Query lifecycle </architecture/deep-dives/query/query-lifecycle>`: what "the plan runs twice" means,
  stage by stage.
- {doc}`Buffer pool </architecture/deep-dives/memory/buffer-pool>`: the memory the cache is yielding back
  when a running query needs it more.
- {doc}`Configuration options </configuration/options>`: `memory.result_cache_max_bytes`,
  `memory.result_cache_disk_max_bytes`, `memory.shared_cache_uri`, and the rest of the memory
  envelope.
- {doc}`On-disk artifacts </architecture/deep-dives/memory/on-disk-artifacts>`: everything the
  engine writes to local disk, the cache's second tier among it.
- {doc}`Optimizing a slow query </getting-started/tutorials/foundations/optimizing-a-slow-query>`: caching in its
  place, among the other fixes.
- {doc}`/cookbook/operations/memory_and_caching`: caching a reused branch, and spilling under a tight budget, as a script.
