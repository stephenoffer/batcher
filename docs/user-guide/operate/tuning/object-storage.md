# Object storage and worker locality

This page covers what changes when the data is in object storage and the query runs on more than one machine: how many reads a scan keeps in flight, what the planner and the workers cache, and how Batcher routes a partition back to the worker that already holds it.

For the levers that apply on one machine, see {doc}`performance`.

## Setup

```python
import batcher as bt
```

## Reading from object storage in parallel

A distributed scan against S3, GCS, or Azure is bound by request *latency*, not by
bandwidth. A single connection sits far below what one node can pull, and every request
waits tens of milliseconds, so what caps throughput is how many reads are outstanding at
once. Each scan task therefore keeps a bounded window of reads in flight rather than
fetching its files one after another, and yields the results in file order, so nothing
downstream can tell the reads overlapped.

The window is bounded rather than unlimited because the reads in flight are also the
task's memory: at most `BATCHER_SCAN_PREFETCH` reads (32 by default) are outstanding, so a
task holds a few files' worth of decoded batches and never its whole partition.

This matters most when a task's splits land on many different files, which is the shape the
balanced split assignment normally produces. The layout you write is what decides that
shape, and you control it directly:

```python
import tempfile

wide = bt.from_pydict({"id": list(range(2000)), "amount": [float(i) for i in range(2000)]})

many_files = tempfile.mkdtemp()
wide.write.parquet(many_files, max_rows_per_file=250)

back = bt.read.parquet(many_files)
print(back.count(), back.agg(total=bt.col("amount").sum()).to_pydict())
# 2000 {'total': [1999000.0]}
```

Very small files are the case to avoid: each one costs a request whose latency the window
can hide but not remove, and the per-file footer read is pure overhead. Aim for splits of
tens of megabytes rather than tens of kilobytes, and prefer fewer, larger files when you
control the writer.

The planner caches too, and on the driver rather than the workers. Reading a Parquet
dataset's footers to learn its row count, byte size and per-column bounds is what lets a
`count()` or a `max()` answer without touching a data page, and it costs real time on a wide
dataset: on a 100-file TPC-H `lineitem` read that aggregation is 237 ms the first time. It is
held against each file's identity, which is its path, size and modification time, so a second
query over the same files reuses it in well under a millisecond, while a file rewritten
underneath you misses and is read again. A file the filesystem cannot stat is never cached,
because there would be no way to notice it changing.

A file's schema is held the same way. Reading it means opening the file, and a strict read
opens two of them: the first file, whose schema stands for the rest, and the last, checked so
a column a later file added can't be dropped without warning. Both are kept against file
identity, so building a second `Dataset` over the same files opens nothing.

A worker also keeps the batches it decoded, so a repeated query against the same files
skips both the fetch and the decode. That cache is per worker process and bounded by
`BATCHER_SCAN_CACHE_FRACTION` of the worker's memory (0.3 by default), or set outright with
`BATCHER_SCAN_CACHE_BYTES`. It is why a second run of the same query is faster than the
first, and why a benchmark that reports only its best-of-N is measuring a warm read.

The figure puts those pieces on one read path:

![The read path of a distributed scan. On the driver, reading footers and schemas gives row counts, byte sizes and column bounds, and the result is cached per file identity, meaning path, size and modification time, so a second query over the same files reuses it. The driver hands splits to a scan task on a worker. The task first looks up the worker scan cache of decoded batches, sized at 0.3 of worker memory. On a hit, the batches go straight to the downstream operators with no fetch and no decode. On a miss, the task keeps up to 32 reads in flight, set by BATCHER_SCAN_PREFETCH, sending GET requests to object storage such as S3, GCS or Azure and receiving bytes back. The bytes are decoded and yielded in file order, so downstream operators can't tell the reads overlapped, and the decoded batches are also kept in the scan cache for the next run. Latency, not bandwidth, caps the scan, and the window keeps a task to a few files in memory.](/_static/diagrams/object_storage_read_path.svg)

## Meeting the worker that holds your cache

A per-worker cache only pays for itself if the same worker sees the same data twice, and on
a cluster that does not happen by default. A stateless task lands wherever there is room at
the moment it is scheduled, so the partition it reads is almost never the partition it read
last time. The cache fills and is never asked for what it holds.

For a `map_batches` pipeline that ends in an aggregate, which is the shape of most batch
inference work, Batcher keeps a pool of workers alive for the session and addresses them by
partition index. Partition 12 goes to the same process every run, so it meets its own
decoded batches instead of fetching them again. On TPC-H sf100 that is worth 1.4x on a
compute-heavy user function and 1.6x on a light one, measured warm.

Those workers also size themselves to the cluster. The pool is the stage's whole
parallelism, so each worker runs your function across `cluster cores / pool size` threads,
capped by the smallest node so the threads always fit the machine they land on. On a
1024-core fleet of 16-core nodes a 64-worker pool runs 16 threads each, which is the fleet
exactly once. Widening it from a fixed 4 threads to that measured a further 1.4x on a
transcendental-heavy function.

```python
import tempfile

import pyarrow as pa

readings = bt.from_pydict(
    {"user": [f"u{i % 50}" for i in range(4000)], "amount": [float(i % 97) for i in range(4000)]}
)
store = tempfile.mkdtemp()
readings.write.parquet(store, max_rows_per_file=500)

scored = bt.read.parquet(store).map_batches(
    lambda batch: batch.append_column(
        "score", pa.array([v * 1.5 for v in batch.column("amount").to_pylist()])
    ),
    output_columns=["user", "amount", "score"],
)
print(scored.agg(total=bt.col("score").sum()).to_pydict())
# {'total': [286723.5]}
```

Add `distributed=True` to that `agg` and the pool is what runs it. The workers outlive the
call, so a second `collect()` of the same pipeline reuses them.

Three behaviors matter before you rely on it.

The pool holds one pipeline at a time. A different `map_batches` function replaces it
rather than joining it, because these workers hold general-purpose cores that every other
stage also wants, and a session that alternates between two pipelines pays a rebuild each
time rather than reserving the cluster twice over.

The pool gives its cores back when the session goes idle, after
`distributed.session_fleet_idle_s` of no use, which is 30 seconds by default and the same
setting the shuffle fleet's own idle release reads. A CPU pool this size is most of a
cluster, and it earns those cores from the scan cache of the query that filled it, so a
query that is not coming keeps earning nothing. Back-to-back queries never wait for a
rebuild, because taking the pool cancels the pending release, and a query that runs longer
than the window keeps its own workers. Measured on a 65-node, 1,024-core cluster: a finished
`map_batches` and aggregate held 960 cores at 25 seconds after the query returned and 0 at
33 seconds.

A GPU pool gives its devices back on its own window, `distributed.warm_inference_idle_s`,
120 seconds by default. It is longer because a model load costs far more than an actor
respawn, and it is bounded because a reserved idle GPU stops another tenant dead where a
reserved idle core only slows one down. Set it to `0` for whole-session residency.

The whole behavior is under `distributed.warm_inference_pools`, on by default. Turn it off
and every stage runs on stateless tasks released as soon as they finish. To hand the cluster
to something else immediately, rather than waiting out the idle window, call
{py:func}`bt.release_cluster() <batcher.release_cluster>`, which releases the warm shuffle fleet
and the inference pools together. It's a no-op when nothing is warm and never raises, so it's
safe at the end of any script:

```python
from batcher.config import Config

cfg = Config()
print(cfg.distributed.warm_inference_pools, cfg.distributed.session_fleet_idle_s)
# True 30.0
print(cfg.distributed.warm_inference_idle_s)
# 120.0

bt.release_cluster()
```

## See also

- {doc}`performance`: morsels, the memory envelope, spilling, and the adaptive loop.
- {doc}`caching`: caching a *result* rather than the bytes a scan read.
- {doc}`large-tables`: what changes once planning a table costs more than reading it.
- {doc}`/user-guide/moving-data/cloud-storage`: paths, credentials, and filesystems for S3, GCS, and Azure.
- {doc}`/configuration/distributed-options`: every `distributed.*` setting named here.
