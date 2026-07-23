# Incremental ingest

An upstream system drops Parquet files into `landing/` every few minutes. Your job
reads the directory and loads it into the warehouse. The naive version:

```python
# docs: skip
bt.read.parquet("s3://bucket/landing/").write.parquet("warehouse/orders")
```

:::{warning}
Run that every five minutes and you re-read every file ever dropped. On day one it takes
two seconds. On day ninety it takes twenty minutes, and the cost is entirely work you
already did.
:::

The usual patch is a hand-rolled bookmark: a JSON file listing processed paths, or a
`WHERE mtime > last_run` filter over the listing. Both drift. Clocks skew, a writer
touches a file after you read it, the bookmark file gets clobbered by two overlapping
runs, and you find out because a day of orders is missing.

| Discovery strategy | What it trusts | How it fails |
|---|---|---|
| Re-read the whole directory | nothing | cost grows with the table, forever |
| `mtime > last_run` | the writer's clock | skew, and a file touched after you read it |
| A JSON bookmark of processed paths | your own bookkeeping | two overlapping runs clobber it |
| `files_incremental(state_dir=...)` | a durable set of paths already read | one writer per state directory |

## The seen-file store

`bt.read.files_incremental(path, format, state_dir=...)` keeps that bookkeeping for
you. Each read is one discovery pass: it lists `path`, subtracts the files already
recorded in `state_dir`, reads only what is left, and records them. The store is a
SQLite file (no service, no extra dependency), so it survives a process restart.

```python
import os
import tempfile

import pyarrow as pa

import batcher as bt

work = tempfile.mkdtemp()
landing = os.path.join(work, "landing")
state = os.path.join(work, "_state")
os.makedirs(landing)


def drop_file(name, ids):
    """Stand in for the upstream producer writing a new file into the landing zone."""
    rows = bt.from_pydict({"order_id": ids, "amount": [10 * i for i in ids]})
    rows.write.parquet(os.path.join(landing, name))


drop_file("2024-01-01T00.parquet", [1, 2])
drop_file("2024-01-01T01.parquet", [2, 3])
```

The source is unbounded (the directory keeps growing), so it will not `collect()`.
Consume it with `iter_batches()`:

```python
def arrivals():
    """One discovery pass: the batches of files not seen on any previous pass."""
    source = bt.read.files_incremental(landing, "parquet", state_dir=state)
    batches = list(source.iter_batches())
    return pa.Table.from_batches(batches) if batches else None


first = arrivals()
print(first.to_pydict())
# {'order_id': [1, 2, 2, 3], 'amount': [10, 20, 20, 30]}
```

Call it again with nothing new on disk and you get nothing back. This is the property
the bookmark file was supposed to have:

```python
print(arrivals())
# None
```

Drop another file and only that file is read:

```python
drop_file("2024-01-01T02.parquet", [4])
print(arrivals().to_pydict())
# {'order_id': [4], 'amount': [40]}
```

## Landing it idempotently

:::{important}
Discovery being exactly-once does not make the *load* exactly-once. Your process can die
after the discovery pass marks a file seen and before the warehouse write commits. That
file is now invisible to the next run, and the rows in it are gone from the warehouse
with nothing to tell you so.
:::

Two things fix it. Make the write a keyed upsert rather than an append, so replaying a
batch cannot double-count. And put the discovery pass and the write in the same
function, so a crash between them costs you one batch of work, not a day of data.

```python
warehouse = os.path.join(work, "orders.parquet")


def ingest():
    batches = list(bt.read.files_incremental(landing, "parquet", state_dir=state).iter_batches())
    if not batches:
        return 0
    arrived = bt.from_arrow(pa.Table.from_batches(batches))
    # The same order can appear in two dropped files; keep the newest.
    latest = arrived.distinct(subset=["order_id"], keep="last", order_by="amount")
    latest.write.merge(warehouse, on="order_id")
    return latest.count()


drop_file("2024-01-01T03.parquet", [5])
print(ingest())
# 1
print(bt.read.parquet(warehouse).sort("order_id").to_pydict())
# {'order_id': [5], 'amount': [50]}
```

A second call finds no new files, so it is a no-op. The load has converged.

```python
print(ingest())
# 0
```

:::{tip}
If exactly-once matters more than a little duplicate work, invert the order: write first,
mark seen second. You then get at-least-once discovery, and the keyed merge makes the
repeat harmless. That is the trade worth taking almost every time.
:::

## What it does not do

Discovery is per *file*, keyed by path. A producer that rewrites `part-0.parquet` in
place with new contents will not be re-read: the path is already marked. If your
upstream mutates files, this pattern is the wrong one. Read a change feed instead (see
[CDC pipeline](cdc-pipeline.md)).

`state_dir` is single-writer. Two ingest jobs pointed at one landing zone and one state
directory will race on the SQLite store. Give each consumer its own `state_dir`, and they
are cheap, and separate consumers of the same directory is exactly what separate stores
are for.

The listing has a lexical fast path: files whose names sort after the greatest name
seen so far are the only candidates. Name your files with a timestamp or a monotonic
sequence (`2024-01-01T03.parquet`, not `orders-a3f9.parquet`) and a landing zone with a
million files still lists cheaply.

## In production

Cloud paths work unchanged (`resolve_filesystem` handles the scheme), and the state
directory can live next to the data. The only thing that changes between a laptop and a
lake is the sink you land the arrivals in.

::::{tab-set}

:::{tab-item} Cloud landing zone, Delta target

```python
# docs: skip
source = bt.read.files_incremental(
    "s3://bucket/landing/",
    "json",
    state_dir="s3://bucket/_batcher_state/orders",
)
for batch in source.iter_batches():
    bt.from_arrow(batch).write.delta("s3://lake/orders", merge_on="order_id")
```

The Delta commit is transactional, so a reader never sees a half-applied batch.
:::

:::{tab-item} Local landing zone, Parquet target

```python
# docs: skip
source = bt.read.files_incremental(landing, "parquet", state_dir=state)
for batch in source.iter_batches():
    bt.from_arrow(batch).write.merge(warehouse, on="order_id")
```

The Parquet merge is copy-on-write and single-writer: it rewrites the files the keys
reach and swaps them in.
:::

::::

For a large one-shot load rather than a trickle, `resume=True` on the write is the
other half of this: it skips output shards already committed, so a job killed by a spot
preemption finishes the parts it did not write instead of starting over.

## See also

- [Deduplication](deduplication.md): the incoming batch has duplicates in it.
- [File compaction](file-compaction.md): a thousand small arrivals make a thousand
  small files.
- [Quality gates](quality-gates.md): what to check before the arrivals reach the table.
- [Reading data](../../user-guide/reading-data.md): the full reader surface.
- [Writing data](../../user-guide/writing-data.md): `merge`, `resume`, and the sink options.
- [Delta Lake](../../integrations/delta-lake.md): the transactional target and its commits.
- [IO API reference](../../api/io.md): `bt.read` and `ds.write` in full.
