# Delta Lake

Delta is the connector Batcher supports most completely: read with time travel, write as one
atomic commit, upsert with `MERGE INTO`, replace a predicate-scoped slice, compact, vacuum.
It needs `pip install 'batcher-engine[delta]'` and nothing else. No JVM, no Spark. The
implementation is delta-rs plus Batcher's own commit path.

| | |
| --- | --- |
| **Read** | `bt.read.delta(uri)`, with `version=`, `timestamp=`, or `stream=True` |
| **Write** | `ds.write.delta(uri)` with `mode="append"`/`"overwrite"`, `merge_on=`, `replace_where=` |
| **Extra** | `pip install 'batcher-engine[delta]'` |
| **Parallelism** | One split per surviving data file, chosen from the log at plan time |
| **Pushdown** | Predicates skip files by the log's per-file min/max statistics |
| **Maintenance** | `bt.compact`, `bt.vacuum` |

:::{important}
The thing that bites people is not the API. It is treating a Delta table as a directory of
Parquet files. Every data file the table has ever added is still referenced by *some* version, so
anything that rewrites or deletes files behind the log destroys time travel silently: a plain
Parquet writer pointed at the table root, an `rm` of an "old-looking" file, a compaction job that
isn't transactional. `count()` keeps answering from the log after the data is gone. Use
`bt.compact` and `bt.vacuum`, which go through the log.
:::

## Setup

Every block below runs against a fresh temp directory.

```python
import os
import tempfile

import batcher as bt

work = tempfile.mkdtemp()
table = os.path.join(work, "events")
```

## Write and read

`ds.write.delta(uri, mode=...)` is one transaction. `mode` is `"append"` (default) or
`"overwrite"`. A reader never sees a partial write, and a crash mid-write leaves nothing to clean
up: the files exist, but no commit references them.

```python
events = bt.from_pydict(
    {
        "id": [1, 2, 3],
        "amount": [10, 20, 30],
        "day": ["2024-01-01", "2024-01-01", "2024-01-02"],
    }
)
events.write.delta(table, mode="overwrite", partition_by=["day"])
bt.from_pydict({"id": [4], "amount": [40], "day": ["2024-01-02"]}).write.delta(
    table, mode="append", partition_by=["day"]
)

print(bt.read.delta(table).sort("id").to_pydict())
# {'id': [1, 2, 3, 4], 'amount': [10, 20, 30, 40],
#  'day': ['2024-01-01', '2024-01-01', '2024-01-02', '2024-01-02']}
```

`bt.read.delta(uri, version=)` or `timestamp=` reads an earlier commit. The first write above is
version `0`; passing both is an error.

```python
print(bt.read.delta(table, version=0).sort("id").to_pydict()["id"])
# [1, 2, 3]
```

## Upsert and partition replace

Two write shapes, answering two different questions.

::::{tab-set}

:::{tab-item} `merge_on=`, upsert by key

`merge_on=` runs a native Delta `MERGE INTO` keyed on those columns: matched rows update,
unmatched rows insert. A merge rewrites the files it touches, so it does read its change set back.
That stays bounded, because a change set is not a bulk load.

```python
updates = bt.from_pydict({"id": [2, 5], "amount": [999, 50], "day": ["2024-01-01", "2024-03-01"]})
updates.write.delta(table, merge_on="id")

print(bt.read.delta(table).sort("id").to_pydict()["amount"])
# [10, 999, 30, 40, 50]
```
:::

:::{tab-item} `replace_where=`, replace a slice

`replace_where=` is the other shape, and the one most backfills actually want: atomically replace
every row matching a predicate and leave the rest alone. Re-running yesterday's job replaces
yesterday and nothing else, so a retry is not a duplicate.

```python
fix = bt.from_pydict({"id": [9], "amount": [1], "day": ["2024-01-02"]})
fix.write.delta(table, replace_where=bt.col("day") == "2024-01-02", partition_by=["day"])

print(bt.read.delta(table).sort("id").to_pydict()["id"])
# [1, 2, 5, 9]
```

The two rows on `2024-01-02` are gone; `id=9` replaced them.
:::

::::

For the full `WHEN`-clause statement (`when_matched` / `when_not_matched_by_source`, SCD-2, CDC
feeds) see the [lakehouse guide](../user-guide/lakehouse.md).

## How it parallelizes

The transaction log records, for every data file, its partition values and per-column
min/max/null counts. Batcher reads that at **plan time**: one split per surviving data file, and a
file whose recorded bounds prove it cannot match is dropped before it is opened. No footer read,
no worker task.

```python
clustered = os.path.join(work, "by_day")
for day in range(4):
    rows = bt.from_pydict({"day": [day] * 3, "amount": [day * 10 + i for i in range(3)]})
    rows.write.delta(clustered, mode="append")

from batcher.io.formats.lakehouse import DeltaSource

predicate = (bt.col("day") == 2).to_ir()
source = DeltaSource(clustered)
print(len(source.splits()), "->", len(source.splits(predicate=predicate)))
# 4 -> 1
```

The write is the same mechanism run backwards. Each worker writes its shard as a final data file
and collects that file's column bounds from data it already holds; the driver commits only the add
actions. Bytes move worker to storage once, the driver's cost is one log write however much the
cluster wrote, and a distributed write is **one** version, not one per worker.

## Compaction and vacuum

Many small appends make many small files. `bt.compact` bin-packs them into one commit; the old
files leave the log but stay on storage, so every earlier version still reads.

```python
metrics = bt.compact(clustered)
print(metrics["numFilesRemoved"], "->", metrics["numFilesAdded"])
# 4 -> 1
print(bt.read.delta(clustered).count(), bt.read.delta(clustered, version=0).count())
# 12 3
```

`z_order=["col", ...]` sorts the rewritten rows along a Z-curve, which narrows each file's bounds
and is what makes the *next* query's skipping bite. `where=` scopes the work to a partition, so a
nightly job compacts today rather than the whole table.

`bt.vacuum` is the only operation that deletes. It defaults to a dry run and to a 7-day retention
window, and both defaults matter: the files it removes are exactly the ones time travel and any
in-flight reader depend on.

```python
print(len(bt.vacuum(clustered)))  # dry run: nothing is old enough
# 0
```

Pass `dry_run=False` to reclaim.

:::{important}
Shortening retention below the table's configured minimum needs an explicit
`enforce_retention_duration=False`. The moment you pass it, a reader mid-scan can have its files
deleted underneath it, and your `version=` history is gone.
:::

## Credentials

Cloud paths take their credentials from the environment (`AWS_ACCESS_KEY_ID`, `AWS_REGION`, and
friends; see [cloud storage](../user-guide/cloud-storage.md)). Delta also accepts
`storage_options=`, whose keys are delta-rs's own. That is how vended credentials (Unity Catalog, a
short-lived STS token) reach the reader and the writer.

:::{dropdown} Passing credentials explicitly instead of through the environment
```python
# docs: skip
ds = bt.read.delta(
    "s3://lake/events",
    storage_options={"aws_access_key_id": "...", "aws_secret_access_key": "..."},
)
```
:::

## Exactly-once writes from a stream

A restarted streaming query replays the micro-batch it was committing when it died. Pass an
`app_id` and a monotonic `txn_version` and the commit is recorded as a Delta application
transaction; a replay of a version already in the log commits nothing.

```python
# docs: skip
batch.write.delta("s3://lake/events", mode="append", app_id="ingest-job", txn_version=offset)
```

:::{warning}
Without those two, a replayed batch appends its rows a second time. Nothing in the format prevents
it.
:::

## Failure modes worth knowing

**Concurrent writers.** Two writers that commit conflicting versions do not corrupt the table. The
loser raises `CommitError`. Catch it and retry the write; the data files it already staged are
unreferenced, and vacuum will reclaim them.

**Deletion vectors.** A table with DVs enabled cannot be read file by file, because the data files
still hold the deleted rows. Batcher detects that from the log and reads through delta-rs's
DataFusion path, which applies the vectors. Correct, but it costs the split-parallel read and the
exact row-count-from-log. If a Delta read is unexpectedly slow and single-threaded, check
`delta.enableDeletionVectors`.

**Change data feed.** `bt.read.delta(uri, stream=True, starting_version=N)` reads the table as an
unbounded stream of commits. It needs `delta.enableChangeDataFeed = true` set on the table *before*
the commits you want to read. CDF is not retroactive.

**Schema drift.** An append whose schema does not match the table's fails at commit. Align the
columns in the plan (`select`, `cast`) rather than hoping the writer coerces.

## See also

- [Lakehouse](../user-guide/lakehouse.md): merge clauses, SCD, CDC, file skipping in depth.
- [Writing data](../user-guide/writing-data.md): modes, partitioning, the commit path.
- [Partition backfill](../examples/data-engineering/partition-backfill.md): `replace_where=`
  as a re-runnable job.
- [File compaction](../examples/data-engineering/file-compaction.md): `bt.compact` and
  Z-ordering on a schedule.
- [Exactly-once sink](../examples/streaming/exactly-once-sink.md): `app_id` and `txn_version`
  in a streaming pipeline.
- [I/O API](../api/io.md): the full reader/writer reference.
- [Iceberg](iceberg.md) and [Hudi](hudi.md): the other two table formats.
