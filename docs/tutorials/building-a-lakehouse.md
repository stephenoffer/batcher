# Building a lakehouse

Build the three medallion layers on a real Delta table: raw files in, a transactional
curated table in the middle, aggregates out. Along the way you get atomic commits, upserts,
time travel, an idempotent backfill, and file skipping: the things that separate a lakehouse
from a directory of Parquet.

Everything here runs as written, in a temp directory, with the `delta` extra
(`pip install 'batcher-engine[delta]'`).

:::{note}
**What you'll build.** Three medallion layers on a real Delta table in a temp directory:
bronze Parquet, a transactional silver table with an upsert and time travel, and a gold
aggregate. Then file skipping, which is the mechanism that makes the whole thing fast.
:::

| You need | For |
|---|---|
| `pip install 'batcher-engine[delta]'` | Every runnable block on this page |
| A temp directory | Provided by the first block |
| A cluster | Only the last block, which is shown and not run |

## 1. Somewhere to work

```python
import os
import tempfile

import batcher as bt

work = tempfile.mkdtemp()
```

## 2. Bronze: land the raw data

Bronze is the raw drop. No cleaning, no dedup, no opinions. Get it stored so nothing is lost.
Parquet files are fine here.

```python
raw = os.path.join(work, "bronze")
bt.from_pydict(
    {
        "order_id": [1, 2, 3, 4],
        "customer": ["ann", "bo", "ann", "cy"],
        "day": ["2024-03-01", "2024-03-01", "2024-03-02", "2024-03-02"],
        "amount": [120.0, 40.0, 80.0, 15.0],
        "status": ["paid", "paid", "refunded", "paid"],
    }
).write.parquet(raw)

print(bt.read.parquet(raw).count())
# 4
```

## 3. Silver: a transactional curated table

Silver is where the table becomes trustworthy: filtered, typed, deduplicated, and
transactional. `ds.write.delta(uri, mode="overwrite")` commits the whole dataset as one
Delta transaction, so a reader either sees the old table or the new one, never half of
either.

```python
orders = os.path.join(work, "orders")

(
    bt.read.parquet(raw)
    .filter(bt.col("status") == "paid")
    .select("order_id", "customer", "day", "amount")
    .write.delta(orders, mode="overwrite", partition_by=["day"])
)

print(bt.read.delta(orders).sort("order_id").to_pydict())
# {'order_id': [1, 2, 4], 'customer': ['ann', 'bo', 'cy'], 'day': ['2024-03-01', '2024-03-01', '2024-03-02'], 'amount': [120.0, 40.0, 15.0]}
```

The refunded order is gone. Three rows survive.

## 4. The upsert

Corrections arrive after the fact: order 2 was re-priced, and order 5 showed up late.
`merge_on=` runs a native Delta `MERGE INTO` keyed on those columns: matched rows update,
unmatched rows insert. This is one commit, not a read-modify-write race.

```python
updates = bt.from_pydict(
    {
        "order_id": [2, 5],
        "customer": ["bo", "dee"],
        "day": ["2024-03-01", "2024-03-03"],
        "amount": [45.0, 60.0],
    }
)
updates.write.delta(orders, merge_on="order_id")

print(bt.read.delta(orders).sort("order_id").to_pydict())
# {'order_id': [1, 2, 4, 5], 'customer': ['ann', 'bo', 'cy', 'dee'], 'day': ['2024-03-01', '2024-03-01', '2024-03-02', '2024-03-03'], 'amount': [120.0, 45.0, 15.0, 60.0]}
```

Order 2 is now 45.0 and order 5 exists. Nothing else moved.

## 5. Time travel

Every commit is a version. Version 0 is the table as `overwrite` left it, before the merge.
Pass `version=` (or `timestamp=`) to read it.

```python
print(bt.read.delta(orders, version=0).sort("order_id").to_pydict()["amount"])
# [120.0, 40.0, 15.0]
```

That is the audit trail, and it is also the fastest way to answer "what changed?" after a
bad load.

## 6. Gold: the aggregate anyone can query

Gold is the layer the dashboard reads. It is a plain query over silver, with no special
machinery, because the curated table already guarantees what it is.

```python
daily = (
    bt.read.delta(orders)
    .group_by("day")
    .agg(revenue=bt.col("amount").sum(), orders=bt.count())
    .sort("day")
)
print(daily.to_pydict())
# {'day': ['2024-03-01', '2024-03-02', '2024-03-03'], 'revenue': [165.0, 15.0, 60.0], 'orders': [2, 1, 1]}
```

## 7. The idempotent backfill

A backfill re-runs a day. If it appends, you get double-counted revenue; if it overwrites the
table, you lose every other day. `replace_where=` is the answer: atomically replace exactly
the rows matching a predicate and leave the rest alone. Re-running it is a no-op.

| The write you reach for | What it does | When it is right |
|---|---|---|
| `mode="append"` | Adds rows | New data only, never a correction |
| `mode="overwrite"` | Replaces the table | A full rebuild |
| `merge_on="key"` | `MERGE INTO`: matched rows update, unmatched insert | Corrections and late arrivals keyed by an id |
| `replace_where=pred` | Atomically replaces exactly the matching rows | Re-running one partition of a backfill |

:::{important}
A backfill written as an `append` is a data-correctness bug, not a performance one. Run it
twice and the revenue doubles, silently, and the number reaches a dashboard before anyone
notices. `replace_where=` is the one that survives being run twice, which is the only
property that matters when the job is retried by a scheduler you do not control.
:::

```python
fixed = bt.from_pydict(
    {
        "order_id": [4],
        "customer": ["cy"],
        "day": ["2024-03-02"],
        "amount": [95.0],
    }
)
fixed.write.delta(orders, replace_where=bt.col("day") == "2024-03-02", partition_by=["day"])

print(bt.read.delta(orders).sort("order_id").to_pydict()["amount"])
# [120.0, 45.0, 95.0, 60.0]
```

Order 4 is the `2024-03-02` row, and it is now the corrected 95.0. The other days are
untouched, and running that block again would produce the same table.

## 8. File skipping, which is why any of this is fast

A lakehouse's transaction log already records, for every data file, its partition values and
its per-column min/max. That is a zone map over the *file* dimension, and Batcher reads it at
**plan time**. A file whose bounds prove it cannot hold a matching row is never opened,
never split, never shipped to a worker.

Write a table clustered by day, so each append lands its own file, and watch the predicate
cut the file list rather than just the rows:

```python
from batcher.io.formats.lakehouse import DeltaSource

by_day = os.path.join(work, "by_day")
for day in ["2024-03-01", "2024-03-02", "2024-03-03", "2024-03-04"]:
    bt.from_pydict({"day": [day] * 3, "amount": [1.0, 2.0, 3.0]}).write.delta(
        by_day, mode="append"
    )

source = DeltaSource(by_day)
predicate = (bt.col("day") == "2024-03-03").to_ir()
print(len(source.splits()), "->", len(source.splits(predicate=predicate)))
# 4 -> 1
```

Four files in the table, one file read. The pruning happens *before* I/O, so the eliminated
files cost nothing at all: no footer read, no task, no bytes.

```python
print(bt.read.delta(by_day).filter(bt.col("day") == "2024-03-03").count())
# 3
```

:::{tip}
Pruning is deliberately one-sided. A file is dropped only when the log *proves* it cannot
match; a missing statistic keeps the file. Skipping can cost you extra I/O. It cannot cost
you a row. That asymmetry is why you can trust it without checking it.
:::

On a 10M-row, 200-file Delta table, that mechanism takes a `count(*) WHERE day = 42` from
98.8 ms reading 200 files to 13.4 ms reading one, past DuckDB's `delta_scan` at 19.0 ms.
See [vs DuckDB](../benchmarks/vs-duckdb.md).

## 9. Do it on a cluster

Nothing above changes. The same plan runs distributed: workers write their shards as final
data files and record each file's bounds while the data is still in memory, and the driver
commits only the *add actions*: paths, sizes, statistics. The bytes move once, worker to
storage, and the driver never re-encodes the result.

::::{tab-set}
:::{tab-item} Local
```python
# docs: skip
import batcher as bt

(
    bt.read.parquet("bronze/")
    .filter(bt.col("status") == "paid")
    .write.delta("silver/orders", mode="overwrite")
)
```
:::

:::{tab-item} Cluster
```python
# docs: skip
import batcher as bt

(
    bt.read.parquet("s3://lake/bronze/")
    .filter(bt.col("status") == "paid")
    .write.delta("s3://lake/silver/orders", mode="overwrite", distributed=True)
)
```
:::
::::

The query is the same one. `distributed=True` and a bucket are the entire difference.

A distributed write is **one** transaction. Workers produce files; the driver commits once.
The commit is `O(files)`, not `O(rows)`, which is what took the 16-shard, 240 MB commit in
the benchmark from 634 ms to 4.9 ms.

## What to take away

- `mode=` for whole-table writes, `merge_on=` for key upserts, `replace_where=` for
  idempotent partition backfills. Reach for the third one more often than you expect.
- Time travel is free; it is what the log already records.
- The statistics a write leaves behind are exactly what the next read prunes against. The
  two halves are one mechanism.

## What you learned

::::{grid} 1 3 3 3
:gutter: 3

:::{grid-item-card} {octicon}`database;1.1em` Lakehouse guide
:link: ../user-guide/lakehouse
:link-type: doc
Iceberg, Hudi, SCD types, CDC feeds.
:::

:::{grid-item-card} {octicon}`broadcast;1.1em` A streaming pipeline
:link: streaming-pipeline
:link-type: doc
Make the bronze layer continuous.
:::

:::{grid-item-card} {octicon}`check;1.1em` Data quality
:link: ../user-guide/data-quality
:link-type: doc
Validate and quarantine before you commit.
:::
::::

## See also

- [Delta Lake integration](../integrations/delta-lake.md) and
  [Iceberg](../integrations/iceberg.md): the connectors underneath.
- [Writing data](../user-guide/writing-data.md): every write mode, in one place.
- [vs DuckDB](../benchmarks/vs-duckdb.md): the file-skipping measurement, and the optimizer
  bug that used to break it.
- [Partition backfill](../examples/data-engineering/partition-backfill.md) and
  [slowly changing dimensions](../examples/data-engineering/slowly-changing-dimensions.md):
  the recipes step 7 generalizes to.
- [Governance](../user-guide/governance.md): row filters and column masks on the curated
  table.
