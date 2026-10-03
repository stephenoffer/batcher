# Building a lakehouse

Build the three medallion layers on a real Delta table: raw files in, a transactional curated table in the middle, aggregates out. Along the way you get atomic commits, upserts, time travel, an idempotent backfill, and file skipping. Everything runs in a temp directory with the `delta` extra (`pip install 'batcher-engine[delta]'`).

| You need | For |
|---|---|
| `pip install 'batcher-engine[delta]'` | Every runnable block on this page |
| A temp directory | Provided by the first block |
| A cluster | Only the last block, which is shown and not run |

The following diagram shows the tables this page builds and the writes that change silver after its first load:

![Bronze holds raw Parquet with no cleaning. An overwrite of its paid rows creates silver, a Delta table partitioned by day. Gold is a plain group_by over silver that sums revenue per day. After the first load, merge_on="order_id" upserts into silver as one commit, and replace_where with a predicate replaces one day's rows as a backfill that is safe to re-run. Reading version=0 returns silver as it was before the merge, because every commit is a version. The transaction log also records each file's bounds, which a filtered read prunes against.](/_static/diagrams/medallion_layers.svg)

## 1. Somewhere to work

Every block below writes into one temp directory.

```python
import os
import tempfile

import batcher as bt

work = tempfile.mkdtemp()
```

## 2. Bronze: land the raw data

Bronze is the raw drop, stored as-is in Parquet.

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

Silver is filtered, typed, and transactional. {py:meth}`ds.write.delta(uri, mode="overwrite") <batcher.api.io_namespace.writer.Writer.delta>` commits the whole dataset as one Delta transaction, so a reader sees the old table or the new one.

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

Order 2 was re-priced and order 5 arrived late. `merge_on=` runs a native Delta `MERGE INTO` as one commit: matched rows update, unmatched rows insert.

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

Every commit is a version. Pass `version=` (or `timestamp=`) to read an earlier one:

```python
print(bt.read.delta(orders, version=0).sort("order_id").to_pydict()["amount"])
# [120.0, 40.0, 15.0]
```

Diff two versions to see exactly what a load changed:

```python
before = set(bt.read.delta(orders, version=0).to_pydict()["order_id"])
after = set(bt.read.delta(orders).to_pydict()["order_id"])
print(sorted(after - before))
# [5]
```

## 6. Gold: the aggregate anyone can query

Gold is the layer the dashboard reads, a plain query over silver:

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

A backfill re-runs a day. `replace_where=` atomically replaces exactly the rows matching a predicate and leaves the rest alone, so re-running it produces the same table. The four write modes cover every case:

| The write you reach for | What it does | When it is right |
|---|---|---|
| `mode="append"` | Adds rows | New data only, never a correction |
| `mode="overwrite"` | Replaces the table | A full rebuild |
| `merge_on="key"` | `MERGE INTO`: matched rows update, unmatched insert | Corrections and late arrivals keyed by an id |
| `replace_where=pred` | Atomically replaces exactly the matching rows | Re-running one partition of a backfill |

:::{tip}
Use `replace_where=` for any backfill a scheduler might retry. It is safe to run twice.
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

Order 4 is now the corrected 95.0, and the other days are untouched. Run it again and nothing changes:

```python
fixed.write.delta(orders, replace_where=bt.col("day") == "2024-03-02", partition_by=["day"])
print(bt.read.delta(orders).count())
# 4
```

## 8. File skipping

The transaction log records every file's partition values and per-column min/max, and Batcher reads them at **plan time**. A file whose bounds rule out a match is never opened. Write one file per day and watch a predicate cut the file list:

```python
from batcher.io.formats.lakehouse import DeltaSource

by_day = os.path.join(work, "by_day")
for day in ["2024-03-01", "2024-03-02", "2024-03-03", "2024-03-04"]:
    bt.from_pydict({"day": [day] * 3, "amount": [1.0, 2.0, 3.0]}).write.delta(by_day, mode="append")

source = DeltaSource(by_day)
predicate = (bt.col("day") == "2024-03-03").to_ir()
print(len(source.splits()), "->", len(source.splits(predicate=predicate)))
# 4 -> 1
```

Four files in the table, one file read, with no footer read or task for the other three.

```python
print(bt.read.delta(by_day).filter(bt.col("day") == "2024-03-03").count())
# 3
```

A file is dropped only when the log proves it cannot match, so skipping never costs a row. On a 200-file table it takes a `count(*) WHERE day = 42` from 98.8 ms to 7.4 ms, ahead of DuckDB's `delta_scan` at 21.8 ms ({doc}`vs DuckDB </benchmarks/comparisons/vs-duckdb>`).

## 9. Do it on a cluster

The same plan runs distributed. Workers write final data files and their bounds, and the driver commits only the *add actions*: paths, sizes, statistics.

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

`distributed=True` and a bucket are the entire difference. The write is still **one** transaction, and the commit is `O(files)`, not `O(rows)`: 16 shards totalling 240 MB commit in 4.1 ms with no data passing through the driver ([`benchmarks/BENCHMARK_RESULTS.md`](https://github.com/stephenoffer/batcher/blob/main/benchmarks/BENCHMARK_RESULTS.md)).

## Where to go next

Three write modes cover almost everything: `mode=` for a whole table, `merge_on=` for a keyed upsert, `replace_where=` for an idempotent backfill. The statistics each write leaves behind are what the next read prunes against.

::::{grid} 1 3 3 3
:gutter: 3

:::{grid-item-card} {octicon}`database;1.1em` Lakehouse guide
:link: /user-guide/moving-data/lakehouse
:link-type: doc
Iceberg, Hudi, SCD types, CDC feeds.
:::

:::{grid-item-card} {octicon}`broadcast;1.1em` A streaming pipeline
:link: /getting-started/tutorials/pipelines/streaming-pipeline
:link-type: doc
Make the bronze layer continuous.
:::

:::{grid-item-card} {octicon}`check;1.1em` Data quality
:link: /user-guide/trust/data-quality
:link-type: doc
Validate and quarantine before you commit.
:::
::::

## See also

- {doc}`Delta Lake integration </integrations/lakehouse/delta-lake>` and
  {doc}`Iceberg </integrations/lakehouse/iceberg>`: the connectors underneath.
- {doc}`Writing data </user-guide/moving-data/writing-data>`: every write mode, in one place.
- {doc}`vs DuckDB </benchmarks/comparisons/vs-duckdb>`: the file-skipping measurement.
- {doc}`Partition backfill </cookbook/data-engineering/maintenance/partition-backfill>` and
  {doc}`slowly changing dimensions </cookbook/data-engineering/modeling/slowly-changing-dimensions>`:
  the recipes step 7 generalizes to.
- {doc}`Governance </user-guide/trust/governance>`: row filters and column masks on the curated
  table.
