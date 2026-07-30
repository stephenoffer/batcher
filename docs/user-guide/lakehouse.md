# Lakehouse table formats

Batcher reads and writes the transactional table formats that back a lakehouse:
Delta Lake, Apache Iceberg, Apache Hudi. It also covers the maintenance patterns
built on top of them, `MERGE` upserts and partition backfills and
slowly-changing-dimension history. A Delta write is a single atomic commit, so a
reader never sees a partial table, and time travel lets you query any past version.
The mergeable engine runs all of this on one core or across a cluster, with an
identical result.

## Setup

The Delta examples below need the optional extra (`pip install 'batcher-engine[delta]'`),
already installed in this environment. Each block writes into a fresh temp directory
so the page is self-contained.

```python
import os
import tempfile

import batcher as bt

work = tempfile.mkdtemp()
```

## Write a Delta table

`ds.write.delta(uri, mode=...)` commits the dataset to a Delta table. `mode` is
`"overwrite"` (replace the table) or `"append"` (add a new version). Each call is
one transaction, so the table is always readable.

```python
events = bt.from_pydict({"id": [1, 2, 3], "amount": [10, 20, 30]})
table_uri = os.path.join(work, "events")
events.write.delta(table_uri, mode="overwrite")

bt.from_pydict({"id": [4], "amount": [40]}).write.delta(table_uri, mode="append")

print(bt.read.delta(table_uri).sort("id").to_pydict())
# {'id': [1, 2, 3, 4], 'amount': [10, 20, 30, 40]}
```

## Read and time-travel

`bt.read.delta(uri)` reads the latest version. Pass `version=` (or `timestamp=`) to
read the table as it was at an earlier commit. The first write above is version `0`.

```python
print(bt.read.delta(table_uri, version=0).sort("id").to_pydict())
# {'id': [1, 2, 3], 'amount': [10, 20, 30]}
```

## Merge (upsert)

`ds.write.delta(uri, merge_on=...)` runs a native Delta `MERGE INTO` keyed on the
given columns: matched rows are updated and unmatched rows inserted. Below, `id=2`
is updated in place and `id=5` is inserted.

```python
updates = bt.from_pydict({"id": [2, 5], "amount": [999, 50]})
updates.write.delta(table_uri, merge_on="id")

print(bt.read.delta(table_uri).sort("id").to_pydict())
# {'id': [1, 2, 3, 4, 5], 'amount': [10, 999, 30, 40, 50]}
```

Use `merge_on` for key-matched upserts. To replace a known slice of a table
wholesale (a backfill, an idempotent reload), use `replace_where` instead.

### The general form: `merge_into`

`merge_on` is the two-clause shorthand: update what matched, insert the rest. When you
need the whole statement, `ds.write.merge_into(target, on=...)` opens an ordered list of
`WHEN` clauses, each with its own condition and its own set of columns to write.

Inside a clause, refer to the two sides explicitly with `source_col` and `target_col`:
`source_col("x")` reads the incoming row, `target_col("x")` reads the row already in the
table. That is what lets a clause compare the two, so you can skip a write when nothing
actually changed, or accumulate instead of overwrite.

```python
# docs: skip
from batcher import source_col, target_col

(
    updates.write.merge_into(table_uri, on="id")
    # only touch the row if the amount really moved
    .when_matched(condition=source_col("amount") != target_col("amount"))
    .update(amount=source_col("amount"))
    .when_not_matched()
    .insert(id=source_col("id"), amount=source_col("amount"))
    .execute()
)
```

The clause `when_not_matched_by_source` acts on target rows the change set never
mentioned, which is how a snapshot load expires departed rows and how an SCD-2 load closes
out a version. It has no source row, so `source_col` is an error inside it (and
`target_col` is an error inside an insert clause, which has no target row yet).

## Partition backfill with replace_where

`replace_where=<predicate>` is a dynamic partition/range overwrite (Delta's
`replaceWhere`): atomically replace only the rows matching the predicate and keep
the rest. Being predicate-scoped rather than key-matched is exactly what the
idempotent backfill wants. Re-running a day's job replaces that day's rows and
nothing else.

```python
sales = bt.from_pydict(
    {
        "day": ["2024-01-01", "2024-01-01", "2024-01-02"],
        "region": ["us", "eu", "us"],
        "amount": [10, 20, 30],
    }
)
sales_path = os.path.join(work, "sales.parquet")
sales.write.parquet(sales_path)

backfill = bt.from_pydict({"day": ["2024-01-01"], "region": ["us"], "amount": [999]})
backfill.write.parquet(sales_path, replace_where=bt.col("day") == "2024-01-01")

print(bt.read.parquet(sales_path).sort("day", "region").to_pydict())
# {'day': ['2024-01-01', '2024-01-02'], 'region': ['us', 'us'], 'amount': [999, 30]}
```

The two `2024-01-01` rows are replaced by the single backfilled row, and the
`2024-01-02` row is untouched. `replace_where` works on any file target, Parquet
here, and on Delta tables.

## Slowly changing dimensions

The `ds.scd` accessor maintains dimension tables from an incoming snapshot. The
dataset is the new snapshot, meaning natural keys plus attributes, and the method writes
the reconciled dimension to `target`. These compose existing operators, with no special
engine path, and they write to any file target, Parquet here, or to a Delta table.

### Type 1: overwrite in place

Type 1 keeps no history: a matched key's attributes are overwritten (a keyed
upsert). Here `id=2` moves from `LA` to `SF`.

```python
t1 = os.path.join(work, "city_t1.parquet")
bt.from_pydict({"id": [1, 2], "city": ["NYC", "LA"]}).write.parquet(t1)
bt.from_pydict({"id": [2], "city": ["SF"]}).scd.type1(t1, keys="id")

print(bt.read.parquet(t1).sort("id").to_pydict())
# {'id': [1, 2], 'city': ['NYC', 'SF']}
```

### Type 2: full history

Type 2 keeps every version with effective-dating columns. When a tracked attribute
changes, the current row is expired (`valid_to = as_of`, `is_current = False`) and a
new version is appended (`valid_from = as_of`, `is_current = True`). `as_of` is the
effective timestamp of the batch.

```python
dim = os.path.join(work, "customer_dim.parquet")
bt.from_pydict({"id": [1, 2], "city": ["NYC", "LA"]}).scd.type2(
    dim, keys="id", track=["city"], as_of="2024-01-01"
)
bt.from_pydict({"id": [1, 2], "city": ["SF", "LA"]}).scd.type2(
    dim, keys="id", track=["city"], as_of="2024-06-01"
)

history = bt.read.parquet(dim).sort("id", "valid_from")
print(history.select("id", "city", "valid_from", "is_current").to_pydict())
# {'id': [1, 1, 2], 'city': ['NYC', 'SF', 'LA'],
#  'valid_from': ['2024-01-01', '2024-06-01', '2024-01-01'],
#  'is_current': [False, True, True]}
```

`id=1` now has two versions, the expired `NYC` and the current `SF`, while the
unchanged `id=2` keeps its single open version.

### Type 3: previous value

Type 3 keeps only the immediately previous value of each tracked attribute in a
`<attr>_prev` column.

```python
t3 = os.path.join(work, "city_t3.parquet")
bt.from_pydict({"id": [1], "city": ["NYC"]}).scd.type3(t3, keys="id", track=["city"])
bt.from_pydict({"id": [1], "city": ["LA"]}).scd.type3(t3, keys="id", track=["city"])

print(bt.read.parquet(t3).sort("id").to_pydict())
# {'id': [1], 'city': ['LA'], 'city_prev': ['NYC']}
```

## Change data capture

`type1`/`type2`/`type3` take a clean snapshot of the dimension as it is *now*.
A change-data-capture connector (Debezium, a Delta change feed, a Snowflake stream)
does not give you that. It gives you the stream of what *happened*: inserts, updates
and deletes, delivered more than once and out of order.

`ds.scd.apply_changes` consumes that feed directly (Delta Live Tables spells the same
thing `APPLY CHANGES INTO ... STORED AS SCD TYPE 1`):

```python
cdc = os.path.join(work, "customers_cdc.parquet")

feed = bt.from_pydict(
    {
        "id": [1, 2, 1],
        "city": ["NYC", "LA", "SF"],
        "op": ["INSERT", "INSERT", "UPDATE"],
        "seq": [1, 2, 3],
    }
)
feed.scd.apply_changes(
    cdc,
    keys="id",
    sequence_by="seq",
    deletes=bt.col("op") == "DELETE",
    columns=["id", "city"],  # `op` drives the delete predicate; it is not stored
)

print(bt.read.parquet(cdc).sort("id").select("id", "city").to_pydict())
# {'id': [1, 2], 'city': ['SF', 'LA']}
```

Three rules make this safe against a feed you do not control:

* Within a batch, only the greatest-`sequence_by` change per key survives, so
  redeliveries and out-of-order rows collapse to the latest.
* `sequence_by` is **stored in the target**, so a *later* batch can recognize a change
  older than what already landed and discard it, rather than resurrecting old data.
* A row matching `deletes` removes the target row. A delete for an absent key is a
  tombstone and changes nothing.

```python
later = bt.from_pydict(
    {"id": [2, 1], "city": ["LA", "OLD"], "op": ["DELETE", "UPDATE"], "seq": [4, 0]}
)
later.scd.apply_changes(
    cdc, keys="id", sequence_by="seq", deletes=bt.col("op") == "DELETE", columns=["id", "city"]
)

print(bt.read.parquet(cdc).select("id", "city").to_pydict())
# {'id': [1], 'city': ['SF']}
```

`id=2` was deleted. The stale `seq=0` update for `id=1` was discarded because the target
already holds `seq=3`.

Re-applying a batch is therefore a no-op. Apply batches in non-decreasing sequence
order, which is what a CDC reader produces, and the target converges on the source's
state.

:::{warning}
The apply is idempotent but not commutative. A delete is physical, not a tombstone, so a
deleted key stores no sequence to compare against: replaying an *old insert* for a key
that was since deleted will resurrect it. Feed batches in sequence order, and treat a full
replay of a feed containing deletes as a rebuild, not a resume. Delta Live Tables' SCD
type 1 has the same shape and the same caveat.
:::

## Iceberg and Hudi

Iceberg uses the same `read`/`write` surface, addressed by catalog identifier with
`snapshot_id=` time travel. It needs `pip install 'batcher-engine[iceberg]'` and a
configured catalog, so the block below is illustrative.

```python
# docs: skip
import batcher as bt

orders = bt.from_pydict({"id": [1, 2], "amount": [10, 20]})
orders.write.iceberg("db.orders", mode="append")

# Time-travel a snapshot, resolving against a named catalog.
snapshot = bt.read.iceberg("db.orders", catalog="prod", snapshot_id=1234567890)
```

Hudi is supported read-only, via `pip install 'batcher-engine[hudi]'`. Writes require
the Spark/Flink write stack.

```python
# docs: skip
import batcher as bt

events = bt.read.hudi("s3://lake/hudi/events")
```

## File skipping: the log answers the query before the data does

A lakehouse table's transaction log already records, for every data file, its partition
values and its per-column min/max/null-count. That is a zone map over the *file*
dimension, and Batcher reads it at **plan time**: a file whose recorded bounds prove it
cannot hold a matching row is dropped before it is opened, split, or shipped to a worker.

So on a table clustered by day, a selective filter touches one file, not all of them.
Each append below lands its own data file, and the log records that file's `day` range:

```python
clustered = os.path.join(work, "by_day")
for day in range(4):
    rows = bt.from_pydict({"day": [day] * 3, "amount": [day * 10 + i for i in range(3)]})
    rows.write.delta(clustered, mode="append")

# The log says only one file can hold day == 2, so only that file is opened.
print(bt.read.delta(clustered).filter(bt.col("day") == 2).count())
# 3
```

You can see the elimination happen at planning time. The predicate cuts the *file* list,
not only the rows:

```python
from batcher.io.formats.lakehouse import DeltaSource

predicate = (bt.col("day") == 2).to_ir()
source = DeltaSource(clustered)
print(len(source.splits()), "->", len(source.splits(predicate=predicate)))
# 4 -> 1
```

This is not merely a row filter that runs earlier. The pruning happens *before* I/O, so
the files it eliminates cost nothing at all: no footer read, no split, no task. On a
distributed read that is the difference between one worker task per file in the table and
one per file that can actually contribute. Projection pushdown prunes columns the same
way, and whatever survives is then row-group and page pruned inside the file.

Pruning is deliberately one-sided: a file is dropped only when the log *proves* it cannot
match. A missing statistic, an unrecorded column, or a predicate the log cannot decide
all keep the file. Skipping therefore only ever costs extra I/O, never a missing row. The
engine re-checks every predicate regardless.

`count()` benefits too. A `count(*)` over a filter is fused into a single counting pass,
and the predicate still reaches the source, so the count is answered from the files the
log says can contribute. An *unfiltered* `count()` needs no files at all, because the
log's own record counts are exact.

Each surviving data file is an independent split carrying its row count from the log, so
a table larger than any single node is read file by file across workers and never
materialized on the driver. One mergeable read path, single-node and distributed.

## Writes leave the index the next read uses

A write is the other half of the same mechanism. Each worker writes its shard as a final
data file and records that file's column bounds while the data is still in memory. The
driver then commits only the resulting *add actions*, meaning paths, sizes, partition
values, and those statistics.

Two consequences follow. The driver never re-encodes the result, so its cost is one log
write no matter how much the cluster wrote, and the bytes move exactly once, from worker
to storage. And the statistics the write leaves behind are precisely what the next query
prunes against, so a table written this way is a table that can be read with file
skipping.

A distributed write is **one** transaction: workers produce files, the driver commits
once. The write is atomic, so a reader never sees half of it, and the log records one
version per logical write, not one per worker.

## See also

- {doc}`Data quality <data-quality>`: validate and quarantine before you commit.
- {doc}`Writing data <writing-data>`: save modes, partitioning, atomic writes.
- {doc}`I/O API <../api/io>`: the full `read`/`write` reference.
- {doc}`Agent skills <../agents/index>`: `manage-a-lakehouse-table` is this page as a
  procedure for a coding agent, covering merge, SCD, CDC, backfill, and compaction.
