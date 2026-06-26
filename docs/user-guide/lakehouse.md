# Lakehouse table formats

Batcher reads and writes the transactional table formats that back a lakehouse —
Delta Lake, Apache Iceberg, and Apache Hudi — plus the maintenance patterns built
on top of them: `MERGE` upserts, partition backfills, and slowly-changing-dimension
history. A Delta write is a single atomic commit, so a reader never sees a partial
table, and time travel lets you query any past version. The same mergeable engine
runs these on one core or across a cluster with an identical result.

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
read the table as it was at an earlier commit — the first write above is version `0`.

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
wholesale — a backfill or idempotent reload — use `replace_where` instead.

## Partition backfill with replace_where

`replace_where=<predicate>` is a dynamic partition/range overwrite (Delta's
`replaceWhere`): atomically replace only the rows matching the predicate and keep
the rest. It is predicate-scoped, not key-matched — exactly the idempotent-backfill
pattern, where re-running a day's job replaces that day's rows and nothing else.

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

The two `2024-01-01` rows are replaced by the single backfilled row; the
`2024-01-02` row is untouched. `replace_where` works on any file target (here
Parquet) and on Delta tables.

## Slowly changing dimensions

The `ds.scd` accessor maintains dimension tables from an incoming snapshot. The
dataset is the new snapshot (natural keys plus attributes); the method writes the
reconciled dimension to `target`. These compose existing operators — there is no
special engine path — and write to any file target (Parquet here) or a Delta table.

### Type 1 — overwrite in place

Type 1 keeps no history: a matched key's attributes are overwritten (a keyed
upsert). Here `id=2` moves from `LA` to `SF`.

```python
t1 = os.path.join(work, "city_t1.parquet")
bt.from_pydict({"id": [1, 2], "city": ["NYC", "LA"]}).write.parquet(t1)
bt.from_pydict({"id": [2], "city": ["SF"]}).scd.type1(t1, keys="id")

print(bt.read.parquet(t1).sort("id").to_pydict())
# {'id': [1, 2], 'city': ['NYC', 'SF']}
```

### Type 2 — full history

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

`id=1` now has two versions — the expired `NYC` and the current `SF` — while the
unchanged `id=2` keeps its single open version.

### Type 3 — previous value

Type 3 keeps only the immediately previous value of each tracked attribute in a
`<attr>_prev` column.

```python
t3 = os.path.join(work, "city_t3.parquet")
bt.from_pydict({"id": [1], "city": ["NYC"]}).scd.type3(t3, keys="id", track=["city"])
bt.from_pydict({"id": [1], "city": ["LA"]}).scd.type3(t3, keys="id", track=["city"])

print(bt.read.parquet(t3).sort("id").to_pydict())
# {'id': [1], 'city': ['LA'], 'city_prev': ['NYC']}
```

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

Hudi is supported read-only (`pip install 'batcher-engine[hudi]'`); writes require
the Spark/Flink write stack.

```python
# docs: skip
import batcher as bt

events = bt.read.hudi("s3://lake/hudi/events")
```

## Pushdown and splits

Lakehouse reads are pushdown-aware. Kyber pushes column projection and row-group
predicates into the Delta/Iceberg reader, so a filtered, projected query reads only
the needed columns and prunes partitions and files using the table's own statistics
(the transaction-log add-action stats for Delta). Each data file becomes an
independent read split, so a table larger than any single node is read fragment by
fragment across workers and never materialized on the driver — the same mergeable
read path single-node and distributed.

## Next steps

- [Data quality](data-quality.md): validate and quarantine before you commit.
- [Writing data](writing-data.md): save modes, partitioning, and atomic writes.
- [I/O API](../api/io.md): the full `read`/`write` reference.
