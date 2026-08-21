# Apache Iceberg

Batcher reads and writes Iceberg tables through pyiceberg
(`pip install 'batcher-engine[iceberg]'`). Reads support snapshot time travel and manifest-level
file pruning; writes are `append` or `overwrite`, committed as one snapshot by the driver.

| | |
| --- | --- |
| **Read** | {py:meth}`bt.read.iceberg(identifier, catalog=...) <batcher.api.io_namespace.reader.Reader.iceberg>`, with `snapshot_id=` |
| **Write** | {py:meth}`ds.write.iceberg(identifier, mode="append"\|"overwrite") <batcher.api.io_namespace.writer.Writer.iceberg>` |
| **Extra** | `pip install 'batcher-engine[iceberg]'` |
| **Parallelism** | One split per data file surviving `plan_files`; serial on merge-on-read |
| **Pushdown** | Predicates into `plan_files`, answered against the manifests |
| **Credentials** | The catalog properties authenticate to the catalog, not to storage |

Budget your time for the catalog, not the code. Iceberg addresses a table by identifier
(`db.orders`), and a catalog is what maps that identifier to a metadata file. Getting the catalog
properties right (REST URI, warehouse location, credential vending, and the storage credentials the
*catalog* hands back versus the ones your process already has) is where an Iceberg integration
actually goes wrong. `bt.read.iceberg("db.orders")` is the easy part.

## The catalog

`catalog=` takes either the name of a catalog already configured in `~/.pyiceberg.yaml` (or the
environment), or a property mapping.

:::{dropdown} The catalog types Batcher recognizes
Batcher normalizes friendly type aliases onto pyiceberg's own: `rest`, `unity` / `databricks` /
`polaris` / `snowflake` (all REST), `glue`, `hive`, `sql` / `jdbc`, `dynamodb`, `in-memory`.
Everything else in the mapping passes through to pyiceberg unchanged.
:::

::::{tab-set}

:::{tab-item} A REST catalog

```python
# docs: skip
import batcher as bt

prod = {
    "type": "rest",
    "uri": "https://catalog.example.com/api/catalog",
    "warehouse": "s3://lake/warehouse",
    "token": "...",
}
orders = bt.read.iceberg("db.orders", catalog=prod)
```
:::

:::{tab-item} A local SQL catalog

A local SQL catalog over SQLite is the fastest way to try the connector end to end, and it is what
the runnable blocks on this page use. The namespace has to exist before a table can be created in
it, and pyiceberg will not do that for you.

```python
import os
import tempfile

import batcher as bt
from batcher.io.catalog import resolve_catalog

work = tempfile.mkdtemp()
warehouse = os.path.join(work, "warehouse")
os.makedirs(warehouse, exist_ok=True)

catalog = {
    "type": "sql",
    "uri": f"sqlite:///{work}/catalog.db",
    "warehouse": f"file://{warehouse}",
}
resolve_catalog(dict(catalog)).create_namespace_if_not_exists("db")
```
:::

::::

## Write and read

`ds.write.iceberg(identifier, mode="append"|"overwrite")`. The table is created from the written
schema if it does not exist.

```python
orders = bt.from_pydict({"id": [1, 2, 3], "amount": [10, 20, 30]})
orders.write.iceberg("db.orders", mode="append", catalog=catalog)

print(bt.read.iceberg("db.orders", catalog=catalog).sort("id").to_pydict())
# {'id': [1, 2, 3], 'amount': [10, 20, 30]}
```

## Time travel

Every commit is a snapshot. Grab its id before you write again, and `snapshot_id=` reads the table
as it was.

```python
before = resolve_catalog(dict(catalog)).load_table("db.orders").current_snapshot().snapshot_id

bt.from_pydict({"id": [4], "amount": [40]}).write.iceberg(
    "db.orders", mode="append", catalog=catalog
)

print(bt.read.iceberg("db.orders", catalog=catalog).sort("id").to_pydict()["id"])
# [1, 2, 3, 4]
print(bt.read.iceberg("db.orders", catalog=catalog, snapshot_id=before).sort("id").to_pydict()["id"])
# [1, 2, 3]
```

`count()` is answered from the snapshot summary's `total-records`, so an unfiltered count reads no
data files at all.

```python
print(bt.read.iceberg("db.orders", catalog=catalog).count())
# 4
```

## How it parallelizes

Kyber pushes the query's predicate into `plan_files`, so pyiceberg answers it against the
manifests' partition values and column bounds and returns only the data files that can hold a
matching row. Batcher makes one split per surviving file, each carrying the manifest's record
count, so the distributed planner bin-packs by real size without opening a single Parquet footer. A
`WHERE day = '2024-03-01'` on a table partitioned by day never lists or schedules the other days.

Writes are shared-nothing. Each worker writes its shard as a real Parquet file into a staging area
under the catalog warehouse and returns only the file locator; the driver registers every staged
file with `add_files` in one snapshot. The files are referenced **in place**, never re-read or
rewritten by the driver.

:::{important}
That has a consequence worth internalizing: the staging directory under
`<warehouse>/<table>/_batcher_staging/` is not scratch. Those Parquet files *are* the table's data
files. Do not sweep the directory.
:::

Staged names carry a per-write token, so a later write cannot clobber a file an earlier snapshot
still references, while a preempted-and-rerun shard overwrites its own file and stays idempotent.

## Failure modes worth knowing

:::{warning}
**Merge-on-read kills the parallel read.** If a scan task carries positional or equality delete
files, reading the data file directly would resurrect every deleted row. Rather than return wrong
data, Batcher falls back to a whole-source scan through pyiceberg, which applies the deletes. It is
correct and it is serial. Copy-on-write tables keep the split-parallel path; if an Iceberg read is
mysteriously single-threaded, look for delete files.
:::

**Writes are append/overwrite only.** No merge-on-read writes, no equality deletes, no row-level
`MERGE`. pyiceberg's support is not solid enough to build on, and Batcher raises rather than
pretending. If you need upserts today, use {doc}`Delta </integrations/lakehouse/delta-lake>`, where `merge_on=` is a real
`MERGE INTO`.

:::{important}
**`overwrite` is a full delete-then-add.** It issues `delete(AlwaysTrue())` and then registers the
new files, in one catalog transaction. That is not a partition-scoped replace, and there is no
`replace_where` for Iceberg.
:::

## Computing a partition value

An Iceberg table does not store the partition column, it stores a *transform* of it:
`days(ts)`, `months(ts)`, `truncate(4, name)`. Batcher exposes those transforms as ordinary
expressions, so you can compute the value a row will be partitioned by before it is written,
group by it, or filter on it:

```python
import batcher as bt
import datetime as dt

events = bt.from_pydict(
    {
        "ts": [dt.datetime(2024, 3, 5, 13, 0), dt.datetime(2024, 3, 5, 21, 0)],
        "amount": [10, 20],
    }
)
by_day = events.group_by(day=bt.partition_days("ts")).agg(total=bt.col("amount").sum())
print(by_day.to_pydict())
# {'day': [19787], 'total': [30]}
```

The four time transforms count from the epoch and go negative before it, exactly as the
specification says: {py:func}`bt.partition_years(ts) <batcher.partition_years>`,
{py:func}`bt.partition_months(ts) <batcher.partition_months>`,
{py:func}`bt.partition_days(ts) <batcher.partition_days>` and
{py:func}`bt.partition_hours(ts) <batcher.partition_hours>`.
{py:func}`bt.partition_truncate(value, width) <batcher.partition_truncate>` rounds a number
down to a multiple of `width`, floored toward negative infinity, so `-7` at width `5` is
`-10`. All five are also callable from SQL under the same names. For the text reading of
`truncate`, take the prefix directly with `col("s").str.substr(1, width)`.

:::{note}
`bucket` is not provided. Iceberg pins it to a specific 32-bit MurmurHash3 over each type's
canonical byte encoding, and computing it any other way would send rows to different files
than the table's own writer chooses. A near-miss here is worse than an absence, because
nothing errors.
:::

**Partitioning belongs to the catalog.** The table's partition spec is a table property, and
`add_files` places each file according to it at commit time. A `partition_by=` on the write is
ignored, so set the spec when the table is created.

**Credentials come from two places.** The catalog properties authenticate to the *catalog*; the
data files are then read from object storage with whatever credentials the process has, or the ones
the catalog vends. A read that authenticates fine and then 403s on the first Parquet file is almost
always this.

**Version drift.** A pyiceberg whose expression API differs degrades to no predicate pushdown
rather than failing, so the result is right and the scan is wide. The incremental append scan
(`read_incremental`) is stricter: on a version that lacks it, you get a clear {py:exc}`BackendError <batcher.BackendError>`.

## See also

- {doc}`Lakehouse </user-guide/moving-data/lakehouse>`: the table-format guide.
- {doc}`Writing data </user-guide/moving-data/writing-data>`: modes, partitioning, and what a commit is.
- {doc}`Schema evolution </cookbook/data-engineering/modeling/schema-evolution>`: adding a column to a
  table other jobs are reading.
- {doc}`Partition backfill </cookbook/data-engineering/maintenance/partition-backfill>`: why `overwrite`
  being table-wide matters here.
- {doc}`I/O API </api/relational/io>`: the full reader/writer reference.
- {doc}`Delta Lake </integrations/lakehouse/delta-lake>`: upserts, `replace_where`, vacuum.
- {doc}`Hudi </integrations/lakehouse/hudi>`: the read-only third format.
