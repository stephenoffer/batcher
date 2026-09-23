# Apache Iceberg

This page covers reading and writing Apache Iceberg tables. Batcher works through pyiceberg, with no JVM: reads get snapshot time travel, manifest-level file pruning, and merge-on-read deletes applied in parallel, and writes stage data files on the workers and commit them as one snapshot. Upserts run as a native Iceberg merge, and a Puffin statistics file someone else's engine published feeds Batcher's join planning.

The following table summarizes the connector:

| | |
| --- | --- |
| Read | {py:meth}`bt.read.iceberg(identifier, catalog=...) <batcher.api.io_namespace.reader.Reader.iceberg>`, with `snapshot_id=` |
| Write | {py:meth}`ds.write.iceberg(identifier, mode="append"\|"overwrite") <batcher.api.io_namespace.writer.Writer.iceberg>`, with `replace_where=`, and {py:meth}`ds.write.merge <batcher.api.io_namespace.writer.Writer.merge>` for upserts |
| Extra | `pip install 'batcher-engine[iceberg]'` |
| Parallelism | One split per data file surviving `plan_files`, merge-on-read included |
| Pushdown | Predicates into `plan_files`, answered against the manifests |
| Maintenance | {py:func}`bt.vacuum <batcher.vacuum>` expires snapshots |
| Credentials | The catalog properties authenticate to the catalog, not to storage |

Budget your time for the catalog, not the code. Iceberg addresses a table by identifier, such as `db.orders`, and a catalog maps that identifier to a metadata file. Getting the catalog properties right is where an Iceberg integration goes wrong: the REST URI, the warehouse location, credential vending, and the storage credentials the catalog hands back versus the ones your process already has. `bt.read.iceberg("db.orders")` is the easy part.

The figure follows a read down the table's metadata tree, from the catalog to the data files that become splits.

![Four layers. The catalog maps the identifier db.orders to the table's metadata file, and catalog= takes a configured name or a property mapping. The metadata holds one snapshot per commit: a write turns the current snapshot into an earlier one, the default read uses the current snapshot, snapshot_id= reads an earlier one, and count() comes from the snapshot's total-records. A snapshot's manifest list leads to manifests carrying partition values and column bounds, which answer the predicate Kyber pushes into plan_files. Only the data files that can match survive, the rest are pruned, and each kept Parquet file becomes one split sized by its record count without reading a footer.](/_static/diagrams/iceberg_snapshot_tree.svg)

## The catalog

`catalog=` takes either the name of a catalog already configured in `~/.pyiceberg.yaml` or the environment, or a property mapping.

:::{dropdown} The catalog types Batcher recognizes
Batcher maps friendly type aliases onto pyiceberg's own. `rest`, `iceberg-rest`, `unity`, `unity-catalog`, `databricks`, `polaris`, and `snowflake` all resolve to REST. `glue`, `hive` or `hive-metastore`, `sql` or `jdbc`, `dynamodb`, and `in-memory` resolve to their pyiceberg types. Everything else in the mapping passes through to pyiceberg unchanged.
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

A local SQL catalog over SQLite is the fastest way to try the connector end to end, and it's what the runnable blocks on this page use. It needs pyiceberg's SQLite support, `pip install 'pyiceberg[sql-sqlite]'`. The namespace has to exist before a table can be created in it, and pyiceberg won't create it for you.

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

`ds.write.iceberg(identifier, mode=...)` takes `mode="append"` (the default) or `"overwrite"`, and creates the table from the written schema if it doesn't exist.

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
print(
    bt.read.iceberg("db.orders", catalog=catalog, snapshot_id=before).sort("id").to_pydict()["id"]
)
# [1, 2, 3]
```

`count()` is answered from the snapshot summary's `total-records`, so an unfiltered count reads no data files at all. When the snapshot carries positional delete files, the summary overcounts, so Batcher counts the rows it reads instead.

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
file with `add_files` in one snapshot. The files are referenced in place, never re-read or rewritten by the driver.

:::{important}
The staging directory under `<warehouse>/<table>/_batcher_staging/` is not scratch. Those Parquet files *are* the table's data
files. Do not sweep the directory.
:::

Staged names carry a per-write token, so a later write can't clobber a file an earlier snapshot still references, while a preempted and rerun shard overwrites its own file and stays idempotent.

A partitioned table owns its layout. The partition spec lives in the catalog, and the writer splits each shard along that spec before staging it, so you don't pass `partition_by=`. Set the spec when you create the table.

## Upsert and partition replace

{py:meth}`ds.write.merge(identifier, on=...) <batcher.api.io_namespace.writer.Writer.merge>` runs pyiceberg's native upsert: it joins the change set against the table, rewrites only the data files the join touches, and commits one snapshot. Pass `format="iceberg"` and the catalog:

```python
# docs: skip
changes = bt.from_pydict({"id": [2, 5], "amount": [999, 50]})
changes.write.merge("db.orders", on="id", format="iceberg", catalog=catalog)
```

Iceberg's upsert is the two-clause merge: update every matched row, insert every unmatched one. A {py:meth}`merge_into <batcher.api.io_namespace.writer.Writer.merge_into>` statement with a clause condition, a `DELETE`, a partial column update, or `when_not_matched_by_source` raises {py:exc}`PlanError <batcher.PlanError>` rather than running a different statement. Use {doc}`Delta </integrations/lakehouse/delta-lake>` for the full `MERGE`.

`mode="overwrite"` is table-wide. For a partition-scoped replace, pass a predicate such as `replace_where=bt.col("day") == "2024-03-01"`. Batcher translates it to an Iceberg expression that deletes exactly the rows it matches, in the same transaction that adds the new files. A predicate the table can't express is refused, never widened into an overwrite of everything.

## Expire snapshots

Every write leaves a snapshot, and each snapshot pins the data files it references. {py:obj}`bt.vacuum <batcher.vacuum>` expires snapshots older than the retention window, 5 days by default, which is what lets storage be reclaimed. Like the Delta version it defaults to a dry run and returns the snapshot ids it would expire:

```python
# docs: skip
print(bt.vacuum("db.orders", format="iceberg", catalog=catalog))
bt.vacuum("db.orders", format="iceberg", catalog=catalog, retention_hours=24, dry_run=False)
```

{py:obj}`bt.compact <batcher.compact>` raises on an Iceberg table, because rewriting data files is Spark's `rewrite_data_files` procedure and pyiceberg doesn't implement it. Cluster the data on the way in with `ds.write.iceberg(..., sort_by=[...])` instead.

## Table statistics

An Iceberg table can carry a statistics file per snapshot: a Puffin blob per column whose
`ndv` property holds the distinct-count estimate that whichever engine last ran an ANALYZE
wrote. Batcher reads them, so the first query against a table someone else analyzed plans
its joins on a real distinct count rather than a default heuristic.

Nothing is required of you. If the table publishes statistics they're used, and if it doesn't the plan falls back to its default estimates.

A published count is treated as an estimate, never as an answer. It informs join ordering,
build-side choice, and equality selectivity. It can never satisfy `count_distinct`, which
still executes. The distinct count and the column's min/max carry separate trust tags, so
attaching a sketch does not disturb the exact manifest bounds that answer `min()` and
`max()` without a scan.

Statistics computed for an ancestor snapshot are used when the current snapshot has none,
which is the normal case: an ANALYZE is almost always older than the newest append. A
statistics file belonging to a snapshot that is not an ancestor of the one being read is
ignored, because it describes rows this read will not see.

Batcher doesn't write statistics files. The specification requires the blob to
be a conformant Apache DataSketches theta sketch, and Batcher's own mergeable sketches are
HLL. Publishing an HLL under a theta blob type would produce a file every other engine
misreads, which is worse than publishing nothing.

## Merge-on-read tables

A split carries the planned `FileScanTask` itself and reads it through Iceberg's own scanner, which resolves columns by field id and applies positional and equality delete files. A table with deletes neither resurrects rows nor collapses to one worker. Batcher falls back to a whole-source scan only when `plan_files` raises or returns nothing.

## Compute a partition value

An Iceberg table doesn't store the partition column. It stores a *transform* of it:
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

## Requirements and limitations

Writes are append, overwrite, `replace_where`, and the two-clause upsert. Batcher doesn't write merge-on-read delete files or equality deletes, and it doesn't compact.

Credentials come from two places. The catalog properties authenticate to the catalog, and the data files are then read from object storage with whatever credentials the process has, or the ones the catalog vends. A read that authenticates and then gets a 403 on the first Parquet file is almost always this.

A pyiceberg whose expression API differs degrades to no predicate pushdown rather than failing, so the result stays right and the scan is wide.

## See also

- {doc}`Lakehouse </user-guide/moving-data/lakehouse>`: the table-format guide.
- {doc}`Writing data </user-guide/moving-data/writing-data>`: modes, partitioning, and what a commit is.
- {doc}`Schema evolution </cookbook/data-engineering/modeling/schema-evolution>`: adding a column to a
  table other jobs are reading.
- {doc}`Partition backfill </cookbook/data-engineering/maintenance/partition-backfill>`: `replace_where=` as a re-runnable job.
- {doc}`I/O API </api/relational/io>`: the full reader/writer reference.
- {doc}`Delta Lake </integrations/lakehouse/delta-lake>`: upserts, `replace_where`, vacuum.
- {doc}`Hudi </integrations/lakehouse/hudi>`: the third table format.
