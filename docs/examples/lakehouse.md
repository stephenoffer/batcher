# Lakehouse tables

This page covers the scripts that manage Delta tables: commits, upserts, history, and the
maintenance that keeps them fast.

## A write is a commit

Every write is one transaction, which is what makes a half-finished write invisible to
readers. They keep seeing the previous version until the commit lands, so there is no window
where a query sees part of a batch.

```python
# docs: skip
import batcher as bt

# Three commits, each a complete snapshot.
orders.head(1_000).write.delta(table)
orders.slice(1_000, 500).write.delta(table, mode="append")
orders.head(200).write.delta(table, mode="overwrite")

assert bt.read.delta(table).count() == 200
assert bt.read.delta(table, version=0).count() == 1_000
```

Time travel is a consequence of that log rather than a backup feature: each commit adds files
rather than replacing them, so an older version is still fully described. That is also what
makes `vacuum` the destructive operation, because it removes the files older versions point
at.

## Upserts and change feeds

`merge_on` performs a `MERGE INTO` keyed on the columns you name: matched rows update, the
rest insert, in one commit. Doing it as a delete followed by an append is two commits with a
window in between where readers see neither version.

That also makes a replay idempotent, which is the mechanism behind exactly-once delivery. An
append replayed twice duplicates its rows; a keyed merge replayed twice is a no-op, and
`examples/streams/exactly_once_semantics.py` asserts both.

## Maintenance

An incremental writer leaves one small file per commit, and the next write cannot fix that.
Eventually the table costs more to plan than to read. Compaction bin-packs the files in a
transaction that never deletes anything an older version still references, so every version
stays readable.

## Every script on this page

The table below lists the lakehouse scripts in path order.

<!-- library-table: lakehouse -->
| Script | Shows |
| --- | --- |
| `examples/lakehouse/change_data_capture.py` | Applying a change feed: inserts, updates and deletes in one commit |
| `examples/lakehouse/compaction.py` | The small-files problem, and compacting a table that has it |
| `examples/lakehouse/delta_upserts.py` | MERGE INTO: upserting keyed rows into a Delta table |
| `examples/lakehouse/partition_backfill.py` | Replacing one partition without touching the rest |
| `examples/lakehouse/scd_type_two.py` | Slowly changing dimensions: keeping the history of a changed row |
| `examples/lakehouse/schema_evolution_on_write.py` | Adding a column to a table that already has data |
| `examples/lakehouse/snapshot_isolation.py` | Snapshot isolation: a reader sees one version, whatever the writer is doing |
| `examples/lakehouse/table_maintenance.py` | Table maintenance: compaction, vacuum, and the version they cost you |
<!-- /library-table -->
