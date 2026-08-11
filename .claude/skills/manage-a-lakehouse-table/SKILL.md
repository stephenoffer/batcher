---
name: manage-a-lakehouse-table
description: Run the full lifecycle of a Delta/Iceberg/Hudi table from Batcher — transactional append and overwrite, the MergeBuilder MERGE INTO DSL for keyed upserts and expiry, ds.scd type 1/2/3 and CDC apply_changes, change feeds, time travel, replace_where partition backfill, schema evolution on write, and compact/vacuum maintenance. Invoke when building or maintaining a lakehouse table — upserts, dimensions, CDC ingestion, backfills, snapshots, or the small-files problem.
---

# Manage a lakehouse table

A lakehouse table is a directory of Parquet plus a transaction log. That log is what buys
atomic multi-file commits, time travel, concurrent writers, and row-level `MERGE` — none of
which a plain directory can offer. This skill is the **table lifecycle**. Getting bytes in
and out of *files* (formats, globs, cloud paths, schema inference) is
`read-and-write-data`; relational verbs are `write-a-batcher-pipeline`.

Verify before citing: `bt.read` has `delta`, `iceberg`, `hudi`, `delta_sharing`,
`read_change_feed`, `lance`; `ds.write` has `delta`, `iceberg`, `hudi`, `merge`,
`merge_into`; `ds.scd` has `type1`, `type2`, `type3`, `apply_changes`.

## Reading a table

```python
bt.read.delta(uri)                          # latest version
bt.read.delta(uri, version=0)               # time travel by version
bt.read.delta(uri, timestamp="2024-06-01")  # ... or by wall clock
bt.read.iceberg("db.t", catalog=spec, snapshot_id=snaps[0])
bt.read.hudi(uri)                           # read-only snapshot query
bt.read.delta_sharing("config.share#share.schema.table")
```

An Iceberg `catalog=` is a **property mapping**, not just a name — `{"type": ...}` selects
the backend (`rest` covers Unity Catalog / Polaris / Tabular; also `glue`, `hive`, `sql`,
`dynamodb`, `in-memory`) and the rest passes through to pyiceberg (`io/catalog.py`). The
parameter is annotated `str | None`, but a dict spec is the working form:

```python
spec = {"type": "sql", "name": "local", "uri": f"sqlite:///{wh}/cat.db", "warehouse": f"file://{wh}"}
bt.from_pydict({"id": [1, 2], "v": [10.0, 20.0]}).write.iceberg("db.t", mode="append", catalog=spec)
bt.read.iceberg("db.t", catalog=spec).sort("id").to_pydict()   # {'id': [1, 2], 'v': [10.0, 20.0]}
```

Note `bt.read(path)` **auto-detects a table** from its marker directory (`_delta_log`,
`metadata`, `.hoodie`) before falling back to the extension. That matters: a Delta table
*is* a directory of Parquet, so reading one as `format="parquet"` bypasses the log, and a
maintenance rewrite that does so deletes files older versions still reference.

## Writing: append, overwrite, upsert

```python
ds.write.delta(uri, mode="append")                    # one transactional commit
ds.write.delta(uri, mode="overwrite")
ds.write.delta(uri, merge_on="id")                    # MERGE INTO upsert, keyed
ds.write.iceberg("db.t", mode="append", catalog=spec)
ds.write.hudi(uri, mode="append")
```

`append` on a lakehouse sink is a real transaction, not a new part file; `mode="append"` is
rejected on plain file sinks for exactly that reason. `merge_on=` builds the match
predicate from the keys (pass `merge_predicate=` for a custom one) — against a table
holding ids 1–4, upserting `{2, 9}` updates 2 and inserts 9, leaving 1/3/4 untouched:

```python
bt.from_pydict({"id": [2, 9], "amount": [99.0, 90.0], "region": ["eu", "ap"]}) \
  .write.delta(uri, merge_on="id")
bt.read.delta(uri).sort("id").to_pydict()
# {'id': [1, 2, 3, 4, 9], 'amount': [10.0, 99.0, 30.0, 40.0, 90.0], ...}
```

`ds.write.merge(target, on=, when_matched=, when_not_matched=)` is the portable two-clause
shorthand — native `MERGE` on Delta, copy-on-write on a plain file target (read, reconcile,
atomically overwrite; single-writer only).

## `MergeBuilder` — the full `MERGE INTO`

`ds.write.merge_into(target, on=...)` opens the general statement. **`ds` is the source**
(the change set); `target` is the table. Each `when_*` names a population and returns only
the actions legal for it — an insert clause has no `delete()`, a not-matched-by-source
clause has no `insert()`. Clauses are tried **in the order added; first match wins**, which
is SQL's rule and the whole reason clause order is part of the semantics.

Reference the two sides with `bt.source_col(name)` and `bt.target_col(name)`.

```python
from batcher import lit, source_col

# target: id 1,2,3 all status="active";  changes: update 2, delete 3, insert 4
(changes.write.merge_into(uri, on="id")
    .when_matched(source_col("op") == lit("D")).delete()
    .when_matched().update(amount=source_col("amount"))
    .when_not_matched().insert(id=source_col("id"),
                               amount=source_col("amount"),
                               status=lit("active"))
    .when_not_matched_by_source().update(status=lit("stale"))
    .execute())

bt.read.delta(uri).sort("id").to_pydict()
# {'id': [1, 2, 4], 'status': ['stale', 'active', 'active'], 'amount': [10, 99, 40]}
```

Read that result: `3` was deleted, `2` was updated, `4` was inserted, and `1` — which the
change set never mentioned — was marked `stale` by the not-matched-by-source clause.

- `when_matched(cond)` → `update(**cols)` / `update_all()` / `delete()`; the condition may
  read both sides.
- `when_not_matched(cond)` → `insert(**cols)` / `insert_all()`; no target row exists, so
  the condition may read `source_col` only, and unnamed columns become NULL.
- `when_not_matched_by_source(cond)` → `update(**cols)` / `update_all()` / `delete()`; the
  population a plain upsert ignores, and the condition may read `target_col` only.
- `update_all()` / `insert_all()` are `UPDATE SET *` / `INSERT *`. An **empty**
  `update()`/`insert()` raises rather than silently writing nothing. `execute()` commits
  and returns a `WriteManifest`.

**Cost model.** A merge rewrites only the data files whose key statistics prove they could
hold one of the source's keys, so an upsert costs the change set rather than the table
(`prune=False` disables it — correctness never depends on it). The exception is inherent:
a `when_not_matched_by_source` clause is *about* the untouched rows, so every file is a
candidate and the whole table is rewritten. Delta and Snowflake pay the same price.

## `ds.scd` — dimension maintenance

Built entirely from merge/join/union — no new IR. The dataset is the **incoming
dimension**.

```python
# type 1 — overwrite in place, no history (a keyed upsert)
bt.from_pydict({"id": [2, 3], "city": ["SF", "BOS"]}).scd.type1(t, keys="id")
# {'id': [1, 2, 3], 'city': ['NYC', 'SF', 'BOS']}

# type 2 — full history via effective dating. id 1 changed NYC→SF: its old row is closed
# (valid_to=as_of, is_current=False) and a new open version appended. id 2 is untouched.
snapshot.scd.type2(t, keys="id", track=["city"], as_of="2024-06-01")
# {'id': [1, 1, 2], 'city': ['NYC', 'SF', 'LA'],
#  'valid_from': ['2024-01-01', '2024-06-01', '2024-01-01'],
#  'valid_to':   ['2024-06-01', None, None], 'is_current': [False, True, True]}

# type 3 — keep one prior value per tracked attribute
bt.from_pydict({"id": [1], "city": ["LA"]}).scd.type3(t, keys="id", track=["city"])
# {'id': [1], 'city': ['LA'], 'city_prev': ['NYC']}
```

Column names are configurable (`valid_from=`, `valid_to=`, `is_current=`). `type2`
compares only the `track` columns, so an unrelated attribute changing does not open a
version.

## CDC: `apply_changes` and the change feed

`type1`–`type3` take a **clean snapshot of now**. A CDC connector (Debezium, a Delta change
feed, a Snowflake stream) emits **what happened** — deletes, redeliveries, out-of-order
rows — which `type1` cannot consume. `ds.scd.apply_changes` is that shape, following Delta
Live Tables' `APPLY CHANGES INTO ... SCD TYPE 1`:

```python
feed = bt.from_pydict({"id": [1, 2, 1], "city": ["NYC", "LA", "SF"],
                       "op": ["I", "I", "U"], "seq": [1, 2, 3]})
feed.scd.apply_changes(t, keys="id", sequence_by="seq",
                       deletes=bt.col("op") == "D", columns=["id", "city"])
# {'id': [1, 2], 'city': ['SF', 'LA']}     ← seq 3 beats seq 1 within the batch

later = bt.from_pydict({"id": [2, 1], "city": ["LA", "OLD"],
                        "op": ["D", "U"], "seq": [4, 0]})
later.scd.apply_changes(t, keys="id", sequence_by="seq",
                        deletes=bt.col("op") == "D", columns=["id", "city"])
# {'id': [1], 'city': ['SF']}    ← 2 deleted; the seq-0 change for 1 is stale, discarded
```

- Within a batch only the greatest-`sequence_by` change per key survives.
- Across batches a change applies only if its sequence is **≥** the one already stored, so
  re-applying a batch is a no-op. `sequence_by` is persisted in the target for exactly this.
- `columns=` selects what to store, dropping CDC control columns; keys and `sequence_by`
  are always kept.

**It is idempotent but not commutative.** A delete is physical, so a deleted key stores no
sequence to compare against and replaying an *old insert* resurrects it. Feed batches in
sequence order; treat a full replay of a feed containing deletes as a rebuild, not a resume.
Like `type1`, it is a copy-on-write overwrite — single-writer only.

To *produce* a feed, read one. The table needs `delta.enableChangeDataFeed = true`, set with
`table_properties=` on any write — when the write creates the table, or as an alter on an
existing one. CDF is not retroactive, so set it before the commits you want to read back.

```python
ds.write.delta(uri, table_properties={"delta.enableChangeDataFeed": "true"})
```

**Name a bound and the feed is a bounded relation; name none and it is a stream.** The
bounded form is what an incremental job wants, because an unbounded source cannot be
collected, counted, or joined.

```python
# Bounded: a closed window you can merge into a target, and re-run identically after a crash.
changes = bt.read.read_change_feed(uri, starting_version=last + 1, ending_version=latest)
changes.is_streaming   # False
changes.count()        # works
# `starting_timestamp=` / `ending_timestamp=` bound by time instead; both take a datetime,
# a date, 'YYYY-MM-DD', or 'YYYY-MM-DD HH:MM:SS' (naive values are the driver's local time).

# Unbounded: no bound named, for a continuous query.
stream = bt.read.read_change_feed(uri, starting_version=0)
stream.is_streaming    # True
stream.columns  # ['id', 'v', '_change_type', '_commit_version', '_commit_timestamp']
# _change_type ∈ insert / update_preimage / update_postimage / delete
```

Record the version you processed, read the next window from there, merge, then advance the
watermark — bounding both ends is what makes the step re-runnable.

`bt.read.delta(uri, stream=True, starting_version=n)` is the same source without the
row-level change columns.

## `replace_where` — partition backfill

Replacing a known slice (one day, one region) is not an upsert: there are no keys to match,
you want the whole range gone and rewritten. That is `replace_where=`, and on Delta it is a
**scoped commit** — workers write the new partition's files, the driver retires exactly the
matching partitions from the log. Backfilling one day of a 100 TB table costs one day.

```python
bt.from_pydict({"day": ["a", "a", "b"], "v": [1, 2, 3]}) \
  .write.delta(ev, mode="append", partition_by=["day"])
bt.from_pydict({"day": ["a"], "v": [99]}) \
  .write(ev, "delta", partition_by=["day"], replace_where=bt.col("day") == lit("a"))
bt.read.delta(ev).sort("v").to_pydict()      # {'day': ['b', 'a'], 'v': [3, 99]}
```

Two things that will bite you, both by design:

- The predicate must be an **AND of `partition_col == value` over the table's partition
  columns**. Anything delta-rs cannot turn into partition filters is refused with
  `CommitError` rather than widened into a full-table overwrite.
- `partition_by=` must be passed on the **replacing** write too, not just the create — the
  sink is constructed per write and has no memory of the last one.

On a non-lakehouse target `replace_where` degrades to copy-on-write: read the whole table,
filter out the range, union, overwrite. Correct, but a one-day backfill becomes a
full-table rewrite. Use Delta.

## Schema evolution on write

New columns are **refused by default** — an unexpected column is more often a bug than an
intent, and writing it into files the table cannot see is unrecoverable:

```python
bt.from_pydict({"day": ["c"], "v": [7], "extra": ["x"]}).write.delta(ev, mode="append")
# CommitError: the write has column(s) ['extra'] that Delta table ... does not have
bt.from_pydict({"day": ["c"], "v": [7], "extra": ["x"]}) \
  .write.delta(ev, mode="append", merge_schema=True)
bt.read.delta(ev).columns          # ['day', 'v', 'extra']
```

## Maintenance: `compact` and `vacuum`

An incremental writer leaves one small file per commit, and the next write cannot fix that
— it only adds another. Eventually the table costs more to *plan* than to read.

```python
bt.compact(uri, target_size_mb=128, z_order=["region", "day"], where=..., by=None)
ds.write.delta(uri, mode="append", auto_compact=True)   # compact after the commit, if needed
would_delete = bt.vacuum(uri)                           # dry_run=True — deletes nothing
bt.vacuum(uri, dry_run=False, retention_hours=168)
```

On a **transactional** table compaction is itself a transaction: old files are retired
*from the log* but left on storage, so every existing version still reads and time travel
survives. `z_order=` narrows each file's min/max bounds, multiplying what the next query
skips from the log alone. On a **plain directory** there is no log, so it is read →
repartition → write → remove the replaced parts (single-writer only). An existing Hive
layout is carried forward — compaction changes file *sizes*, not how the table is
organized — and `sort_by=[cols]` is the plain-directory answer to `z_order`. A partitioned
directory in a format whose reader cannot recover partition columns from the layout
(anything but Parquet) is refused rather than flattened.

Compaction never deletes. `bt.vacuum` is the only thing that does, and it **defaults to a
dry run** returning the list it *would* remove. The retention window is the safety
argument: a file goes only once it has been unreferenced longer than any reader could still
need it. Shortening it below the table's configured minimum lets an active reader lose its
files mid-scan, so the backend refuses unless you waive the check explicitly. Vacuum
destroys time travel past the window — irreversibly.

## Checklist

- [ ] Table read through `bt.read.delta/.iceberg/.hudi`, never as `format="parquet"`.
- [ ] Upsert by key → `merge_on=` or `merge_into`; replace a known slice → `replace_where`.
      They are not interchangeable.
- [ ] Merge clauses ordered deliberately — first match wins.
- [ ] `when_not_matched_by_source` used knowingly (it rewrites the whole table).
- [ ] CDC feed → `apply_changes` with `sequence_by`, never `type1`.
- [ ] Batches applied in sequence order; a replay over deletes treated as a rebuild.
- [ ] `replace_where` predicate is partition-column equality, and `partition_by=` is
      repeated on the replacing write.
- [ ] `merge_schema=True` only where a new column is genuinely intended.
- [ ] Small files compacted (or `auto_compact=True`); `vacuum` reviewed as a dry run first.
- [ ] Copy-on-write paths (`scd.*`, file `merge`) have exactly one writer.

## See also

- Docs: `docs/user-guide/moving-data/lakehouse.md`; `docs/tutorials/pipelines/building-a-lakehouse.md`;
  `docs/integrations/lakehouse/delta-lake.md`, `docs/integrations/lakehouse/iceberg.md`, `docs/integrations/lakehouse/hudi.md`, `docs/integrations/warehouses/databricks.md`, `docs/integrations/warehouses/snowflake.md`;
  `docs/cookbook/data-engineering/ingest/cdc-pipeline.md`, `docs/cookbook/data-engineering/modeling/slowly-changing-dimensions.md`, `docs/cookbook/data-engineering/maintenance/partition-backfill.md`, `docs/cookbook/data-engineering/ingest/late-arriving-data.md`, `docs/cookbook/data-engineering/maintenance/file-compaction.md`, `docs/cookbook/dataset/cleaning/deduplication.md`.
- Code: `api/merge/{builder,clauses,cdc,execute}.py`; `api/dataset/scd.py`;
  `api/io_namespace/writer.py`; `io/formats/lakehouse/`; `io/catalog.py`.
- Skills: `read-and-write-data` (formats, paths, credentials, schema at the file layer);
  `write-a-batcher-pipeline` (the relational core); `run-a-distributed-job` (cluster
  writes and the mergeable commit); `optimize-a-slow-query` (layout, Z-order, pruning);
  `debug-a-batcher-query` (when a merge or commit misbehaves).
