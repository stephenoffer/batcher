---
name: read-and-write-data
description: Get data into and out of Batcher — the bt.read source families (structured files, text/binary, multimodal, databases, warehouses, streams), the ds.write sinks with save modes/partitioning/atomic manifests/resume, cloud paths and credentials, globs and partition pruning, schema inference and cross-file evolution, and per-file error tolerance. Invoke when choosing a reader or writer, wiring up object storage, or debugging a format/schema/path/credential problem at the IO boundary.
---

# Read and write data

The IO boundary: how bytes become a `Dataset` and how a `Dataset` becomes bytes.
Relational basics (lazy plans, expressions, joins, `group_by`) are **not** repeated here —
read `write-a-batcher-pipeline` for those. Lakehouse *table* lifecycle (Delta/Iceberg/Hudi,
MERGE, SCD, time travel, compaction) is `manage-a-lakehouse-table`.

The surface is large — 44 readers on `bt.read`, 20 sinks on `ds.write`. Verify before you
cite: `python -c "import batcher as bt; print([m for m in dir(bt.read) if m[0] != '_'])"`.

## Every read is lazy; pushdown is why there is no `scan_*`

`bt.read.parquet(...)` opens metadata to learn the schema and **reads no data**. The
`Dataset` it returns is a plan handle. Kyber then pushes projection and predicates *into*
the scan, so the file layer only touches the columns and row groups the query needs:

```python
e = bt.read.parquet(big).filter(bt.col("x") > 1).select("y").explain()
# project    est≈1 (default)
#   filter   est≈1 (default)
#     scan   est≈3 (exact)
```

That is why Batcher has no `scan_parquet`/`read_parquet` split the way Polars does: there
is one entry point and it is always the lazy one. Two consequences worth internalizing:

- **Filter early and `select` last.** Both push back into the scan. Wrapping a column in a
  function before comparing it (`bt.col("d").str.slice(0,4) == "2024"`) blocks the push.
- **`ds.schema` / `ds.columns` / `ds.dtypes` cost a metadata read, not a scan.** Cheap to
  inspect before committing to a query.

## The read surface, by family

**Structured files** — `parquet`, `parquet_dataset`, `csv`, `json` (newline-delimited),
`orc`, `arrow` (Feather/IPC), `avro`, `lance`, `excel`, `xml`. All take a file, a
directory, or a glob. Use `parquet_dataset` — not `parquet` — for a **Hive-partitioned
directory**; it recovers the partition columns from the directory names and prunes on them.

```python
ds.write(root, format="parquet", mode="overwrite", partition_by=["region"])
back = bt.read.parquet_dataset(root)                    # columns: id, amount, region
back.filter(bt.col("region") == "us").select("id", "amount").sort("id").to_pydict()
# {'id': [1, 3], 'amount': [10.0, 30.0]}   ← the eu directory is never opened
```

**Semi-/unstructured** — `text` (one row per line: `path`, `line_number`, `text`),
`binary` (whole files as `uri`/`bytes`/`size`/`mime` — the entry point for custom
decoding), `logs` (`pattern=` for grok capture), `documents` (PDF), `numpy`, `tfrecord`,
`webdataset`, `hdf5`, `zarr`.

```python
bt.read.text(p).to_pydict()      # {'path': [...], 'line_number': [1, 2], 'text': ['hello', 'world']}
bt.read.binary(p).select("size", "mime").to_pydict()   # {'size': [3], 'mime': ['application/octet-stream']}
```

**Multimodal** — `images`, `audio`, `video`, `point_cloud`. These **list and describe by
default**; decoding is opt-in and costs an optional extra:
`bt.read.images(path, decode=True, size=(224, 224))` appends an `image` uint8 tensor,
`audio(decode=True, sample_rate=16000)` appends `waveform`, `video(decode=True,
num_frames=16)` appends `frames`. `point_cloud` needs no extra: every point of a
`.pcd`/`.ply`/`.bin` sweep becomes a row plus a `frame` column. For what to *do* with
these, see `build-an-ml-pipeline`.

**Databases and warehouses** — `sql` (any database from one `uri=`), `snowflake`,
`bigquery` (Storage Read API), `databricks` (Unity Catalog credential vending),
`clickhouse`, `mongo`, `cassandra`, `dynamodb`, `elasticsearch`, `redis`, `hbase`. Pass a
query positionally where the connector takes one; connection details are keywords. These
fan out natively (BigQuery streams, Cassandra token ranges, DynamoDB scan segments) rather
than pulling through one cursor.

`bt.read.sql(query, uri=...)` picks its own backend from the scheme *and* from what is
installed: ADBC where an Arrow-native driver is present, ConnectorX where it is, and the
scheme's PEP 249 driver otherwise. `module=`/`connection=` name a PEP 249 driver directly.

A filter that **pins the partition key** stops the fan-out rather than filtering after it:
DynamoDB issues one `Query` instead of N `Scan` segments (which are billed per item
examined, filter or no filter), and Cassandra reads one partition instead of 64 token
ranges with `ALLOW FILTERING`. It needs a top-level `key == literal` term; an `OR`, a range
on the key, or a composite key only partly pinned falls back to the scan. The projection,
the predicate, a row cap and a top-N all push into the submitted SQL, the last two only
where the dialect is known — a cap the server cannot parse is a syntax error, and a top-N
without a `NULLS` clause returns the *wrong rows*.

**Streaming sources** — `kafka`, `kinesis`, `pulsar`, `pubsub`, `eventhubs`, `socket`,
`rate`, and `files_incremental` (an Auto Loader analog: tracks already-seen files across
runs). These return **unbounded** datasets — `ds.is_streaming` is True and `collect()`
will not work. Triggers, output modes, checkpoints, and watermarks are out of scope here —
that is the whole of `write-a-streaming-pipeline`.

**Escape hatch** — `bt.read.table("<registry-name>", *args, **opts)` reaches any registered
source the typed methods do not wrap.

## Paths, globs, and format inference

`io/detect.py` resolves a format in this order: an explicit `format=` → the URI scheme
(`delta://`) → a table marker directory at the path (`_delta_log` / `metadata` /
`.hoodie`) → the file extension. That last step is why **the bare `bt.read(path)` callable
needs a plain file path**:

```python
bt.read("out/one.csv").count()             # 4    — extension inferred
bt.read("out/*.parquet")                   # FormatError: could not infer a format
bt.read("out/", format="parquet")          # fine — name it explicitly
bt.read.parquet("out/**/*.parquet")        # fine — the typed method never guesses
```

Prefer the typed `bt.read.<format>` method whenever the path is a directory or a glob.

Globs are `fnmatch` over an object-store listing, and **`*` does not cross a directory
boundary — use `**` for a recursive match** (`events/**/*.parquet`, not
`events/*/*.parquet`, which matches nothing).

**Cloud storage** is the same API with a different scheme: `s3://`/`s3a://`, `gs://`/
`gcs://`, `az://`/`abfs://`/`abfss://`/`wasb[s]://`, `hdfs://`, `file://`. One
`pyarrow.fs` façade (`io/filesystem.py`) backs all of them, with an fsspec fallback behind
the same interface, so nothing above the IO layer knows which cloud it is on. Needs
`pip install 'batcher-engine[cloud]'`.

Credentials come from the standard provider environment variables — `AWS_ACCESS_KEY_ID` /
`AWS_SECRET_ACCESS_KEY` / `AWS_REGION` (instance roles picked up automatically),
`GOOGLE_APPLICATION_CREDENTIALS`, `AZURE_STORAGE_ACCOUNT_NAME` plus a key/SAS/AAD
principal. Self-hosted S3 (MinIO, Ceph) is an endpoint override, either process-wide via
`AWS_ENDPOINT_URL` or per path:
`bt.read.parquet("s3://b/d/*.parquet?endpoint_override=https://minio.internal:9000")`.
Databricks is the exception that is *not* ambient: `io/credentials.py` vends short-lived,
table-scoped Unity Catalog credentials so an Arrow-native reader touches the Parquet
directly with no Spark in the path.

## The write surface

`ds.write(path, format=None, *, mode=, partition_by=, resume=, max_rows_per_file=,
sort_by=, replace_where=, distributed=, num_workers=, **opts)` is the general form;
`ds.write.parquet/csv/json/orc/arrow/avro/lance/msgpack` are typed wrappers, and
`ds.write.sql/snowflake/mongo/dynamodb/cassandra/redis/elasticsearch/hbase` target
databases and operational stores. Lakehouse sinks and `merge`/`merge_into` belong to
`manage-a-lakehouse-table`.

**Save modes** (`mode=`, Spark `SaveMode` parity) — `"overwrite"` (default), `"error"`
(raise if the path exists), `"ignore"` (skip, return an empty manifest), `"append"`, and
`"overwrite_partitions"` (replace only the partitions the new data covers).

```python
ds.write(p, mode="error")     # PlanError: path already exists
ds.write(p, mode="ignore")    # returns a manifest with 0 files
ds.write(p, mode="append")    # PlanError for a file sink; fine for a table sink
ds.write(p, mode="overwrite_partitions", partition_by=["dt"])  # other dt= dirs survive
```

**Row-level modes** are a *different vocabulary*, accepted only by the database and
operational-store sinks, and they are not save modes: `upsert`, `update`, `delete` and
`delete_insert` change only the rows whose keys match and leave every other row alone.
Spark's JDBC writer has no spelling for any of them.

```python
ds.write.sql("orders", uri="mysql://db/shop", mode="upsert", key_columns="order_id")
ds.write.sql("sessions", uri="postgresql://db/app", mode="delete", key_columns="sid")
ds.write.dynamodb("scores", region_name="us-east-1")          # upsert is the default
```

Pass `key_columns=` on the write that *creates* the table too. An upsert conflicts on the
target's `PRIMARY KEY`, and a table created by an earlier keyless `append` has none —
PostgreSQL and SQLite then raise, and **MySQL silently duplicates every row**. A mode a
store cannot express is refused by name rather than approximated (DynamoDB and Cassandra
have no `append`, because their put/insert *is* an upsert). One `write` call is one
transaction; a distributed write is one per shard, and `overwrite` is refused past the
first shard because every shard would discard the ones before it. Full detail:
`docs/integrations/databases/writing.md`.

`overwrite_partitions` is Spark's `partitionOverwriteMode="dynamic"` and Hive's
`INSERT OVERWRITE` (both spellings resolve to it). Reach for it on **any** reload of part
of a partitioned table: `mode="overwrite"` there deletes every partition the new batch
does not mention, silently and at full speed. It needs `partition_by`; on a transactional
table use `replace_where` instead, which scopes the same intent inside one commit.

`append` is refused on a file sink on purpose: a directory of part files has nothing to
append *to* transactionally. An overwrite is a true replace — files a previous,
differently-shaped write left behind are pruned after the commit, so a 5-file output
overwritten by a 2-file one does not silently union the three stale files back in.

**Knowing a write finished.** Per-file atomicity does not make a *directory* safe: a run
that dies partway leaves valid files that read back silently short. A directory write ends
by publishing an empty `_SUCCESS` marker, only once every shard is accounted for, and
`bt.read(path, require_success=True)` refuses a directory that has none. Off by default —
a directory Batcher did not write has no marker and is not thereby incomplete — so reach
for it on a path *another* job produces.

**Atomicity and the manifest.** Every write returns a `WriteManifest` (`io/manifest.py`) of
`WrittenFile` records — `path`, `rows`, `bytes`, `partition_values`, and per-column
`stats` (min/max/null counts collected while the data was already in memory). Files are
staged and published atomically; a reader never sees a half-written output. The manifest is
also what makes a **distributed** write mergeable: each worker returns its `WrittenFile`s
and the driver concatenates them (a commutative merge) into one commit.

```python
m = ds.write(root, format="parquet", partition_by=["region"])
len(m.files)                        # 2
sum(f.rows for f in m.files)        # 4
m.files[0].partition_values         # {'region': 'us'}
```

**Layout knobs.** `partition_by=["region"]` writes Hive `region=us/` directories that a
later read prunes on (`bt.read.parquet(dir)` routes a Hive tree to the partition-aware
reader by itself, so the partition columns come back). A key may be an **expression** with
an `.alias(...)` — `partition_by=[bt.col("ts").dt.year().alias("year")]` — which is how a
partition transform (Iceberg `days()`/`bucket()`) is spelled — partition by a column you actually filter on, and
one with modest cardinality. `max_rows_per_file=` caps file size (4 rows at 2 → 2 files).
`sort_by=[cols]` clusters rows before writing so each file's min/max bounds are tight and
downstream zonemap/bloom skipping multiplies; bounded batch writes only.
`ds.repartition(num_files=, target_size_mb=, by=)` supplies these as defaults.

All three sizing knobs reach the **workers** on a distributed write (the layout travels and
is resolved against each shard, since a streaming distributed write never materializes on
the driver), and `max_rows_per_file` over a lazy source *streams*, rolling over to a new
file as each fills — so capping file size no longer costs a full materialization. A
partitioned write whose plan has a breaker cuts its shards by partition key, so it emits
one file per partition rather than one per partition per worker; a breaker-free one cannot
(nothing reaches the driver), and `sort_by=` the partition columns is the lever there.

**`resume=True`** skips already-present (therefore fully committed) output files, so a job
re-run after a crash or spot preemption finishes only the unwritten shards. It is
**exactly-once only on a deterministic plan** — read → `map_batches`/`filter`/`select` →
write. The engine enforces this rather than trusting you:

```python
ds.write(p, format="parquet", resume=True)     # twice → still 4 rows
ds.group_by("region").agg(n=bt.count()).write(p2, format="parquet", resume=True)
# PlanError: resume=True is exactly-once only on a deterministic plan ...
```

A shuffle makes the row→file assignment vary between runs, so resuming could skip a file
that now holds different rows. Write to a fresh path, or materialize a keyed intermediate.

## Schema: inference, evolution, widening

A file source infers its schema from metadata once and caches it. In the default
`schema_mode="strict"`, **file 0's schema stands for every file** — which is exactly wrong
for a directory written over months. Pass a reconciliation mode (`io/schema/evolution.py`):

```python
bt.read.parquet(d)                         # ArrowInvalid: schema at index 1 was different
bt.read.parquet(d, schema_mode="union")    # {'a': [1.0, 2.5], 'b': [None, 'x']}
bt.read.parquet(d, schema_mode="latest")   # ['a', 'b'] — last file's schema wins
```

- `"strict"` — every file must match the first; anything else raises.
- `"union"` — union of columns (first-seen order, new columns appended), each promoted to
  the common non-lossy supertype. Missing columns read as typed nulls. Promotion recurses
  into nested types, so `list<int32>` + `list<int64>` → `list<int64>` and
  `struct<a>` + `struct<a,b>` → `struct<a,b>`.
- `"latest"` — the last schema wins; older files are cast toward it.

An unbridgeable collision (int vs string, mismatched list kinds) raises `SchemaError`
rather than picking a lossy winner. `schema_drift(inferred, expected)` reports
added/removed/retyped columns when you want to *gate* on drift instead of absorbing it.

**Narrow types widen at the FFI boundary**, once: Int8/16/32 → Int64, Float16/32 → Float64.
Don't re-implement coercion upstream. Note the asymmetry — the plan-time schema reports the
*source* types, the materialized result is widened:

```python
ds.dtypes                # [DataType(int32), DataType(float)]   ← the file's types
ds.collect().schema      # i: int64,  f: double                 ← what you actually get
```

## Error tolerance

Two failures look alike here and take **opposite** flags. Reaching for the wrong one is how
a job silently returns a fraction of its corpus, so decide which you have before you type.

**An unreadable file** — a truncated upload, a zero-byte object, a JPEG with no trailer.
`on_error=` (`io/base/_tolerance.py`) decides whether one of them kills the read:

```python
bt.read.json(d)                      # ArrowInvalid — the default, "raise", is all-or-nothing
bt.read.json(d, on_error="skip")     # {'a': [1]} — logs a WARNING, drops the file, continues
```

`"skip"` records every dropped path (`source.corrupt_files()`) so the loss is auditable
rather than silent. It is explicit and not the default because dropping data is the right
call across 10,000 files and the wrong one for the single file you just wrote. It covers
the metadata probes too, including Parquet's footer `row_count()`.

**A malformed record inside a readable file** — a CSV row with a field the header lacks, an
NDJSON line that is not JSON. `on_error` is the *wrong* flag: it drops the whole file, so
one stray line discards every good row in it. `on_bad_lines=` (`io/base/_bad_rows.py`)
drops the record instead:

```python
bt.read.csv(p,  on_bad_lines="skip")   # drop it silently, counted as malformed_rows_total
bt.read.json(p, on_bad_lines="warn")   # drop it and log the offending text
bt.read.csv(p)                         # "error" — the default, and Spark's FAILFAST
```

Both readers carry the flag into byte-range splits, so a distributed read returns what the
single-node read returns. Dropped records are counted on the event bus (`events.MALFORMED`)
and exported as `malformed_rows_total`, kept apart from `skipped_total`, which counts whole
files — a total mixing rows with files answers neither question.

**Neither flag touches a value that will not convert** (an `"N/A"` in a column inference
typed `int64`). That is inference having been shown too little, and its fix is `schema=`.
`on_bad_lines` deliberately refuses to delete such a record, because dropping it removes the
rows that were about to tell you the type is wrong.

## Checklist

- [ ] Typed reader (`bt.read.parquet`) for any directory or glob; bare `bt.read` only for
      a plain file path with a known extension.
- [ ] Hive-partitioned directory read with `parquet_dataset`, not `parquet`.
- [ ] `**` (not `*`) for a recursive glob.
- [ ] `filter` before `select`, `select` last — both push into the scan.
- [ ] `mode=` stated explicitly; `append` only where a transactional sink backs it.
- [ ] `partition_by` on a column that is actually filtered on, with sane cardinality.
- [ ] `resume=True` only on a deterministic (breaker-free) plan.
- [ ] `schema_mode="union"` whenever files were written across schema changes.
- [ ] `on_error="skip"` on a large multi-file corpus — and its skipped list is inspected.
- [ ] `on_bad_lines="skip"` where the *records* are untrusted (a CSV/NDJSON export from a
      producer you do not control) — `on_error` is not a substitute and costs the file.
- [ ] Credentials come from the environment, never hardcoded in a path.

## See also

- Docs: `docs/user-guide/moving-data/reading-data.md`, `docs/user-guide/moving-data/writing-data.md`, `docs/user-guide/moving-data/cloud-storage.md`, `docs/user-guide/transform/columns/type-system.md`, `docs/user-guide/moving-data/streaming.md`, `docs/user-guide/moving-data/custom-connectors.md`, `docs/user-guide/trust/data-quality.md`; `docs/integrations/` (per-connector setup);
  `docs/cookbook/data-engineering/modeling/schema-evolution.md`, `docs/cookbook/data-engineering/ingest/incremental-ingest.md`, `docs/cookbook/data-engineering/maintenance/file-compaction.md`.
- Code: `python/batcher/api/io_namespace/{reader,writer}.py`; `io/{detect,filesystem,
  credentials,manifest}.py`; `io/schema/evolution.py`; `io/base/_tolerance.py`.
- Skills: `write-a-batcher-pipeline` (the relational core); `manage-a-lakehouse-table`
  (Delta/Iceberg/Hudi, MERGE, SCD, time travel, compaction); `build-an-ml-pipeline`
  (multimodal decode and inference); `write-a-streaming-pipeline` (unbounded sources,
  triggers, checkpoints); `add-an-io-format-or-connector` (adding a reader);
  `validate-data-quality` (gating on what you read); `optimize-a-slow-query` (file layout
  and pushdown as a performance lever).
