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

**Databases and warehouses** — `sql` (any ADBC/FlightSQL database, `uri=`), `snowflake`,
`bigquery` (Storage Read API), `databricks` (Unity Catalog credential vending),
`clickhouse`, `mongo`, `cassandra`, `dynamodb`, `elasticsearch`. Pass a query positionally
where the connector takes one; connection details are keywords. These fan out natively
(BigQuery streams, Cassandra token ranges, DynamoDB scan segments) rather than pulling
through one cursor.

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
`ds.write.sql/snowflake/mongo` target databases. Lakehouse sinks and `merge`/`merge_into`
belong to `manage-a-lakehouse-table`.

**Save modes** (`mode=`, Spark `SaveMode` parity) — `"overwrite"` (default), `"error"`
(raise if the path exists), `"ignore"` (skip, return an empty manifest), `"append"`.

```python
ds.write(p, mode="error")     # PlanError: path already exists
ds.write(p, mode="ignore")    # returns a manifest with 0 files
ds.write(p, mode="append")    # PlanError — append is delta/iceberg/hudi/snowflake only
```

`append` is refused on a file sink on purpose: a directory of part files has nothing to
append *to* transactionally. An overwrite is a true replace — files a previous,
differently-shaped write left behind are pruned after the commit, so a 5-file output
overwritten by a 2-file one does not silently union the three stale files back in.

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
later `parquet_dataset` read prunes on — partition by a column you actually filter on, and
one with modest cardinality. `max_rows_per_file=` caps file size (4 rows at 2 → 2 files).
`sort_by=[cols]` clusters rows before writing so each file's min/max bounds are tight and
downstream zonemap/bloom skipping multiplies; bounded batch writes only.
`ds.repartition(num_files=, target_size_mb=, by=)` supplies these as defaults.

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

A corpus at scale always contains a truncated upload or a zero-byte object. `on_error=`
(`io/base/_tolerance.py`) decides whether one bad file kills the read:

```python
bt.read.json(d)                      # ArrowInvalid — the default, "raise", is all-or-nothing
bt.read.json(d, on_error="skip")     # {'a': [1]} — logs a WARNING, drops the file, continues
```

`"skip"` records every dropped path (`source.corrupt_files()`) so the loss is auditable
rather than silent. It is explicit and not the default because dropping data is the right
call across 10,000 files and the wrong one for the single file you just wrote.

**Known gap:** `on_error="skip"` covers the *data* read but not Parquet's footer-based
`row_count()`, which the autotuner calls. A corrupt Parquet file therefore still aborts the
read with `ArrowInvalid` despite the policy. Other formats (JSON, CSV, images) skip
correctly.

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
