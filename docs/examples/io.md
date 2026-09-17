# Reading and writing

This page covers the scripts that move data in and out: the formats, the paths, the write modes, the schema questions that show up at the boundary, and the Delta tables that turn a write into a commit.

## Reading

The path scheme picks the filesystem. Nothing else changes, so a query against object storage is the same query as one against a local file. Cost changes. Every byte read from object storage is a round trip, which is why the projection and the predicate matter more there.

```python
# docs: skip
import batcher as bt

# Anonymous access to the public benchmark corpus; no credentials configured.
region = bt.read.parquet("s3://ray-benchmark-data/tpch/parquet/sf1/region/*.parquet")
assert region.count() == 5
```

Format inference reads the extension off the literal part of a path and stops at the first `*`, so a globbed path has nothing to infer from and needs a typed reader. A `*` matches within one path segment only. Crossing directories in a Hive layout needs `**`.

Two behaviours are worth pinning down before they surprise you. A directory of Parquet files whose schemas disagree reads with the first file's schema by default, `schema_mode="strict"`, and a column added in a later file is dropped with a `DataWarning` naming it. Pass `schema_mode="union"` to reconcile the files' schemas, once you have decided what a missing value means. And a Hive-partitioned directory recovers its partition column from the directory names, through `read.parquet` or the explicit `read.parquet_dataset`. The value is parsed rather than stored, so its type is inferred: a date-shaped string comes back as a date.

## Writing

`mode="append"` works for the transactional sinks, where a commit is a real thing. A plain file sink has no table to add to, so appending would mean rewriting the whole output. The writer refuses rather than doing that silently.

```python
import tempfile
from pathlib import Path

import batcher as bt

data = bt.from_pydict({"id": [1, 2, 3], "name": ["a", "b", "c"]})

with tempfile.TemporaryDirectory() as directory:
    root = Path(directory) / "batches"
    root.mkdir()

    # One file per batch, read back as a single relation.
    data.limit(2).write.parquet(str(root / "batch-0.parquet"))
    data.limit(1, offset=2).write.parquet(str(root / "batch-1.parquet"))

    combined = bt.read.parquet(str(root / "*.parquet"))
    assert combined.count() == 3
```

The other route is a transactional sink, where the same accumulation is a sequence of commits and a replay is a keyed merge rather than a duplicate.

## Lakehouse tables

The Delta scripts are the transactional end of the same boundary: a commit rather than a file, with the history that follows from it.

### A write is a commit

Every write is one transaction, which is what makes a half-finished write invisible to readers. They keep seeing the previous version until the commit lands, so there is no window where a query sees part of a batch.

```python
# docs: skip
import batcher as bt

# Three commits, each a complete snapshot.
orders.limit(1_000).write.delta(table)
orders.limit(500, offset=1_000).write.delta(table, mode="append")
orders.limit(200).write.delta(table, mode="overwrite")

assert bt.read.delta(table).count() == 200
assert bt.read.delta(table, version=0).count() == 1_000
```

Time travel falls out of that log rather than being a backup feature. Each commit adds files rather than replacing them, so an older version is still fully described. That is also why `vacuum` is the destructive operation: it removes the files older versions point at.

### Upserts and change feeds

`merge_on` performs a `MERGE INTO` keyed on the columns you name: matched rows update, the rest insert, in one commit. Doing it as a delete followed by an append is two commits with a window in between where readers see neither version.

That also makes a replay idempotent, which is the mechanism behind exactly-once delivery. An append replayed twice duplicates its rows; a keyed merge replayed twice is a no-op, and `examples/streams/exactly_once_semantics.py` asserts both.

### Maintenance

An incremental writer leaves one small file per commit, and the next write cannot fix that. Eventually the table costs more to plan than to read. Compaction bin-packs the files in a transaction that never deletes anything an older version still references, so every version stays readable.

## Verifying

A write that reports success and a file that holds the right rows are two different claims. Read the output back. Comparing row count, schema and a control total against the source is the only version of "the job worked" that means anything, and `examples/io/write_and_verify.py` does exactly that.

## Every script on this page

The table below lists the IO and lakehouse scripts in path order.

<!-- library-table: io,lakehouse -->
| Script | Shows |
| --- | --- |
| `examples/io/arrow_interop.py` | Moving data in and out of other frameworks, zero-copy where possible |
| `examples/io/arrow_ipc.py` | Arrow IPC: the format with no conversion cost |
| `examples/io/binary_and_blobs.py` | Carrying binary payloads through a pipeline without materializing them |
| `examples/io/cloud_paths.py` | Cloud paths: the schemes, the globs, and what format inference can see |
| `examples/io/compression_tradeoffs.py` | Compression codecs measured on real data |
| `examples/io/csv_error_tolerance.py` | Reading text that is not entirely well-formed |
| `examples/io/csv_from_s3.py` | Reading delimited text from S3, including files that carry no header |
| `examples/io/csv_roundtrip_and_options.py` | Writing and re-reading CSV, and the fidelity you lose on the way |
| `examples/io/dataframe_interop.py` | Handing data to and from pandas, Polars, NumPy and Arrow |
| `examples/io/delta_roundtrip.py` | Delta Lake: transactional appends and overwrites over a real table |
| `examples/io/delta_time_travel.py` | Reading an earlier version of a Delta table |
| `examples/io/format_round_trip_matrix.py` | Every writable format, round-tripped and compared |
| `examples/io/globs_and_multiple_files.py` | Reading many files as one dataset, and what the glob can and cannot cross |
| `examples/io/images_from_s3.py` | Reading real images from object storage as a table of bytes |
| `examples/io/json_and_ndjson.py` | JSON on the way out and back, and why newline-delimited is the one to write |
| `examples/io/lance_and_msgpack.py` | Two less common formats: Lance for vectors, MessagePack for interchange |
| `examples/io/numpy_arrays.py` | Reading a NumPy array file as a Dataset |
| `examples/io/orc_and_avro.py` | ORC and Avro: the two formats you meet in someone else's warehouse |
| `examples/io/parquet_from_s3.py` | Reading Parquet straight from S3, with no download step |
| `examples/io/parquet_pushdown.py` | Projection and predicate pushdown: reading less of a file, not filtering after it |
| `examples/io/parquet_roundtrip.py` | Writing and reading Parquet, with partitioning and column pruning |
| `examples/io/parquet_write_options.py` | Writing Parquet: choosing a compression codec and reading the file back |
| `examples/io/partitioned_writes.py` | Partitioned output: writing a directory tree a reader can prune |
| `examples/io/reading_a_directory.py` | Reading a directory of files as one relation, and controlling what is included |
| `examples/io/reading_from_memory.py` | Constructing a Dataset from data already in the process |
| `examples/io/reading_with_a_declared_schema.py` | Declaring the schema instead of letting the reader infer it |
| `examples/io/save_modes.py` | Save modes and write manifests: what happens when the target already exists |
| `examples/io/save_modes_and_transactional_append.py` | Save modes: overwrite, and why a plain file sink has no append |
| `examples/io/schema_evolution.py` | Files whose schemas disagree: what the reader does, and what you must do |
| `examples/io/sources_and_sinks.py` | The source and sink registries: what formats exist, and the objects behind them |
| `examples/io/sql_database.py` | Reading from a SQL database with a connection URI |
| `examples/io/streaming_reads.py` | Reading in bounded memory: iter_batches, limits, and lazy metadata |
| `examples/io/streaming_reads_iter_batches.py` | Reading a large result without materializing it: iter_batches |
| `examples/io/text_and_binary.py` | The two untyped readers: whole-file bytes and line-by-line text |
| `examples/io/text_formats.py` | CSV, JSON, and Arrow IPC round trips |
| `examples/io/write_and_verify.py` | Writing a result and proving what landed on disk |
| `examples/io/write_modes_and_atomicity.py` | A write that either lands completely or not at all |
| `examples/io/writing_partitioned_reports.py` | Writing a report partitioned by a business key, and reading one partition back |
| `examples/io/xml_and_excel.py` | Two formats that arrive from outside engineering: XML and Excel |
| `examples/lakehouse/change_data_capture.py` | Applying a change feed: inserts, updates and deletes in one commit |
| `examples/lakehouse/compaction.py` | The small-files problem, and compacting a table that has it |
| `examples/lakehouse/delta_upserts.py` | MERGE INTO: upserting keyed rows into a Delta table |
| `examples/lakehouse/partition_backfill.py` | Replacing one partition without touching the rest |
| `examples/lakehouse/scd_type_two.py` | Slowly changing dimensions: keeping the history of a changed row |
| `examples/lakehouse/schema_evolution_on_write.py` | Adding a column to a table that already has data |
| `examples/lakehouse/snapshot_isolation.py` | Snapshot isolation: a reader sees one version, whatever the writer is doing |
| `examples/lakehouse/table_maintenance.py` | Table maintenance: compaction, vacuum, and the version they cost you |
<!-- /library-table -->

## See also

- {doc}`/cookbook/io/index`: six IO recipes with the whole script on the page.
- {doc}`/user-guide/moving-data/reading-data` and {doc}`/user-guide/moving-data/writing-data`: the reader and writer guides.
- {doc}`/user-guide/moving-data/lakehouse`: Delta, Iceberg, and Hudi tables in depth.
- {doc}`/user-guide/moving-data/cloud-storage`: paths, credentials, and object-store cost.
