# Reading and writing

This page covers the scripts that move data in and out: the formats, the paths, the write
modes, and the schema questions that show up at the boundary.

## Reading

The path scheme picks the filesystem and nothing else changes, so a query against object
storage is the same query as one against a local file. What does change is that every byte
costs a round trip, which is why the projection and the predicate matter more there.

```python
# docs: skip
import batcher as bt

# Anonymous access to the public benchmark corpus; no credentials configured.
region = bt.read.parquet("s3://ray-benchmark-data/tpch/parquet/sf1/region/*.parquet")
assert region.count() == 5
```

Format inference reads the extension off the literal part of a path and stops at the first
`*`, so a globbed path has nothing to infer from and needs a typed reader. A `*` also matches
within one path segment only, so crossing directories in a Hive layout needs `**`.

Two behaviours are worth pinning down before they surprise you. A directory of Parquet files
whose schemas disagree takes the first file's schema and silently drops the later columns, so
read the generations separately and union them once you have decided what a missing value
means. And a partitioned directory needs `read.parquet_dataset` rather than `read.parquet`,
because the partition value lives in the directory name rather than in the files.

## Writing

`mode="append"` works for the transactional sinks, where a commit is a real thing. A plain
file sink has no table to add to, so appending would mean rewriting the whole output and the
writer refuses rather than doing that silently.

```python
import tempfile
from pathlib import Path

import batcher as bt

data = bt.from_pydict({"id": [1, 2, 3], "name": ["a", "b", "c"]})

with tempfile.TemporaryDirectory() as directory:
    root = Path(directory) / "batches"
    root.mkdir()

    # One file per batch, read back as a single relation.
    data.head(2).write.parquet(str(root / "batch-0.parquet"))
    data.slice(2, 1).write.parquet(str(root / "batch-1.parquet"))

    combined = bt.read.parquet(str(root / "*.parquet"))
    assert combined.count() == 3
```

The other route is a transactional sink, where the same accumulation is a sequence of
commits and a replay is a keyed merge rather than a duplicate.

## Verifying

A write that reports success and a file that holds the right rows are two different claims.
Reading the output back and comparing row count, schema and a control total against the
source is the only version of "the job worked" that means anything, and
`examples/io/write_and_verify.py` does exactly that.

## Every script on this page

The table below lists the IO scripts in path order.

<!-- library-table: io -->
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
<!-- /library-table -->
