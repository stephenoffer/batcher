# Reading and writing

This page lists every reader and writer, then the connector types behind them. For the transformations that sit between a read and a write, see {doc}`Dataset </api/relational/dataset>`.

Readers hang off {py:obj}`bt.read <batcher.read>` and return a lazy {py:class}`Dataset <batcher.Dataset>`. Writers hang off {py:obj}`ds.write <batcher.Dataset.write>` and are terminal, so they execute the plan and return a `WriteManifest`. {py:obj}`bt.read(path, format=None, **opts) <batcher.read>` infers the format from the path, and the dedicated readers below are explicit. Some connectors need an optional dependency. The "Extra" column gives the install name for `pip install 'batcher-engine[<extra>]'`.

## Readers

The readers are grouped by the kind of system they pull from. Within each group they're ordered by how often you'll reach for them.

### Files

These read one file, a directory, or a glob from local disk or object storage:

| Reader | Reads | Extra |
| --- | --- | --- |
| {py:meth}`bt.read.parquet(path) <batcher.api.io_namespace.reader.Reader.parquet>` | a Parquet file, directory, or glob | |
| {py:meth}`bt.read.parquet_dataset(path) <batcher.api.io_namespace.reader.Reader.parquet_dataset>` | a (Hive-)partitioned Parquet dataset directory | |
| {py:meth}`bt.read.csv(path) <batcher.api.io_namespace.reader.Reader.csv>` | a CSV file, directory, or glob | |
| {py:meth}`bt.read.json(path) <batcher.api.io_namespace.reader.Reader.json>` | newline-delimited JSON | |
| {py:meth}`bt.read.orc(path) <batcher.api.io_namespace.reader.Reader.orc>` | ORC file(s) | |
| {py:meth}`bt.read.arrow(path) <batcher.api.io_namespace.reader.Reader.arrow>` | Arrow/Feather IPC file(s) | |
| {py:meth}`bt.read.avro(path) <batcher.api.io_namespace.reader.Reader.avro>` | Avro file(s) | `avro` |
| {py:meth}`bt.read.excel(path) <batcher.api.io_namespace.reader.Reader.excel>` | Excel workbook(s) | `excel` |
| {py:meth}`bt.read.fasta(path) <batcher.api.io_namespace.reader.Reader.fasta>` | FASTA file(s) as `{id, description, sequence}`; `.fa`/`.faa`/`.fna`/`.ffn` too | |
| {py:meth}`bt.read.fastq(path) <batcher.api.io_namespace.reader.Reader.fastq>` | FASTQ read(s) as `{id, description, sequence, quality}`; `.fq` too | |
| {py:meth}`bt.read.bed(path) <batcher.api.io_namespace.reader.Reader.bed>` | BED intervals, columns named for the file's width (0-based, half-open) | |
| {py:meth}`bt.read.gff(path) <batcher.api.io_namespace.reader.Reader.gff>` | GFF3 / GTF annotations, nine columns (1-based, inclusive) | |
| {py:meth}`bt.read.vcf(path) <batcher.api.io_namespace.reader.Reader.vcf>` | VCF variants, sample columns named by the file's header | |
| {py:meth}`bt.read.xml(path) <batcher.api.io_namespace.reader.Reader.xml>` | XML file(s) | `xml` |
| {py:meth}`bt.read.text(path, mode="line") <batcher.api.io_namespace.reader.Reader.text>` | text file(s) as rows (`mode="line"` or `"file"`) | |
| {py:meth}`bt.read.binary(path) <batcher.api.io_namespace.reader.Reader.binary>` | whole files as `{uri, bytes, size, mime}` rows | |
| {py:meth}`bt.read.warc(path) <batcher.api.io_namespace.reader.Reader.warc>` | web-archive (WARC) file(s), one row per record; `.warc.gz` read transparently | |
| {py:meth}`bt.read.numpy(path) <batcher.api.io_namespace.reader.Reader.numpy>` | NumPy `.npy` / `.npz` file(s) | |
| {py:meth}`bt.read.hdf5(path) <batcher.api.io_namespace.reader.Reader.hdf5>` | HDF5 file(s) | `hdf5` |
| {py:meth}`bt.read.zarr(path) <batcher.api.io_namespace.reader.Reader.zarr>` | a Zarr store | `zarr` |
| {py:meth}`bt.read.logs(path, pattern=None) <batcher.api.io_namespace.reader.Reader.logs>` | line-delimited logs; `pattern=` for grok extraction | |
| {py:meth}`bt.read.files_incremental(path) <batcher.api.io_namespace.reader.Reader.files_incremental>` | incrementally discover new files under `path` | |
| {py:meth}`bt.read.table(name) <batcher.api.io_namespace.reader.Reader.table>` | any registered non-file source by name (escape hatch) | |

### Lakehouse tables

These read a transactional table through its metadata layer, so a read sees one consistent snapshot:

| Reader | Reads | Extra |
| --- | --- | --- |
| {py:meth}`bt.read.delta(path, version=, timestamp=) <batcher.api.io_namespace.reader.Reader.delta>` | a Delta Lake table (time travel) | |
| {py:meth}`bt.read.iceberg(table, catalog=, snapshot_id=) <batcher.api.io_namespace.reader.Reader.iceberg>` | an Iceberg table | |
| {py:meth}`bt.read.hudi(path) <batcher.api.io_namespace.reader.Reader.hudi>` | an Apache Hudi table (read-only) | |
| {py:meth}`bt.read.lance(path) <batcher.api.io_namespace.reader.Reader.lance>` | a Lance dataset | `lance` |
| {py:meth}`bt.read.databricks(table) <batcher.api.io_namespace.reader.Reader.databricks>` | a Databricks / Unity Catalog table (→ Delta) | |
| {py:meth}`bt.read.delta_sharing(url) <batcher.api.io_namespace.reader.Reader.delta_sharing>` | a Delta Sharing table by profile URL | |

### Warehouses and databases

These submit a query to an external engine and stream the Arrow result back:

| Reader | Reads |
| --- | --- |
| {py:meth}`bt.read.sql(query, uri=) <batcher.api.io_namespace.reader.Reader.sql>` | ADBC / FlightSQL in a single submission (or `table=` for a whole table) |
| {py:meth}`bt.read.snowflake(query, connection_kwargs=) <batcher.api.io_namespace.reader.Reader.snowflake>` | a Snowflake query (parallel result-chunk fetch) |
| {py:meth}`bt.read.bigquery(...) <batcher.api.io_namespace.reader.Reader.bigquery>` | BigQuery via the Storage Read API (parallel Arrow streams) |
| {py:meth}`bt.read.clickhouse(query) <batcher.api.io_namespace.reader.Reader.clickhouse>` | a ClickHouse query (Arrow-native) |

### NoSQL

Each of these splits the keyspace so the collection reads in parallel:

| Reader | Reads |
| --- | --- |
| {py:meth}`bt.read.mongo(...) <batcher.api.io_namespace.reader.Reader.mongo>` | a MongoDB collection (Arrow-native via pymongoarrow) |
| {py:meth}`bt.read.cassandra(...) <batcher.api.io_namespace.reader.Reader.cassandra>` | Cassandra / Scylla via token-range splits |
| {py:meth}`bt.read.dynamodb(...) <batcher.api.io_namespace.reader.Reader.dynamodb>` | DynamoDB via native parallel scan segments |
| {py:meth}`bt.read.elasticsearch(...) <batcher.api.io_namespace.reader.Reader.elasticsearch>` | Elasticsearch via ES\|QL Arrow / sliced scroll |

### Streaming

These return an unbounded `Dataset`. See {doc}`streaming </user-guide/moving-data/streaming>` for triggers and checkpoints.

| Reader | Reads |
| --- | --- |
| {py:meth}`bt.read.kafka(topic) <batcher.api.io_namespace.reader.Reader.kafka>` | a Kafka topic as an unbounded streaming source |
| {py:meth}`bt.read.kinesis(stream_name) <batcher.api.io_namespace.reader.Reader.kinesis>` | an AWS Kinesis stream as an unbounded source |
| {py:meth}`bt.read.pulsar(topic) <batcher.api.io_namespace.reader.Reader.pulsar>` | an Apache Pulsar topic as an unbounded source |
| {py:meth}`bt.read.pubsub(subscription) <batcher.api.io_namespace.reader.Reader.pubsub>` | a Google Cloud Pub/Sub subscription as an unbounded source |
| {py:meth}`bt.read.eventhubs(hub) <batcher.api.io_namespace.reader.Reader.eventhubs>` | an Azure Event Hubs stream as an unbounded source |

### Multimodal and ML formats

These read media and document files as rows of bytes plus metadata, decoding only when you ask:

| Reader | Reads | Extra |
| --- | --- | --- |
| {py:meth}`bt.read.images(path, decode=False) <batcher.api.io_namespace.reader.Reader.images>` | images (uri/bytes/size/mime + header meta) | `image` |
| {py:meth}`bt.read.audio(path, decode=False) <batcher.api.io_namespace.reader.Reader.audio>` | audio files (+ `waveform` when decoded) | `audio` |
| {py:meth}`bt.read.video(path, decode=False) <batcher.api.io_namespace.reader.Reader.video>` | video files (+ frames when decoded) | `video` |
| {py:meth}`bt.read.documents(path) <batcher.api.io_namespace.reader.Reader.documents>` | PDF document(s) as text rows | `pdf` |
| {py:meth}`bt.read.webdataset(path) <batcher.api.io_namespace.reader.Reader.webdataset>` | WebDataset `.tar` shard(s) | |
| {py:meth}`bt.read.training_shards(path) <batcher.api.io_namespace.reader.Reader.training_shards>` | a training corpus written by {py:meth}`ds.ml.write_shards <batcher.api.dataset.ml.DatasetML.write_shards>` | |

## Writers

`ds.write(path, fmt=None, ...)` infers the format, and the dedicated writers are explicit. Each executes the plan and returns a `WriteManifest`.

### Files

These write one file per output partition:

| Writer | Writes | Extra |
| --- | --- | --- |
| {py:meth}`ds.write.parquet(path, compression="zstd") <batcher.api.io_namespace.writer.Writer.parquet>` | Parquet | |
| {py:meth}`ds.write.csv(path) <batcher.api.io_namespace.writer.Writer.csv>` | CSV | |
| {py:meth}`ds.write.json(path) <batcher.api.io_namespace.writer.Writer.json>` | newline-delimited JSON | |
| {py:meth}`ds.write.orc(path) <batcher.api.io_namespace.writer.Writer.orc>` | ORC | |
| {py:meth}`ds.write.arrow(path) <batcher.api.io_namespace.writer.Writer.arrow>` | Arrow/Feather IPC | |
| {py:meth}`ds.write.avro(path) <batcher.api.io_namespace.writer.Writer.avro>` | Avro | `avro` |
| {py:meth}`ds.write.msgpack(path) <batcher.api.io_namespace.writer.Writer.msgpack>` | MessagePack | |

### Lakehouse tables

These commit through the table's transaction log rather than writing loose files:

| Writer | Writes | Extra |
| --- | --- | --- |
| {py:meth}`ds.write.delta(path) <batcher.api.io_namespace.writer.Writer.delta>` | a Delta Lake table (one transactional commit) | |
| {py:meth}`ds.write.iceberg(table, mode="append") <batcher.api.io_namespace.writer.Writer.iceberg>` | an Iceberg table (`append` / `overwrite`) | |
| {py:meth}`ds.write.hudi(path, mode="append") <batcher.api.io_namespace.writer.Writer.hudi>` | an Apache Hudi table | |
| {py:meth}`ds.write.lance(path) <batcher.api.io_namespace.writer.Writer.lance>` | a Lance dataset | `lance` |
| {py:meth}`ds.write.merge(target, on=) <batcher.api.io_namespace.writer.Writer.merge>` | upsert (`MERGE INTO`) this dataset into an existing `target`, keyed on `on` | |
| {py:meth}`ds.write.merge_into(target, on=) <batcher.api.io_namespace.writer.Writer.merge_into>` | the full `MERGE INTO`: ordered `WHEN` clauses, each writing its own columns | |

### Merge clauses

`merge` is the two-clause shorthand. `merge_into` is the whole statement, and inside its
clauses {py:obj}`source_col <batcher.source_col>` and
{py:obj}`target_col <batcher.target_col>` name the two sides of the match: the incoming
row and the row already in the table. See the
{doc}`lakehouse guide </user-guide/moving-data/lakehouse>` for worked upserts.

```{eval-rst}
.. currentmodule:: batcher

.. autofunction:: source_col

.. autofunction:: target_col
```

### Warehouses and databases

These load the result into an external system:

| Writer | Writes |
| --- | --- |
| {py:meth}`ds.write.snowflake(table, connection_kwargs=) <batcher.api.io_namespace.writer.Writer.snowflake>` | a Snowflake table |
| {py:meth}`ds.write.sql(table, driver=, db_kwargs=) <batcher.api.io_namespace.writer.Writer.sql>` | a database table via ADBC / FlightSQL |
| {py:meth}`ds.write.mongo(...) <batcher.api.io_namespace.writer.Writer.mongo>` | a MongoDB collection |

## The connector surface

Everything above is built from the same four types, exported from `batcher.io`. You only
need them to add a format the engine doesn't ship. See
{doc}`extending Batcher </architecture/internals/extending>` for the walkthrough.

```python
from batcher.io import Source, Sink, Split, SOURCES
```

A {py:obj}`Source <batcher.io.Source>` answers two questions. What is your schema, and how
do you break into independently readable pieces? Each piece is a
{py:obj}`Split <batcher.io.Split>`, and that split is the unit of parallelism. That's why a 1,000-file Parquet directory and a single 1,000-row-group file both parallelize, and why a format that can't be divided still works, serially.

### Protocols

```{eval-rst}
.. currentmodule:: batcher.io

.. autoclass:: Source
   :members:

.. autoclass:: Sink
   :members:
```

### Splits

The unit of read parallelism. A {py:obj}`RowGroupSplit <batcher.io.RowGroupSplit>` reads
one Parquet row group, a {py:obj}`FileSplit <batcher.io.FileSplit>` reads a byte range of
one file, and a {py:obj}`WholeSourceSplit <batcher.io.WholeSourceSplit>` is the
degenerate case for a source that can't be divided.

```{eval-rst}
.. autoclass:: Split
   :members:

.. autoclass:: RowGroupSplit
   :members:

.. autoclass:: FileSplit
   :members:

.. autoclass:: WholeSourceSplit
   :members:
```

### Built-in sources and sinks

The concrete implementations behind `bt.read.*` and `ds.write.*`.

```{eval-rst}
.. autoclass:: FileSource
   :members:

.. autoclass:: FileSink
   :members:

.. autoclass:: ParquetSource
   :members:

.. autoclass:: ParquetSink
   :members:

.. autoclass:: CSVSource
   :members:

.. autoclass:: CSVSink
   :members:

.. autoclass:: JSONSource
   :members:

.. autoclass:: JSONSink
   :members:

.. autoclass:: InMemorySource
   :members:

.. autoclass:: IteratorSource
   :members:

.. autofunction:: read_blob_bytes
```

### The registries

Formats are discovered, not hard-coded. Registering a source under a name is what makes `bt.read(path, format="myfmt")` resolve.

```{eval-rst}
.. autodata:: SOURCES

.. autodata:: SINKS
```

### Write results

A write is terminal and returns a {py:obj}`WriteManifest <batcher.io.WriteManifest>`: the
list of files it produced. That's what makes a write auditable and a failed run resumable.

```{eval-rst}
.. autoclass:: WriteManifest
   :members:

.. autoclass:: WrittenFile
   :members:
```

## See also

- {doc}`Reading data </user-guide/moving-data/reading-data>` and {doc}`Writing data </user-guide/moving-data/writing-data>`:
  the guided tour of these readers and writers.
- {doc}`Cloud storage </user-guide/moving-data/cloud-storage>`: credentials and object-store paths.
- {doc}`Lakehouse </user-guide/moving-data/lakehouse>`: Delta, Iceberg, and Hudi tables.
- {doc}`Extending Batcher </architecture/internals/extending>`: adding your own source or sink.
- {doc}`/cookbook/io/index`: 6 runnable recipes for these readers and writers.
