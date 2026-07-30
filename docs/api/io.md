# Reading and writing

This page lists every reader and writer, then the connector types behind them. For the transformations that sit between a read and a write, see {doc}`Dataset <dataset>`.

Readers hang off {py:obj}`bt.read <batcher.read>` and return a lazy `Dataset`. Writers hang off `ds.write` and are terminal, so they execute the plan and return a `WriteManifest`. {py:obj}`bt.read(path, format=None, **opts) <batcher.read>` infers the format from the path, and the dedicated readers below are explicit. Some connectors need an optional dependency. The "Extra" column gives the install name for `pip install 'batcher-engine[<extra>]'`.

## Readers

The readers are grouped by the kind of system they pull from. Within each group they're ordered by how often you'll reach for them.

### Files

These read one file, a directory, or a glob from local disk or object storage:

| Reader | Reads | Extra |
| --- | --- | --- |
| `bt.read.parquet(path)` | a Parquet file, directory, or glob | |
| `bt.read.parquet_dataset(path)` | a (Hive-)partitioned Parquet dataset directory | |
| `bt.read.csv(path)` | a CSV file, directory, or glob | |
| `bt.read.json(path)` | newline-delimited JSON | |
| `bt.read.orc(path)` | ORC file(s) | |
| `bt.read.arrow(path)` | Arrow/Feather IPC file(s) | |
| `bt.read.avro(path)` | Avro file(s) | `avro` |
| `bt.read.excel(path)` | Excel workbook(s) | `excel` |
| `bt.read.xml(path)` | XML file(s) | `xml` |
| `bt.read.text(path, mode="line")` | text file(s) as rows (`mode="line"` or `"file"`) | |
| `bt.read.binary(path)` | whole files as `{uri, bytes, size, mime}` rows | |
| `bt.read.warc(path)` | web-archive (WARC) file(s), one row per record; `.warc.gz` read transparently | |
| `bt.read.numpy(path)` | NumPy `.npy` / `.npz` file(s) | |
| `bt.read.hdf5(path)` | HDF5 file(s) | `hdf5` |
| `bt.read.zarr(path)` | a Zarr store | `zarr` |
| `bt.read.logs(path, pattern=None)` | line-delimited logs; `pattern=` for grok extraction | |
| `bt.read.files_incremental(path)` | incrementally discover new files under `path` | |
| `bt.read.table(name)` | any registered non-file source by name (escape hatch) | |

### Lakehouse tables

These read a transactional table through its metadata layer, so a read sees one consistent snapshot:

| Reader | Reads | Extra |
| --- | --- | --- |
| `bt.read.delta(path, version=, timestamp=)` | a Delta Lake table (time travel) | |
| `bt.read.iceberg(table, catalog=, snapshot_id=)` | an Iceberg table | |
| `bt.read.hudi(path)` | an Apache Hudi table (read-only) | |
| `bt.read.lance(path)` | a Lance dataset | `lance` |
| `bt.read.databricks(table)` | a Databricks / Unity Catalog table (→ Delta) | |
| `bt.read.delta_sharing(url)` | a Delta Sharing table by profile URL | |

### Warehouses and databases

These submit a query to an external engine and stream the Arrow result back:

| Reader | Reads |
| --- | --- |
| `bt.read.sql(query, uri=)` | ADBC / FlightSQL in a single submission (or `table=` for a whole table) |
| `bt.read.snowflake(query, connection_kwargs=)` | a Snowflake query (parallel result-chunk fetch) |
| `bt.read.bigquery(...)` | BigQuery via the Storage Read API (parallel Arrow streams) |
| `bt.read.clickhouse(query)` | a ClickHouse query (Arrow-native) |

### NoSQL

Each of these splits the keyspace so the collection reads in parallel:

| Reader | Reads |
| --- | --- |
| `bt.read.mongo(...)` | a MongoDB collection (Arrow-native via pymongoarrow) |
| `bt.read.cassandra(...)` | Cassandra / Scylla via token-range splits |
| `bt.read.dynamodb(...)` | DynamoDB via native parallel scan segments |
| `bt.read.elasticsearch(...)` | Elasticsearch via ES\|QL Arrow / sliced scroll |

### Streaming

These return an unbounded `Dataset`. See {doc}`streaming <../user-guide/streaming>` for triggers and checkpoints.

| Reader | Reads |
| --- | --- |
| `bt.read.kafka(topic)` | a Kafka topic as an unbounded streaming source |
| `bt.read.kinesis(stream_name)` | an AWS Kinesis stream as an unbounded source |
| `bt.read.pulsar(topic)` | an Apache Pulsar topic as an unbounded source |
| `bt.read.pubsub(subscription)` | a Google Cloud Pub/Sub subscription as an unbounded source |
| `bt.read.eventhubs(hub)` | an Azure Event Hubs stream as an unbounded source |

### Multimodal and ML formats

These read media and document files as rows of bytes plus metadata, decoding only when you ask:

| Reader | Reads | Extra |
| --- | --- | --- |
| `bt.read.images(path, decode=False)` | images (uri/bytes/size/mime + header meta) | `image` |
| `bt.read.audio(path, decode=False)` | audio files (+ `waveform` when decoded) | `audio` |
| `bt.read.video(path, decode=False)` | video files (+ frames when decoded) | `video` |
| `bt.read.documents(path)` | PDF document(s) as text rows | `pdf` |
| `bt.read.webdataset(path)` | WebDataset `.tar` shard(s) | |

## Writers

`ds.write(path, fmt=None, ...)` infers the format, and the dedicated writers are explicit. Each executes the plan and returns a `WriteManifest`.

### Files

These write one file per output partition:

| Writer | Writes | Extra |
| --- | --- | --- |
| `ds.write.parquet(path, compression="zstd")` | Parquet | |
| `ds.write.csv(path)` | CSV | |
| `ds.write.json(path)` | newline-delimited JSON | |
| `ds.write.orc(path)` | ORC | |
| `ds.write.arrow(path)` | Arrow/Feather IPC | |
| `ds.write.avro(path)` | Avro | `avro` |
| `ds.write.msgpack(path)` | MessagePack | |

### Lakehouse tables

These commit through the table's transaction log rather than writing loose files:

| Writer | Writes | Extra |
| --- | --- | --- |
| `ds.write.delta(path)` | a Delta Lake table (one transactional commit) | |
| `ds.write.iceberg(table, mode="append")` | an Iceberg table (`append` / `overwrite`) | |
| `ds.write.hudi(path, mode="append")` | an Apache Hudi table | |
| `ds.write.lance(path)` | a Lance dataset | `lance` |
| `ds.write.merge(target, on=)` | upsert (`MERGE INTO`) this dataset into an existing `target`, keyed on `on` | |
| `ds.write.merge_into(target, on=)` | the full `MERGE INTO`: ordered `WHEN` clauses, each writing its own columns | |

### Merge clauses

`merge` is the two-clause shorthand. `merge_into` is the whole statement, and inside its
clauses {py:obj}`source_col <batcher.source_col>` and
{py:obj}`target_col <batcher.target_col>` name the two sides of the match: the incoming
row and the row already in the table. See the
{doc}`lakehouse guide <../user-guide/lakehouse>` for worked upserts.

```{eval-rst}
.. currentmodule:: batcher

.. autofunction:: source_col

.. autofunction:: target_col
```

### Warehouses and databases

These load the result into an external system:

| Writer | Writes |
| --- | --- |
| `ds.write.snowflake(table, connection_kwargs=)` | a Snowflake table |
| `ds.write.sql(table, driver=, db_kwargs=)` | a database table via ADBC / FlightSQL |
| `ds.write.mongo(...)` | a MongoDB collection |

## The connector surface

Everything above is built from the same four types, exported from `batcher.io`. You only
need them to add a format the engine doesn't ship. See
{doc}`extending Batcher <../internals/extending>` for the walkthrough.

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

- {doc}`Reading data <../user-guide/reading-data>` and {doc}`Writing data <../user-guide/writing-data>`:
  the guided tour of these readers and writers.
- {doc}`Cloud storage <../user-guide/cloud-storage>`: credentials and object-store paths.
- {doc}`Lakehouse <../user-guide/lakehouse>`: Delta, Iceberg, and Hudi tables.
- {doc}`Extending Batcher <../internals/extending>`: adding your own source or sink.
