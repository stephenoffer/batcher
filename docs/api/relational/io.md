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
| {py:meth}`bt.read.files_incremental(path, format) <batcher.api.io_namespace.reader.Reader.files_incremental>` | incrementally discover new files under `path` | |
| {py:meth}`bt.read.table(name) <batcher.api.io_namespace.reader.Reader.table>` | any registered non-file source by name (escape hatch) | |

### Lakehouse tables

These read a transactional table through its metadata layer, so a read sees one consistent snapshot:

| Reader | Reads | Extra |
| --- | --- | --- |
| {py:meth}`bt.read.delta(path, version=, timestamp=) <batcher.api.io_namespace.reader.Reader.delta>` | a Delta Lake table (time travel) | |
| {py:meth}`bt.read.iceberg(table, catalog=, snapshot_id=) <batcher.api.io_namespace.reader.Reader.iceberg>` | an Iceberg table | |
| {py:meth}`bt.read.hudi(path) <batcher.api.io_namespace.reader.Reader.hudi>` | an Apache Hudi table (read-only) | |
| {py:meth}`bt.read.lance(path) <batcher.api.io_namespace.reader.Reader.lance>` | a Lance dataset | `lance` |
| {py:meth}`bt.read.databricks(table) <batcher.api.io_namespace.reader.Reader.databricks>` | a Databricks / Unity Catalog table (-> Delta) | |
| {py:meth}`bt.read.delta_sharing(url) <batcher.api.io_namespace.reader.Reader.delta_sharing>` | a Delta Sharing table by profile URL | |

### Warehouses and databases

These submit a query to an external engine and stream the Arrow result back:

| Reader | Reads |
| --- | --- |
| {py:meth}`bt.read.sql(query, uri=) <batcher.api.io_namespace.reader.Reader.sql>` | ADBC / FlightSQL in a single submission (or `table=` for a whole table) |
| {py:meth}`bt.read.snowflake(query, connection_kwargs=) <batcher.api.io_namespace.reader.Reader.snowflake>` | a Snowflake query (parallel result-chunk fetch) |
| {py:meth}`bt.read.bigquery(...) <batcher.api.io_namespace.reader.Reader.bigquery>` | BigQuery via the Storage Read API (parallel Arrow streams) |
| {py:meth}`bt.read.clickhouse(query) <batcher.api.io_namespace.reader.Reader.clickhouse>` | a ClickHouse query (Arrow-native) |
| {py:meth}`bt.read.athena(query, region=) <batcher.api.io_namespace.reader.Reader.athena>` | an Amazon Athena query, through PyAthena and the DB-API reader |

### NoSQL

Each of these splits the keyspace so the store reads in parallel:

| Reader | Reads |
| --- | --- |
| {py:meth}`bt.read.mongo(...) <batcher.api.io_namespace.reader.Reader.mongo>` | a MongoDB collection (Arrow-native via pymongoarrow) |
| {py:meth}`bt.read.cassandra(...) <batcher.api.io_namespace.reader.Reader.cassandra>` | Cassandra / Scylla via token-range splits |
| {py:meth}`bt.read.dynamodb(...) <batcher.api.io_namespace.reader.Reader.dynamodb>` | DynamoDB via native parallel scan segments |
| {py:meth}`bt.read.elasticsearch(...) <batcher.api.io_namespace.reader.Reader.elasticsearch>` | Elasticsearch via ES\|QL Arrow / sliced scroll |
| {py:meth}`bt.read.redis(...) <batcher.api.io_namespace.reader.Reader.redis>` | a Redis keyspace as `(key, value)` rows, by hash-slot range |
| {py:meth}`bt.read.hbase(...) <batcher.api.io_namespace.reader.Reader.hbase>` | an HBase table, one split per region key range |
| {py:meth}`bt.read.qdrant(collection) <batcher.api.io_namespace.reader.Reader.qdrant>` | a Qdrant collection, by scroll |
| {py:meth}`bt.read.pinecone(index) <batcher.api.io_namespace.reader.Reader.pinecone>` | a Pinecone namespace, by list and fetch (serverless) |
| {py:meth}`bt.read.milvus(collection, uri=) <batcher.api.io_namespace.reader.Reader.milvus>` | a Milvus collection, one split per partition |
| {py:meth}`bt.read.turbopuffer(namespace) <batcher.api.io_namespace.reader.Reader.turbopuffer>` | a Turbopuffer namespace, paged by id |

A parallel scan is the right shape for reading a *table* and the wrong shape for reading one row. When a filter pins the partition key to a single value, DynamoDB and Cassandra skip the fan-out entirely and read the one partition that can hold a match. See {doc}`Key-value stores </integrations/databases/key-value-stores>`. The four vector stores are covered in {doc}`Vector stores </integrations/databases/vector-stores>`, untested against live services.

### HTTP APIs and SaaS

Each of these walks an API page by page, so it reads as one split. See {doc}`/integrations/apis/index`:

| Reader | Reads |
| --- | --- |
| {py:meth}`bt.read.http_json(url, pagination=) <batcher.api.io_namespace.reader.Reader.http_json>` | any paginated JSON API, one Arrow batch per page |
| {py:meth}`bt.read.graphql(url, query, records_path=) <batcher.api.io_namespace.reader.Reader.graphql>` | a GraphQL query, paging a Relay cursor; any `errors` entry fails the read |
| {py:meth}`bt.read.github(repo, resource) <batcher.api.io_namespace.reader.Reader.github>` | GitHub issues, pull requests or releases |
| {py:meth}`bt.read.salesforce(sobject, instance_url=, schema=) <batcher.api.io_namespace.reader.Reader.salesforce>` | a Salesforce object via a Bulk API 2.0 query job |
| {py:meth}`bt.read.google_sheets(spreadsheet_id, range) <batcher.api.io_namespace.reader.Reader.google_sheets>` | a range of a Google Sheet |
| {py:meth}`bt.read.sharepoint(drive_id=) <batcher.api.io_namespace.reader.Reader.sharepoint>` | a SharePoint or OneDrive library through Microsoft Graph delta |
| {py:meth}`bt.read.airbyte(stream, image=) <batcher.api.io_namespace.reader.Reader.airbyte>` | one stream of an Airbyte source connector |

### Streaming

These return an unbounded `Dataset`. See {doc}`streaming </user-guide/moving-data/streaming/index>` for triggers and checkpoints.

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
| {py:meth}`ds.write.arrow(path) <batcher.api.io_namespace.writer.Writer.arrow>` | Arrow/Feather IPC; `ipc_format="stream"` writes the footer-less IPC stream | |
| {py:meth}`ds.write.avro(path) <batcher.api.io_namespace.writer.Writer.avro>` | Avro | `avro` |
| {py:meth}`ds.write.msgpack(path) <batcher.api.io_namespace.writer.Writer.msgpack>` | MessagePack | |
| {py:meth}`ds.write.text(path) <batcher.api.io_namespace.writer.Writer.text>` | one string column as plain text, one value per line | |
| {py:meth}`ds.write.xml(path, row_tag="ROW", root_tag="ROWS") <batcher.api.io_namespace.writer.Writer.xml>` | XML in Spark's row-element layout | |
| {py:meth}`ds.write.numpy(path, column=) <batcher.api.io_namespace.writer.Writer.numpy>` | one column as NumPy `.npy` arrays | |
| {py:meth}`ds.write.webdataset(path) <batcher.api.io_namespace.writer.Writer.webdataset>` | WebDataset `.tar` shards, one sample per row keyed by `__key__` | |
| {py:meth}`ds.write.tfrecord(path, record_format="example") <batcher.api.io_namespace.writer.Writer.tfrecord>` | TFRecord: a `tf.train.Example` per row, or raw record payloads | `tfrecord` |

### Lakehouse tables

These commit through the table's transaction log rather than writing loose files:

| Writer | Writes | Extra |
| --- | --- | --- |
| {py:meth}`ds.write.delta(path) <batcher.api.io_namespace.writer.Writer.delta>` | a Delta Lake table (one transactional commit) | |
| {py:meth}`ds.write.iceberg(table, mode="append") <batcher.api.io_namespace.writer.Writer.iceberg>` | an Iceberg table (`append` / `overwrite`) | |
| {py:meth}`ds.write.hudi(path, mode="append") <batcher.api.io_namespace.writer.Writer.hudi>` | an Apache Hudi table | |
| {py:meth}`ds.write.lance(path) <batcher.api.io_namespace.writer.Writer.lance>` | a Lance dataset | `lance` |
| {py:meth}`ds.write.table(name, mode="error") <batcher.api.io_namespace.writer.Writer.table>` | a catalog table by name, with a save mode (see {doc}`/user-guide/moving-data/catalogs-and-tables`) | |
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

.. autosummary::
   :toctree: generated
   :nosignatures:

   source_col
   target_col
```

### Warehouses and databases

These load the result into an external system. `mode` says what the write does to the target, and every one of them defaults to the non-destructive choice. See {doc}`Writing to a database </integrations/databases/writing>` for the modes and the transaction they run in.

| Writer | Writes | `mode` |
| --- | --- | --- |
| {py:meth}`ds.write.snowflake(table, connection_kwargs=) <batcher.api.io_namespace.writer.Writer.snowflake>` | a Snowflake table | `append` / `overwrite` |
| {py:meth}`ds.write.bigquery(table, project=) <batcher.api.io_namespace.writer.Writer.bigquery>` | a BigQuery table, one Parquet load job per shard; each file's `job` names the load job | `append` / `overwrite` |
| {py:meth}`ds.write.databricks(table, volume_path=) <batcher.api.io_namespace.writer.Writer.databricks>` | an existing Databricks table, staged in a volume and loaded with `COPY INTO` | `append` |
| {py:meth}`ds.write.clickhouse(table, host=) <batcher.api.io_namespace.writer.Writer.clickhouse>` | an existing ClickHouse table, via `insert_arrow` | `append` / `overwrite` (truncates first) |
| {py:meth}`ds.write.sql(table, uri=) <batcher.api.io_namespace.writer.Writer.sql>` | a SQL table, via ADBC for a bulk append and any PEP 249 driver otherwise | `append` / `overwrite` / `upsert` / `update` / `delete` / `delete_insert` |
| {py:meth}`ds.write.mongo(collection, uri=) <batcher.api.io_namespace.writer.Writer.mongo>` | a MongoDB collection | `upsert` / `append` / `overwrite` / `delete` |
| {py:meth}`ds.write.dynamodb(table, region_name=) <batcher.api.io_namespace.writer.Writer.dynamodb>` | a DynamoDB table, via `BatchWriteItem` | `upsert` / `delete` |
| {py:meth}`ds.write.cassandra(table, contact_points=, keyspace=) <batcher.api.io_namespace.writer.Writer.cassandra>` | a Cassandra / Scylla table, one prepared statement run concurrently | `upsert` / `delete` |
| {py:meth}`ds.write.redis(key_prefix, host=) <batcher.api.io_namespace.writer.Writer.redis>` | a Redis keyspace, one pipeline per batch | `upsert` / `delete` |
| {py:meth}`ds.write.elasticsearch(index, hosts=) <batcher.api.io_namespace.writer.Writer.elasticsearch>` | an Elasticsearch index, via `_bulk` | `upsert` / `append` / `overwrite` / `delete` |
| {py:meth}`ds.write.hbase(table, host=) <batcher.api.io_namespace.writer.Writer.hbase>` | an HBase table, one happybase batch per Arrow batch | `upsert` / `delete` |
| {py:meth}`ds.write.qdrant(collection) <batcher.api.io_namespace.writer.Writer.qdrant>` | a Qdrant collection | `upsert` / `delete` |
| {py:meth}`ds.write.pinecone(index, api_key=) <batcher.api.io_namespace.writer.Writer.pinecone>` | a Pinecone index namespace | `upsert` / `delete` |
| {py:meth}`ds.write.milvus(collection, uri=) <batcher.api.io_namespace.writer.Writer.milvus>` | a Milvus collection | `upsert` / `append` / `delete` |
| {py:meth}`ds.write.turbopuffer(namespace, region=) <batcher.api.io_namespace.writer.Writer.turbopuffer>` | a Turbopuffer namespace | `upsert` / `delete` |
| {py:meth}`ds.write.google_sheets(spreadsheet_id, range) <batcher.api.io_namespace.writer.Writer.google_sheets>` | a Google Sheet range, in bounded batches | `overwrite` (clears exactly the range) / `append` |

A mode a store cannot express is refused by name rather than approximated. DynamoDB has no `append`, because `PutItem` replaces the item holding the same key and no batch operation inserts only when the key is absent; Cassandra and HBase have none for the same reason, since a CQL `INSERT` and an HBase `Put` are both upserts. None of the four has `overwrite`, because emptying those stores is a scan-and-delete, a `TRUNCATE`, a `FLUSHDB`, or a disable-and-truncate through an admin API rather than a write.

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

.. autosummary::
   :toctree: generated
   :nosignatures:

   Source
   Sink
```

### Splits

The unit of read parallelism. A {py:obj}`RowGroupSplit <batcher.io.RowGroupSplit>` reads
one Parquet row group, a {py:obj}`FileSplit <batcher.io.FileSplit>` reads a byte range of
one file, and a {py:obj}`WholeSourceSplit <batcher.io.WholeSourceSplit>` is the
degenerate case for a source that can't be divided.

```{eval-rst}
.. currentmodule:: batcher.io

.. autosummary::
   :toctree: generated
   :nosignatures:

   Split
   RowGroupSplit
   FileSplit
   WholeSourceSplit
```

### HTTP source options

The typed options `bt.read.http_json` and the SaaS readers take: a pagination style, an auth provider holding a secret reference, a retry policy, and a resumable incremental state.

```{eval-rst}
.. currentmodule:: batcher.io

.. autosummary::
   :toctree: generated
   :nosignatures:

   CursorPagination
   NextLinkPagination
   OffsetPagination
   PagePagination
   BearerToken
   OAuth2ClientCredentials
   RetryPolicy
   Incremental
```

### Built-in sources and sinks

The concrete implementations behind `bt.read.*` and `ds.write.*`.

```{eval-rst}
.. currentmodule:: batcher.io

.. autosummary::
   :toctree: generated
   :nosignatures:

   FileSource
   FileSink
   ParquetSource
   ParquetSink
   CSVSource
   CSVSink
   JSONSource
   JSONSink
   InMemorySource
   IteratorSource
   read_blob_bytes
```

### The registries

Formats register themselves rather than being listed anywhere. Registering a source under a name makes {py:obj}`bt.read(path, format="myfmt") <batcher.read>` resolve.

```{eval-rst}
.. currentmodule:: batcher.io

.. autodata:: SOURCES

.. autodata:: SINKS
```

### Write results

A write is terminal and returns a {py:obj}`WriteManifest <batcher.io.WriteManifest>`: the
list of files it produced. That's what makes a write auditable and a failed run resumable.

```{eval-rst}
.. currentmodule:: batcher.io

.. autosummary::
   :toctree: generated
   :nosignatures:

   WriteManifest
   WrittenFile
```

## See also

- {doc}`Reading data </user-guide/moving-data/reading-data>` and {doc}`Writing data </user-guide/moving-data/writing-data>`:
  the guided tour of these readers and writers.
- {doc}`Cloud storage </user-guide/moving-data/cloud-storage>`: credentials and object-store paths.
- {doc}`Lakehouse </user-guide/moving-data/lakehouse>`: Delta, Iceberg, and Hudi tables.
- {doc}`Extending Batcher </architecture/internals/extending>`: adding your own source or sink.
- {doc}`/cookbook/io/index`: 6 runnable recipes for these readers and writers.
