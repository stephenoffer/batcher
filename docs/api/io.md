# Reading and Writing

Readers hang off `bt.read` and return a lazy `Dataset`; writers hang off
`ds.write` and are terminal (they execute the plan and return a `WriteManifest`).
`bt.read(path, format=None, **opts)` infers the format from the path; the dedicated
readers below are explicit. Some connectors need an optional dependency — the
"Extra" column gives the install (`pip install 'batcher-engine[<extra>]'`).

## Readers

### Files

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
| `bt.read.numpy(path)` | NumPy `.npy` / `.npz` file(s) | |
| `bt.read.hdf5(path)` | HDF5 file(s) | `hdf5` |
| `bt.read.zarr(path)` | a Zarr store | `zarr` |
| `bt.read.logs(path, pattern=None)` | line-delimited logs; `pattern=` for grok extraction | |
| `bt.read.files_incremental(path)` | incrementally discover new files under `path` | |
| `bt.read.table(name)` | any registered non-file source by name (escape hatch) | |

### Lakehouse tables

| Reader | Reads | Extra |
| --- | --- | --- |
| `bt.read.delta(path, version=, timestamp=)` | a Delta Lake table (time travel) | |
| `bt.read.iceberg(table, catalog=, snapshot_id=)` | an Iceberg table | |
| `bt.read.hudi(path)` | an Apache Hudi table (read-only) | |
| `bt.read.lance(path)` | a Lance dataset | `lance` |
| `bt.read.databricks(table)` | a Databricks / Unity Catalog table (→ Delta) | |
| `bt.read.delta_sharing(url)` | a Delta Sharing table by profile URL | |

### Warehouses and databases

| Reader | Reads |
| --- | --- |
| `bt.read.sql(query=, table=)` | ADBC / FlightSQL in a single submission |
| `bt.read.snowflake(query)` | a Snowflake query (parallel result-chunk fetch) |
| `bt.read.bigquery(...)` | BigQuery via the Storage Read API (parallel Arrow streams) |
| `bt.read.clickhouse(query)` | a ClickHouse query (Arrow-native) |

### NoSQL

| Reader | Reads |
| --- | --- |
| `bt.read.mongo(...)` | a MongoDB collection (Arrow-native via pymongoarrow) |
| `bt.read.cassandra(...)` | Cassandra / Scylla via token-range splits |
| `bt.read.dynamodb(...)` | DynamoDB via native parallel scan segments |
| `bt.read.elasticsearch(...)` | Elasticsearch via ES\|QL Arrow / sliced scroll |

### Streaming

| Reader | Reads |
| --- | --- |
| `bt.read.kafka(topic)` | a Kafka topic as an unbounded streaming source |
| `bt.read.kinesis(stream)` | an AWS Kinesis stream as an unbounded source |

### Multimodal and ML formats

| Reader | Reads | Extra |
| --- | --- | --- |
| `bt.read.images(path, decode=False)` | images (uri/bytes/size/mime + header meta) | `image` |
| `bt.read.audio(path, decode=False)` | audio files (+ `waveform` when decoded) | `audio` |
| `bt.read.video(path, decode=False)` | video files (+ frames when decoded) | `video` |
| `bt.read.documents(path)` | PDF document(s) as text rows | `pdf` |
| `bt.read.webdataset(path)` | WebDataset `.tar` shard(s) | |

## Writers

`ds.write(path, fmt=None, ...)` infers the format; the dedicated writers are
explicit. Each executes the plan and returns a `WriteManifest`.

### Files

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

| Writer | Writes | Extra |
| --- | --- | --- |
| `ds.write.delta(path)` | a Delta Lake table (one transactional commit) | |
| `ds.write.iceberg(table, mode="append")` | an Iceberg table (`append` / `overwrite`) | |
| `ds.write.hudi(path, mode="append")` | an Apache Hudi table | |
| `ds.write.lance(path)` | a Lance dataset | `lance` |
| `ds.write.merge(target, on=)` | upsert (`MERGE INTO`) this dataset into an existing `target`, keyed on `on` | |

### Warehouses and databases

| Writer | Writes |
| --- | --- |
| `ds.write.snowflake(table)` | a Snowflake table |
| `ds.write.sql(...)` | a database table via ADBC / FlightSQL |
| `ds.write.mongo(...)` | a MongoDB collection |
