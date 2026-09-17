# Construction and I/O

Every way to build a {py:class}`Dataset <batcher.Dataset>`, every way to get data back out,
and the two namespaces those calls hand you.

## Top-level functions

The module-level names `batcher` exports, less the expressions, the configuration
functions and governance, which have pages of their own. Most of them construct a dataset.
The `from_*` family wraps an object already in the process. `read` is the entry point for
files, tables and streams, and `read.table` constructs any registered connector by name.

The rest is a mixed bag. Nothing in it hangs off a `Dataset`, so the list is worth reading
rather than guessing at: {py:func}`compact <batcher.compact>` and
{py:func}`vacuum <batcher.vacuum>` maintain a transactional table and take a path rather
than a dataset, {py:func}`streams <batcher.streams>`,
{py:func}`await_any_termination <batcher.await_any_termination>` and
{py:func}`reset_terminated <batcher.reset_terminated>` control running streaming queries
under Spark's names, and {py:func}`register_function <batcher.register_function>` and
{py:func}`register_model <batcher.register_model>` extend the default SQL session that
{py:func}`bt.sql <batcher.sql>` falls back on. {py:func}`versions <batcher.versions>`
reports the build, compiled engine included.

```{eval-rst}
.. currentmodule:: batcher

.. autosummary::
   :toctree: generated
   :nosignatures:

   from_pydict
   from_pylist
   from_records
   from_items
   from_iter
   from_arrow
   from_batches
   from_numpy
   from_pandas
   from_polars
   from_duckdb
   from_spark
   from_daft
   from_dask
   from_huggingface
   from_torch
   from_tf
   from_ray_dataset
   from_any
   read
   read_memory
   sql
   streams
   await_any_termination
   reset_terminated
   register_function
   register_model
   udf
   compact
   vacuum
   release_cluster
   engine_version
   versions
   show_versions
   accelerators
   show_accelerators
   measure_energy
   start_ui
   stop_ui
   ui_url
```

## Reading and writing

The format-specific calls hang off two namespaces rather than the top level.
{py:obj}`bt.read <batcher.read>` holds every reader and returns a lazy dataset: nothing is
opened yet. {py:obj}`ds.write <batcher.Dataset.write>` holds every writer and is terminal,
so it runs the plan and hands back a manifest of the files it wrote.

## Readers

`bt.read` is an instance of `Reader`. Call it to infer the format from the path, or call the typed method for a format.

```{eval-rst}
.. currentmodule:: batcher.api.io_namespace.reader

.. autoclass:: Reader
   :no-members:
```

### Structured files

These read a file, a directory, or a glob of a tabular or record format.

```{eval-rst}
.. currentmodule:: batcher.api.io_namespace.reader

.. autosummary::
   :toctree: generated
   :nosignatures:

   Reader.parquet
   Reader.parquet_dataset
   Reader.csv
   Reader.json
   Reader.orc
   Reader.arrow
   Reader.avro
   Reader.excel
   Reader.xml
   Reader.msgpack
```

### Text, documents and logs

These turn text, whole files, and archives into rows, one line, file, or record at a time.

```{eval-rst}
.. currentmodule:: batcher.api.io_namespace.reader

.. autosummary::
   :toctree: generated
   :nosignatures:

   Reader.text
   Reader.logs
   Reader.binary
   Reader.documents
   Reader.warc
```

### Multimodal and training data

These list media files with their header metadata, or read the shard containers training pipelines consume.

```{eval-rst}
.. currentmodule:: batcher.api.io_namespace.reader

.. autosummary::
   :toctree: generated
   :nosignatures:

   Reader.images
   Reader.audio
   Reader.video
   Reader.webdataset
   Reader.tfrecord
   Reader.training_shards
```

### Scientific arrays and sensor logs

These read n-dimensional arrays and the recording formats robots and vehicles write.

```{eval-rst}
.. currentmodule:: batcher.api.io_namespace.reader

.. autosummary::
   :toctree: generated
   :nosignatures:

   Reader.numpy
   Reader.hdf5
   Reader.zarr
   Reader.mcap
   Reader.mdf
   Reader.point_cloud
```

### Genomics

These read the sequence, interval, annotation, and variant formats of bioinformatics.

```{eval-rst}
.. currentmodule:: batcher.api.io_namespace.reader

.. autosummary::
   :toctree: generated
   :nosignatures:

   Reader.fasta
   Reader.fastq
   Reader.bed
   Reader.gff
   Reader.vcf
```

### Lakehouse tables

These read a table through its metadata layer rather than as loose files.

```{eval-rst}
.. currentmodule:: batcher.api.io_namespace.reader

.. autosummary::
   :toctree: generated
   :nosignatures:

   Reader.delta
   Reader.read_change_feed
   Reader.iceberg
   Reader.hudi
   Reader.lance
   Reader.databricks
   Reader.delta_sharing
```

### Databases and warehouses

These submit a query or scan to an external system, and `table` reaches any registered non-file source by name.

```{eval-rst}
.. currentmodule:: batcher.api.io_namespace.reader

.. autosummary::
   :toctree: generated
   :nosignatures:

   Reader.sql
   Reader.snowflake
   Reader.bigquery
   Reader.clickhouse
   Reader.mongo
   Reader.cassandra
   Reader.dynamodb
   Reader.elasticsearch
   Reader.redis
   Reader.hbase
   Reader.table
```

### Streams

These return an unbounded source for a streaming query, including incremental file discovery and the development sources.

```{eval-rst}
.. currentmodule:: batcher.api.io_namespace.reader

.. autosummary::
   :toctree: generated
   :nosignatures:

   Reader.kafka
   Reader.kinesis
   Reader.pulsar
   Reader.pubsub
   Reader.eventhubs
   Reader.files_incremental
   Reader.socket
   Reader.rate
   Reader.rate_micro_batch
```

## Writers

`ds.write` is an instance of `Writer`. Call it to infer the sink format from the path, or call the typed method for a format.

```{eval-rst}
.. currentmodule:: batcher.api.io_namespace.writer

.. autoclass:: Writer
   :no-members:
```

### Files

These write the dataset as files in a tabular, text, record, training-data, or genomics format.

```{eval-rst}
.. currentmodule:: batcher.api.io_namespace.writer

.. autosummary::
   :toctree: generated
   :nosignatures:

   Writer.parquet
   Writer.csv
   Writer.json
   Writer.text
   Writer.xml
   Writer.orc
   Writer.arrow
   Writer.avro
   Writer.msgpack
   Writer.tfrecord
   Writer.webdataset
   Writer.numpy
   Writer.fasta
   Writer.fastq
   Writer.bed
   Writer.gff
```

### Lakehouse tables and merges

These commit to a transactional table, or upsert into one with `MERGE INTO`. `hudi` raises, because Batcher reads Hudi tables but does not write them.

```{eval-rst}
.. currentmodule:: batcher.api.io_namespace.writer

.. autosummary::
   :toctree: generated
   :nosignatures:

   Writer.delta
   Writer.iceberg
   Writer.merge
   Writer.merge_into
   Writer.lance
   Writer.hudi
```

### Databases and warehouses

These write rows into an external database, warehouse, or search index.

```{eval-rst}
.. currentmodule:: batcher.api.io_namespace.writer

.. autosummary::
   :toctree: generated
   :nosignatures:

   Writer.sql
   Writer.clickhouse
   Writer.snowflake
   Writer.mongo
   Writer.dynamodb
   Writer.cassandra
   Writer.redis
   Writer.elasticsearch
   Writer.hbase
```

### Streaming sinks

These run each micro-batch of a streaming query into a topic, a callback, memory, the console, or nowhere.

```{eval-rst}
.. currentmodule:: batcher.api.io_namespace.writer

.. autosummary::
   :toctree: generated
   :nosignatures:

   Writer.kafka
   Writer.for_each_batch
   Writer.for_each
   Writer.memory
   Writer.console
   Writer.noop
```

## See also

- {doc}`/api/relational/io`: the same readers and writers with their options, save modes, and the extras each connector installs.
- {doc}`dataset`: what a `Dataset` does once you hold one.
- {doc}`/user-guide/moving-data/reading-data`: choosing a reader, and the cloud paths and credentials behind it.
