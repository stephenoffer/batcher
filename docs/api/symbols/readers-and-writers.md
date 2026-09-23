# Readers and writers

This page lists every reader reachable from {py:obj}`bt.read <batcher.read>` and every writer reachable from {py:obj}`ds.write <batcher.Dataset.write>`, grouped by the kind of system on the other end. Building a dataset from data already in the process is on {doc}`construction`.

A reader returns a lazy dataset, so nothing is read when you call it. The optimizer decides what to read once it has seen the whole query, which is how a filter and a column list reach the Parquet reader as row-group pruning and a projection rather than as a pass over everything.

The two namespaces are not symmetric, and the asymmetry is the thing to remember.
{py:obj}`bt.read <batcher.read>` is lazy. {py:obj}`ds.write <batcher.Dataset.write>` is
terminal: it runs the plan and hands back a manifest of the files it wrote.

## Readers

{py:obj}`bt.read <batcher.read>` is an instance of `Reader`. Call it to infer the format from the path, or call the typed method for a format.

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

These commit to a transactional table, or upsert into one with `MERGE INTO`. {py:obj}`table <batcher.api.io_namespace.writer.Writer.table>` writes to a catalog table by name and lets the attached catalog decide the format. `hudi` raises, because Batcher reads Hudi tables but does not write them.

```{eval-rst}
.. currentmodule:: batcher.api.io_namespace.writer

.. autosummary::
   :toctree: generated
   :nosignatures:

   Writer.table
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

- {doc}`construction`: building a dataset from data you already hold.
- {doc}`/user-guide/moving-data/index`: the guides these readers and writers are the reference for.
- {doc}`/integrations/index`: the setup each external system needs.
- {doc}`/api/relational/io`: the same surface with a runnable example per source family.
