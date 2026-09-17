# Move data

This section covers getting data into and out of Batcher: files and object storage, databases and warehouses, lakehouse tables, and unbounded streams.

All of it goes through two entry points. {py:obj}`bt.read <batcher.read>` turns a path, a table or a topic into a lazy `Dataset`, and {py:obj}`ds.write <batcher.Dataset.write>` sends one anywhere a reader can reach. A local Parquet file, an S3 prefix, a Postgres table, a Delta table and a Kafka topic all sit behind those two namespaces. Adding a `trigger=` to the write turns a batch job into a continuous one without touching the pipeline in between.

```python
import os
import tempfile

import batcher as bt

out = os.path.join(tempfile.mkdtemp(), "orders")
bt.from_pydict({"region": ["eu", "us", "eu"], "amount": [10, 20, 30]}).write.parquet(
    out, partition_by=["region"]
)
print(bt.read.parquet(out).filter(bt.col("region") == "eu").count())
# 2
```

That round trip already shows several things the pages below explain. The partitioned write lands one directory per region and finishes with a `_SUCCESS` marker. The read gets the `region` column back from the directory names, and the filter prunes the `us` directory before any file in it opens.

## Files, tables and connectors

These pages cover the bounded side: one reader and one writer namespace, from a local file up to a distributed write across a cluster.

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {octicon}`download;1.1em` Reading data
:link: /user-guide/moving-data/reading-data
:link-type: doc
In-memory constructors, file readers, and tolerance for unreadable files, bad lines and wrong encodings.
:::

:::{grid-item-card} {octicon}`server;1.1em` Databases and specialized formats
:link: /user-guide/moving-data/reading-databases
:link-type: doc
SQL databases, warehouses, NoSQL stores, web crawls, PDFs, LiDAR, and robot and vehicle logs.
:::

:::{grid-item-card} {octicon}`upload;1.1em` Writing data
:link: /user-guide/moving-data/writing-data
:link-type: doc
Partitioned layouts, file sizing, completion markers, compaction, and distributed writes.
:::

:::{grid-item-card} {octicon}`database;1.1em` Catalogs and tables
:link: /user-guide/moving-data/catalogs-and-tables
:link-type: doc
Named tables, save modes, attached catalogs.
:::

:::{grid-item-card} {octicon}`cloud;1.1em` Cloud storage
:link: /user-guide/moving-data/cloud-storage
:link-type: doc
S3, Google Cloud Storage, Azure, HDFS, and S3-compatible stores, with credentials from where you already keep them.
:::

:::{grid-item-card} {octicon}`stack;1.1em` Lakehouse tables
:link: /user-guide/moving-data/lakehouse
:link-type: doc
Delta, Iceberg and Hudi: time travel, `MERGE INTO`, slowly changing dimensions, CDC, and file skipping from the log.
:::

:::{grid-item-card} {octicon}`plug;1.1em` Custom connectors
:link: /user-guide/moving-data/custom-connectors
:link-type: doc
Register your own source or sink and inherit splitting, projection pushdown and atomic writes.
:::
::::

## Streaming

A stream is the same `Dataset` over an unbounded source. These pages cover what changes when the input never ends: when results appear, what the engine has to remember, and how you watch it run.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`broadcast;1.1em` Streaming
:link: /user-guide/moving-data/streaming
:link-type: doc
Brokers and incremental files, triggers, event-time windows, and exactly-once writes to Delta.
:::

:::{grid-item-card} {octicon}`clock;1.1em` When a stream emits
:link: /user-guide/moving-data/streaming-emission
:link-type: doc
Which shapes produce rows as data arrives, and what to do about the ones that wait.
:::

:::{grid-item-card} {octicon}`database;1.1em` Stateful streaming
:link: /user-guide/moving-data/streaming-stateful
:link-type: doc
Watermark dedup, interval joins, session windows, keyed state, and state that spills and checkpoints incrementally.
:::

:::{grid-item-card} {octicon}`pulse;1.1em` Monitoring a stream
:link: /user-guide/moving-data/streaming-monitoring
:link-type: doc
Lag, late rows, retained state, and listeners that fire on every micro-batch.
:::
::::

## See also

- {doc}`/api/relational/io`: the full `bt.read` and `ds.write` reference.
- {doc}`/integrations/index`: setup for a specific database, warehouse or service.
- {doc}`/user-guide/operate/tuning/object-storage`: making object-store reads fast.
- {doc}`/user-guide/trust/data-quality`: validating data on its way in or out.

```{toctree}
:hidden:

reading-data
reading-databases
writing-data
catalogs-and-tables
cloud-storage
lakehouse
custom-connectors
streaming
streaming-emission
streaming-stateful
streaming-monitoring
```
