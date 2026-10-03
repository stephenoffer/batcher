# Move data

This section covers getting data into and out of Batcher, whether it lives in a file, a warehouse, a lakehouse table or a stream that never ends.

There are two entry points. {py:obj}`bt.read <batcher.read>` turns a path, table or topic into a lazy `Dataset`. {py:obj}`ds.write <batcher.Dataset.write>` sends one anywhere a reader can reach: a local file, an S3 prefix, a Postgres table, a Delta table, a Kafka topic.

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

Each region gets its own directory. Reading it back recovers `region` from the directory names, and the filter skips `us` without opening a single file in it.

Other formats work the same way:

```python
d = tempfile.mkdtemp()
ds = bt.from_pydict({"id": [1, 2, 3], "amount": [10, 20, 30]})
ds.write.csv(os.path.join(d, "orders.csv"))
ds.write.json(os.path.join(d, "orders.json"))
print(bt.read.csv(os.path.join(d, "orders.csv")).to_pydict())
# {'id': [1, 2, 3], 'amount': [10, 20, 30]}
print(bt.read.json(os.path.join(d, "orders.json")).to_pydict())
# {'id': [1, 2, 3], 'amount': [10, 20, 30]}
```

Lakehouse tables add transactions and time travel:

```python
table = os.path.join(d, "orders_delta")
ds.write.delta(table, mode="overwrite")
bt.from_pydict({"id": [4], "amount": [40]}).write.delta(table, mode="append")
print(bt.read.delta(table).count(), bt.read.delta(table, version=0).count())
# 4 3
```

A stream is the same `Dataset` over a source with no end. You consume it batch by batch:

```python
stream = bt.read.rate_micro_batch(4, num_rows=8)
print([batch.num_rows for batch in stream.iter_batches()])
# [4, 4]
```

## Files, tables and connectors

The bounded side. It runs from one local file up to a write spread across a cluster.

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {octicon}`download;1.1em` Reading data
:link: /user-guide/moving-data/reading-data
:link-type: doc
In-memory constructors and file readers that tolerate broken files, ragged lines and the wrong encoding.
:::

:::{grid-item-card} {octicon}`server;1.1em` Databases and warehouses
:link: /integrations/databases/databases
:link-type: doc
SQL over a connection URI, where credentials come from, and splitting one extract into parallel queries.
:::

:::{grid-item-card} {octicon}`beaker;1.1em` Scientific and specialized formats
:link: /user-guide/moving-data/specialized-formats
:link-type: doc
Zarr and HDF5 arrays, web crawls, PDFs, LiDAR point clouds, and robot and vehicle logs.
:::

:::{grid-item-card} {octicon}`upload;1.1em` Writing data
:link: /user-guide/moving-data/writing-data
:link-type: doc
Partitioned layouts and file sizing, plus completion markers, compaction and distributed writes.
:::

:::{grid-item-card} {octicon}`database;1.1em` Catalogs and tables
:link: /user-guide/moving-data/catalogs-and-tables
:link-type: doc
Named tables and save modes, and catalogs you attach.
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

What changes when the input never ends? Mostly timing. These pages cover when results appear, what the engine has to remember, and how to watch it.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`broadcast;1.1em` Streaming
:link: /user-guide/moving-data/streaming/index
:link-type: doc
Brokers and incremental files, triggers, event-time windows, and exactly-once writes to Delta.
:::

:::{grid-item-card} {octicon}`clock;1.1em` When a stream emits
:link: /user-guide/moving-data/streaming/emission
:link-type: doc
Which shapes produce rows as data arrives, and what to do about the ones that wait.
:::

:::{grid-item-card} {octicon}`database;1.1em` Stateful streaming
:link: /user-guide/moving-data/streaming/stateful
:link-type: doc
Watermark dedup, interval joins, session windows, keyed state, and state that spills and checkpoints incrementally.
:::

:::{grid-item-card} {octicon}`pulse;1.1em` Monitoring a stream
:link: /user-guide/moving-data/streaming/monitoring
:link-type: doc
Lag and late rows, retained state, and listeners that fire on every micro-batch.
:::
::::

## See also

Reference and setup pages for the readers and writers above.

- {doc}`/api/relational/io`: the full {py:obj}`bt.read <batcher.read>` and `ds.write` reference.
- {doc}`/integrations/index`: setup for a specific database, warehouse or service.
- {doc}`/user-guide/operate/tuning/object-storage`: making object-store reads fast.
- {doc}`/user-guide/trust/data-quality`: validating data on its way in or out.
- {doc}`/examples/io`: 39 reader and writer scripts, each run on every commit.

```{toctree}
:hidden:

reading-data
specialized-formats
writing-data
catalogs-and-tables
cloud-storage
lakehouse
streaming/index
custom-connectors
```
