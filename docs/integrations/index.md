# Integrations

Connecting Batcher to the systems already in your stack.

Batcher does not ask you to move your data somewhere else first. It reads the table, the
topic, or the folder of images where it already lives, and writes back to the same place.
Each page here covers the reader, the writer if there is one, how credentials work, how the
source divides itself for parallel reads, and the failure mode that bites people in
production.

Every connector is built from the same four types (`Source`, `Sink`, `Split`, and the
format registry). If yours is not on this list, that surface is public and documented in
{doc}`custom connectors <../user-guide/custom-connectors>`.

:::{tip}
Every connector page opens with a capability table: what it reads, what it writes, the pip extra,
how it splits, and what pushes down. Start there before you read the prose.
:::

## Streams

Unbounded sources. The engine treats batch as the bounded special case of streaming, so the
operators are the same ones you already use.

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {octicon}`broadcast;1.1em` Kafka
:link: kafka
:link-type: doc
One reader per topic partition. No sink; publish back through `for_each_batch`.
:::

:::{grid-item-card} {octicon}`broadcast;1.1em` Kinesis
:link: kinesis
:link-type: doc
A split per shard, and an exact resume from the stored sequence number.
:::

:::{grid-item-card} {octicon}`broadcast;1.1em` Pulsar
:link: pulsar
:link-type: doc
Partitions are a number you declare, not one the broker tells you.
:::

:::{grid-item-card} {octicon}`broadcast;1.1em` Pub/Sub
:link: pubsub
:link-type: doc
A single reader, however big the cluster. Ack deadlines make the duplicates.
:::

:::{grid-item-card} {octicon}`broadcast;1.1em` Event Hubs
:link: eventhubs
:link-type: doc
Native AMQP, or the Kafka protocol endpoint that actually resumes.
:::

::::

## Warehouses

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {octicon}`database;1.1em` Snowflake
:link: snowflake
:link-type: doc
Read and write. Splits follow Snowflake's result chunks; the write appends.
:::

:::{grid-item-card} {octicon}`database;1.1em` BigQuery
:link: bigquery
:link-type: doc
Read-only, over the Storage Read API. Prune columns server-side or pay for them.
:::

:::{grid-item-card} {octicon}`database;1.1em` Databricks
:link: databricks
:link-type: doc
No sink. Unity vends credentials, and the scan lands on Delta files.
:::

::::

## Lakehouse tables

Transactional table formats: one atomic commit per write, and time travel to any past
version.

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {octicon}`versions;1.1em` Delta Lake
:link: delta-lake
:link-type: doc
The most complete connector: merge, replace-where, compact, vacuum, CDF.
:::

:::{grid-item-card} {octicon}`versions;1.1em` Apache Iceberg
:link: iceberg
:link-type: doc
Snapshot time travel and manifest pruning. Budget your time for the catalog.
:::

:::{grid-item-card} {octicon}`versions;1.1em` Apache Hudi
:link: hudi
:link-type: doc
No writer here. That is Spark or Flink; Batcher is the consumer.
:::

::::

## Databases

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {octicon}`database;1.1em` SQL databases
:link: databases
:link-type: doc
One connection URI for Postgres, MySQL, Oracle, and the rest. Plus any DB-API driver.
:::

:::{grid-item-card} {octicon}`stack;1.1em` MongoDB
:link: mongodb
:link-type: doc
Arrow off the wire, and a bulk upsert back. Splits the `_id` range on request.
:::

:::{grid-item-card} {octicon}`search;1.1em` Elasticsearch
:link: elasticsearch
:link-type: doc
No sink. Take the ES|QL Arrow path, not the scroll you land on by accident.
:::

::::

## ML and compute

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {octicon}`git-merge;1.1em` Ray
:link: ray
:link-type: doc
Scheduling only. Bulk data moves over Arrow Flight, not the object store.
:::

:::{grid-item-card} {octicon}`zap;1.1em` PyTorch
:link: pytorch
:link-type: doc
Streaming tensors into a training loop, and a shard per DDP rank.
:::

:::{grid-item-card} {octicon}`beaker;1.1em` Hugging Face
:link: huggingface
:link-type: doc
Datasets in with no copy; model ids that load once per worker.
:::

::::

## See also

- {doc}`Reading data <../user-guide/reading-data>` and {doc}`Writing data <../user-guide/writing-data>`:
  the reader and writer surface every connector on this page plugs into, including how a source
  divides itself into splits and which predicates reach it.
- {doc}`Custom connectors <../user-guide/custom-connectors>`: the protocol, for a system not listed.
- {doc}`Cloud storage <../user-guide/cloud-storage>`: credentials and object-store paths.
- {doc}`I/O API <../api/io>`: the reference.

```{toctree}
:hidden:

kafka
kinesis
pulsar
pubsub
eventhubs
snowflake
bigquery
databricks
delta-lake
iceberg
hudi
databases
mongodb
elasticsearch
ray
pytorch
huggingface
```
