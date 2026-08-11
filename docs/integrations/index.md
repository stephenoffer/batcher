# Integrations

This section holds one page per connector, covering the systems already in your stack: streams, warehouses, lakehouse tables, databases, and the ML runtimes on either side of a model.

Batcher does not ask you to move your data somewhere else first. It reads the table, the
topic, or the folder of images where it already lives, and writes back to the same place.
Each page here covers the reader, the writer if there is one, how credentials work, how the
source divides itself for parallel reads, and the failure mode that bites people in
production.

Every connector is built from the same four types ({py:class}`Source <batcher.io.Source>`, {py:class}`Sink <batcher.io.Sink>`, {py:class}`Split <batcher.io.Split>`, and the
format registry). If yours is not on this list, that surface is public and documented in
{doc}`custom connectors </user-guide/moving-data/custom-connectors>`.

:::{tip}
Every connector page opens with a capability table: what it reads, what it writes, the pip extra,
how it splits, and what pushes down. Start there before you read the prose.
:::

## In this section

| Group | Pages | Covers |
|---|---|---|
| {doc}`/integrations/streams/index` | 5 | Kafka, Kinesis, Pulsar, Pub/Sub, and Event Hubs |
| {doc}`/integrations/warehouses/index` | 3 | Snowflake, BigQuery, and Databricks |
| {doc}`/integrations/lakehouse/index` | 3 | Delta Lake, Apache Iceberg, and Apache Hudi |
| {doc}`/integrations/databases/index` | 3 | SQL databases over one URI, plus MongoDB and Elasticsearch |
| {doc}`/integrations/compute/index` | 3 | Ray, PyTorch, and Hugging Face |

## See also

- {doc}`Reading data </user-guide/moving-data/reading-data>` and {doc}`Writing data </user-guide/moving-data/writing-data>`:
  the reader and writer surface every connector on this page plugs into, including how a source
  divides itself into splits and which predicates reach it.
- {doc}`Custom connectors </user-guide/moving-data/custom-connectors>`: the protocol, for a system not listed.
- {doc}`Cloud storage </user-guide/moving-data/cloud-storage>`: credentials and object-store paths.
- {doc}`I/O API </api/relational/io>`: the reference.

```{toctree}
:hidden:

streams/index
warehouses/index
lakehouse/index
databases/index
compute/index
```
