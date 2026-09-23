# Databases

Batcher reads straight out of the operational databases your applications already write to, and writes results back into them. Point it at a PostgreSQL connection string, a MongoDB collection or a DynamoDB table, and the extract arrives as Arrow with your filter and column list already pushed to the server.

The same `mode` vocabulary covers every write: `append` to load, `upsert`, `update` and `delete` to maintain a table key by key. A retried or replayed job lands on the same rows instead of duplicating them.

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {octicon}`database;1.1em` SQL databases
:link: /integrations/databases/databases
:link-type: doc
One connection URI for PostgreSQL, MySQL, SQL Server, Oracle, SQLite, Trino and the rest, 36 schemes in all. Parallel range extraction, and any PEP 249 driver besides.
:::

:::{grid-item-card} {octicon}`pencil;1.1em` Writing to a database
:link: /integrations/databases/writing
:link-type: doc
Append a load, or maintain a table with upsert, update and delete, one transaction per write.
:::

:::{grid-item-card} {octicon}`key;1.1em` Key-value stores
:link: /integrations/databases/key-value-stores
:link-type: doc
DynamoDB, Cassandra, ScyllaDB, Redis and HBase. A pinned partition key reads one partition instead of the table.
:::

:::{grid-item-card} {octicon}`stack;1.1em` MongoDB
:link: /integrations/databases/mongodb
:link-type: doc
Arrow built directly off the wire, `_id` range splits, and bulk upserts back.
:::

:::{grid-item-card} {octicon}`search;1.1em` Elasticsearch
:link: /integrations/databases/elasticsearch
:link-type: doc
ES|QL results as an Arrow stream, sliced scroll for bulk pulls, and `_bulk` indexing with every item checked.
:::

::::

A few more stores have readers without a page of their own. ClickHouse reads Arrow natively through {py:meth}`bt.read.clickhouse <batcher.api.io_namespace.reader.Reader.clickhouse>`. DB2, Teradata, SAP HANA, Vertica and other ODBC-reachable systems read through `bt.read.table("odbc", ...)`. Couchbase Columnar and Neo4j read through `bt.read.table("couchbase", ...)` and `bt.read.table("neo4j", ...)`. {doc}`/api/symbols/readers-and-writers` lists the reader surface they share.

```{toctree}
:hidden:

databases
writing
key-value-stores
mongodb
elasticsearch
```
