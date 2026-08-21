# Databases

Operational stores. These are the sources a pipeline extracts from, and the partitioning
options are what decide whether the extract runs in parallel.

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {octicon}`database;1.1em` SQL databases
:link: /integrations/databases/databases
:link-type: doc
One connection URI for Postgres, MySQL, Oracle, and the rest. Plus any DB-API driver.
:::

:::{grid-item-card} {octicon}`pencil;1.1em` Writing to a database
:link: /integrations/databases/writing
:link-type: doc
Append a load, or maintain a table key by key: upsert, update, delete, in one transaction.
:::

:::{grid-item-card} {octicon}`stack;1.1em` MongoDB
:link: /integrations/databases/mongodb
:link-type: doc
Arrow off the wire, and a bulk upsert back. Splits the `_id` range on request.
:::

:::{grid-item-card} {octicon}`key;1.1em` Key-value stores
:link: /integrations/databases/key-value-stores
:link-type: doc
DynamoDB, Cassandra, Redis. A pinned partition key reads one partition, not the table.
:::

:::{grid-item-card} {octicon}`search;1.1em` Elasticsearch
:link: /integrations/databases/elasticsearch
:link-type: doc
Take the ES|QL Arrow path, not the scroll you land on by accident. Indexes back over `_bulk`.
:::

::::

```{toctree}
:hidden:

databases
writing
key-value-stores
mongodb
elasticsearch
```
