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

:::{grid-item-card} {octicon}`stack;1.1em` MongoDB
:link: /integrations/databases/mongodb
:link-type: doc
Arrow off the wire, and a bulk upsert back. Splits the `_id` range on request.
:::

:::{grid-item-card} {octicon}`search;1.1em` Elasticsearch
:link: /integrations/databases/elasticsearch
:link-type: doc
No sink. Take the ES|QL Arrow path, not the scroll you land on by accident.
:::

::::

```{toctree}
:hidden:

databases
mongodb
elasticsearch
```
