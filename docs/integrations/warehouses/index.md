# Warehouses

Columnar query services. Each one is reached through its own bulk-read protocol rather than
a row-by-row cursor, so a scan arrives as Arrow.

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {octicon}`database;1.1em` Snowflake
:link: /integrations/warehouses/snowflake
:link-type: doc
Read and write. Splits follow Snowflake's result chunks; the write appends.
:::

:::{grid-item-card} {octicon}`database;1.1em` BigQuery
:link: /integrations/warehouses/bigquery
:link-type: doc
Read-only, over the Storage Read API. Prune columns server-side or pay for them.
:::

:::{grid-item-card} {octicon}`database;1.1em` Databricks
:link: /integrations/warehouses/databricks
:link-type: doc
No sink. Unity vends credentials, and the scan lands on Delta files.
:::

::::

```{toctree}
:hidden:

snowflake
bigquery
databricks
```
