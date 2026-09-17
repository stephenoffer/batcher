# Warehouses

Batcher reads your warehouse through each service's bulk Arrow protocol, never a row-by-row cursor. A Snowflake result arrives as parallel cloud-storage chunks, a BigQuery table as parallel Storage Read API streams, and a Databricks table as its own Delta files, read without a warehouse at all. Filters push down on all three, so rows a query discards never cross the network.

The connectors also plan cheaply. A Snowflake query or a Databricks warehouse query gets its schema from a zero-row probe, and a BigQuery or Databricks table gets it from metadata, so learning a table's columns doesn't cost a scan. A warehouse table then joins against the lake in one plan.

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {octicon}`database;1.1em` Snowflake
:link: /integrations/warehouses/snowflake
:link-type: doc
Read and write. One query submission, then one parallel split per result chunk.
:::

:::{grid-item-card} {octicon}`database;1.1em` BigQuery
:link: /integrations/warehouses/bigquery
:link-type: doc
Parallel Arrow streams over the Storage Read API, with filters and columns pushed server-side.
:::

:::{grid-item-card} {octicon}`database;1.1em` Databricks
:link: /integrations/warehouses/databricks
:link-type: doc
Unity Catalog vends credentials and Batcher reads the Delta files directly, with no warehouse in the path.
:::

::::

For Postgres, MySQL, ClickHouse, and other SQL engines reached by a connection URI, see {doc}`/integrations/databases/index`.

```{toctree}
:hidden:

snowflake
bigquery
databricks
```
