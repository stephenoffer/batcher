# Warehouses

Batcher reads your warehouse through each service's bulk Arrow protocol, never a row-by-row cursor. A Snowflake result arrives as parallel cloud-storage chunks, a BigQuery table as parallel Storage Read API streams, and a Databricks table as its own Delta files. Filters push down on all three, and the schema comes from a zero-row probe or table metadata, so planning costs no scan.

Each read is one call that returns a lazy dataset, ready to join against the lake. These blocks need an account, so they are shown but not executed:

::::{tab-set}
:::{tab-item} Snowflake

```python
# docs: skip
import batcher as bt

conn = {"account": "acme-prod", "user": "svc_batcher", "password": "env:SNOWFLAKE_PASSWORD"}
orders = bt.read.snowflake("SELECT * FROM SALES.ORDERS", connection_kwargs=conn)
```

:::
:::{tab-item} BigQuery

```python
# docs: skip
events = bt.read.bigquery(table="acme-data.analytics.events", project="acme-billing")
purchases = events.filter(bt.col("event_type") == "purchase")
```

:::
:::{tab-item} Databricks

```python
# docs: skip
orders = bt.read.databricks(
    "main.sales.orders", workspace="https://acme.cloud.databricks.com", token="env:DATABRICKS_TOKEN"
)
```

:::
::::

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
