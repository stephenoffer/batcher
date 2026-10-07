# Integrations

Batcher reads the Kafka topic, the Snowflake query, the Iceberg table, or the Postgres database where your data already lives, and writes results back into it. There's no export step and no staging copy. Batcher also sits beside the dataframe libraries in your process and inside the scheduler that runs your pipelines.

Every connector speaks Arrow. It splits its source so the pieces read in parallel, whether a piece is a Kafka partition, a BigQuery read stream, a Delta data file, or a MongoDB `_id` range. Filters run where the data lives. Credentials accept secret references such as `env:NAME` and `file:PATH`, which resolve on the worker, so a password never travels in the plan.

## A taste of each

Reading is a `bt.read.<format>` call. Writing, where a sink exists, is `ds.write.<format>`. The tabs below run on local files and in-process libraries:

::::{tab-set}
:::{tab-item} Lakehouse

```python
import os
import tempfile

import batcher as bt

table = os.path.join(tempfile.mkdtemp(), "events")
bt.from_pydict({"id": [1, 2], "amount": [10, 20]}).write.delta(table)
bt.from_pydict({"id": [3], "amount": [30]}).write.delta(table, mode="append")

print(bt.read.delta(table).sort("id").to_pydict()["id"])
# [1, 2, 3]
print(bt.read.delta(table, version=0).sort("id").to_pydict()["id"])
# [1, 2]
```

:::
:::{tab-item} DuckDB

```python
import duckdb

con = duckdb.connect()
con.execute("CREATE TABLE orders AS SELECT * FROM (VALUES (1, 9.5::DOUBLE), (2, 20.0), (3, 41.0)) v(id, amount)")

orders = bt.from_duckdb(con, "SELECT * FROM orders")
print(orders.filter(bt.col("amount") > 10).sort("id").to_pydict())
# {'id': [2, 3], 'amount': [20.0, 41.0]}
```

:::
:::{tab-item} DataFrames

```python
import polars as pl

spend = pl.DataFrame({"user": ["a", "b", "a"], "spend": [10.0, 2.0, 7.0]})
totals = bt.from_polars(spend).group_by("user").agg(total=bt.col("spend").sum()).sort("user")
print(totals.to_pydict())
# {'user': ['a', 'b'], 'total': [17.0, 2.0]}
print(type(totals.to_pandas()).__name__, type(totals.to_arrow()).__name__)
# DataFrame Table
```

:::
:::{tab-item} Stream

```python
# docs: skip
orders = bt.read.kafka("orders", bootstrap_servers="broker-1:9092", value_format="json")
totals = orders.group_by(bt.col("value").struct.field("user")).agg(n=bt.count())
```

:::
::::

## Systems over the network

The figure arranges the six connector groups around the engine. The four that hold data sit on the left. Compute, which schedules the work, and observability, which receives its signals, sit on the right.

![On the left, four groups where data lives, Streams (Kafka, Kinesis, Pulsar, Pub/Sub), Warehouses (Snowflake, BigQuery, Databricks), Lakehouse (Delta Lake, Iceberg, Hudi), and Databases (SQL, key-value, MongoDB), connect to the Batcher engine through parallel reads, and results are written back to them where a connector has a sink. On the right, Compute and ML (Ray, schedulers, PyTorch) handles scheduling, and Observability receives metrics, traces, and lineage as signals. Every connector is built on the same public Source, Sink, and Split contracts, so a system not listed plugs in the same way. Not every connector writes: BigQuery and Databricks have no sink.](/_static/diagrams/integrations_hub.svg)

Each connector page opens with a capability table: what it reads and writes, the pip extra, how it splits, and what pushes down. The `warehouse`, `lakehouse`, `nosql`, and `streaming` extras install a whole group at once.

## Libraries in your process

Two more groups aren't connectors. {doc}`/integrations/dataframes/index` covers Polars, pandas, DuckDB, PyArrow, and NumPy, which share Arrow with Batcher, so a table crosses between them as a pointer rather than a copy. {doc}`/integrations/orchestration/index` covers Airflow, Dagster, and Prefect. Batcher has no daemon, so a scheduled task is a Python function that imports it.

### Model runtimes

A model library runs inside the worker, on the batch the engine already assembled, so no serving fleet sits between the data and the model. The ML guides document these:

| Runtime | Page |
| --- | --- |
| scikit-learn, XGBoost, LightGBM, CatBoost | {doc}`/integrations/compute/scikit-learn`, {doc}`/integrations/compute/gradient-boosting` |
| PyTorch, TorchScript | {doc}`/integrations/compute/pytorch` |
| ONNX Runtime, OpenVINO, Triton | {doc}`/ml/inference/runtimes` |
| vLLM, SGLang, and hosted LLM APIs | {doc}`/ml/retrieval/llm/engines` |
| Hugging Face Hub, MLflow registry | {doc}`/integrations/compute/huggingface`, {doc}`/integrations/compute/mlflow` |

## Find your integration

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {octicon}`broadcast;1.1em` Streams
:link: /integrations/streams/index
:link-type: doc
Kafka, Kinesis, Pulsar, Pub/Sub, and Event Hubs as unbounded datasets.
:::

:::{grid-item-card} {octicon}`database;1.1em` Warehouses
:link: /integrations/warehouses/index
:link-type: doc
Snowflake, BigQuery, and Databricks through each service's bulk Arrow protocol.
:::

:::{grid-item-card} {octicon}`stack;1.1em` Lakehouse
:link: /integrations/lakehouse/index
:link-type: doc
Delta Lake, Apache Iceberg, and Apache Hudi tables.
:::

:::{grid-item-card} {octicon}`server;1.1em` Databases
:link: /integrations/databases/index
:link-type: doc
SQL databases, key-value stores, MongoDB, Elasticsearch, and vector stores, with writes back into them.
:::

:::{grid-item-card} {octicon}`globe;1.1em` HTTP APIs and SaaS
:link: /integrations/apis/index
:link-type: doc
Paginated JSON APIs, GraphQL, GitHub, Salesforce, Google Sheets, SharePoint, and Airbyte connectors.
:::

:::{grid-item-card} {octicon}`cpu;1.1em` ML and compute
:link: /integrations/compute/index
:link-type: doc
Ray, batch schedulers, PyTorch, Hugging Face, and MLflow.
:::

:::{grid-item-card} {octicon}`pulse;1.1em` Observability
:link: /integrations/observability/index
:link-type: doc
Prometheus and Grafana, OpenTelemetry traces, and OpenLineage.
:::
:::{grid-item-card} {octicon}`table;1.1em` DataFrames and arrays
:link: /integrations/dataframes/index
:link-type: doc
Polars, pandas, DuckDB, PyArrow, and NumPy, exchanged through Arrow at no copy cost.
:::

:::{grid-item-card} {octicon}`workflow;1.1em` Orchestrators
:link: /integrations/orchestration/index
:link-type: doc
Airflow, Dagster, and Prefect, with no operator to install.
:::

::::

The following table maps each group to what it covers in full:

| Group | Covers |
|---|---|
| {doc}`/integrations/streams/index` | Kafka, Amazon Kinesis, Apache Pulsar, Google Cloud Pub/Sub, Azure Event Hubs, and the Avro, JSON, and Protobuf payloads they carry |
| {doc}`/integrations/warehouses/index` | Snowflake, BigQuery, and Databricks |
| {doc}`/integrations/lakehouse/index` | Delta Lake, Apache Iceberg, and Apache Hudi |
| {doc}`/integrations/databases/index` | SQL databases over one connection URI, writing back to a database, DynamoDB, Cassandra, ScyllaDB, Redis, HBase, MongoDB, Elasticsearch, and the Qdrant, Pinecone, Milvus, and Turbopuffer vector stores |
| {doc}`/integrations/apis/index` | Paginated HTTP JSON APIs, GraphQL, GitHub, Salesforce, Google Sheets, SharePoint and OneDrive, and Airbyte source connectors |
| {doc}`/integrations/compute/index` | Ray, batch schedulers, PyTorch, Hugging Face, and MLflow |
| {doc}`/integrations/observability/index` | Prometheus and Grafana, OpenTelemetry traces, and OpenLineage |
| {doc}`/integrations/dataframes/index` | Polars, pandas, DuckDB, PyArrow, and NumPy, in the same process and over the same Arrow |
| {doc}`/integrations/orchestration/index` | Airflow, Dagster, and Prefect, plus what makes a task safe to retry |

Every connector is built on the same public contracts: {py:class}`Source <batcher.io.Source>`, {py:class}`Sink <batcher.io.Sink>`, {py:class}`Split <batcher.io.Split>`, and the format registry they register into. A system not listed here plugs in the same way, as {doc}`custom connectors </user-guide/moving-data/custom-connectors>` describes.

## See also

For the general reader and writer surface, start here:

- {doc}`Reading data </user-guide/moving-data/reading-data>` and {doc}`Writing data </user-guide/moving-data/writing-data>`: the reader and writer surface every connector plugs into.
- {doc}`Cloud storage </user-guide/moving-data/cloud-storage>`: credentials and object-store paths for files.
- {doc}`Secrets and keys </user-guide/trust/secrets>`: the secret reference schemes.
- {doc}`Custom connectors </user-guide/moving-data/custom-connectors>`: the protocol, for a system not listed.
- {doc}`I/O API </api/relational/io>`: the reference.

```{toctree}
:hidden:

streams/index
warehouses/index
lakehouse/index
databases/index
apis/index
compute/index
observability/index
dataframes/index
orchestration/index
```
