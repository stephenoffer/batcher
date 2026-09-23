# Integrations

Batcher works with the stack you already run. It reads the Kafka topic, the Snowflake query, the Iceberg table, or the Postgres database where your data already lives, and writes results back into them, without an export step or a staging copy in between. It also sits beside the libraries already in your process, and inside the scheduler that runs your pipelines. This section has one page per integration.

The integrations divide by *where the other system is*, and the difference is not cosmetic. A connector reaches a system over a network, so it splits the read, pushes down filters, and resolves credentials on the worker. An in-process library shares memory with Batcher, so the handoff is a pointer rather than a transfer. The two halves are documented separately because almost nothing you need to know about one applies to the other.

Connectors speak Arrow and divide their sources into splits that read in parallel, whether a split is a Kafka partition, a BigQuery read stream, a Delta data file, or a MongoDB `_id` range. Filters push down to the system that holds the data wherever it can evaluate them, so a warehouse filters before it returns rows and a lakehouse table skips files its metadata rules out. Most connector credentials accept secret references such as `env:NAME` and `file:PATH`, resolved on the worker that opens the connection, so a password never travels in the plan.

## Systems over the network

The figure arranges the six connector groups around the engine. The four that hold data are on the left, where connectors read in parallel and write back wherever a sink exists. Compute and observability are on the right: one schedules the work and consumes what it produces, the other receives the signals it emits.

![On the left, four groups where data lives, Streams (Kafka, Kinesis, Pulsar, Pub/Sub), Warehouses (Snowflake, BigQuery, Databricks), Lakehouse (Delta Lake, Iceberg, Hudi), and Databases (SQL, key-value, MongoDB), connect to the Batcher engine through parallel reads, and results are written back to them where a connector has a sink. On the right, Compute and ML (Ray, schedulers, PyTorch) handles scheduling, and Observability receives metrics, traces, and lineage as signals. Every connector is built on the same public Source, Sink, and Split contracts, so a system not listed plugs in the same way. Not every connector writes: BigQuery and Databricks have no sink.](/_static/diagrams/integrations_hub.svg)

Each connector page opens with a capability table: what it reads and writes, the pip extra, how it splits, and what pushes down. The `warehouse`, `lakehouse`, `nosql`, and `streaming` extras install a whole group at once.

## Libraries in your process

Two more groups are not connectors at all. {doc}`/integrations/dataframes/index` covers Polars, pandas, DuckDB, PyArrow, and NumPy, which share Arrow with Batcher, so a table crosses between them as a pointer handoff rather than a copy. {doc}`/integrations/orchestration/index` covers Airflow, Dagster, and Prefect, where there is nothing to install at all: Batcher has no daemon, so a scheduled task is a Python function that imports it.

### Model runtimes

A model library is a third shape again: it runs *inside* the worker, on the batch the engine already assembled, so there is no serving fleet between the data and the model. Those are documented with the ML guides rather than here, because what you need to know about them is the inference pipeline rather than a connection.

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
SQL databases, key-value stores, MongoDB, and Elasticsearch, with writes back into them.
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
Airflow, Dagster, and Prefect. No operator to install, and the retry story that matters.
:::

::::

The following table maps each group to what it covers in full:

| Group | Covers |
|---|---|
| {doc}`/integrations/streams/index` | Kafka, Amazon Kinesis, Apache Pulsar, Google Cloud Pub/Sub, Azure Event Hubs, and the Avro, JSON, and Protobuf payloads they carry |
| {doc}`/integrations/warehouses/index` | Snowflake, BigQuery, and Databricks |
| {doc}`/integrations/lakehouse/index` | Delta Lake, Apache Iceberg, and Apache Hudi |
| {doc}`/integrations/databases/index` | SQL databases over one connection URI, writing back to a database, DynamoDB, Cassandra, ScyllaDB, Redis, HBase, MongoDB, and Elasticsearch |
| {doc}`/integrations/compute/index` | Ray, batch schedulers, PyTorch, Hugging Face, and MLflow |
| {doc}`/integrations/observability/index` | Prometheus and Grafana, OpenTelemetry traces, and OpenLineage |
| {doc}`/integrations/dataframes/index` | Polars, pandas, DuckDB, PyArrow, and NumPy, in the same process and over the same Arrow |
| {doc}`/integrations/orchestration/index` | Airflow, Dagster, and Prefect, plus what makes a task safe to retry |

Every connector is built on the same public contracts: {py:class}`Source <batcher.io.Source>`, {py:class}`Sink <batcher.io.Sink>`, {py:class}`Split <batcher.io.Split>`, and the format registry they register into. A system not listed here plugs in the same way, as {doc}`custom connectors </user-guide/moving-data/custom-connectors>` describes.

## See also

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
compute/index
observability/index
dataframes/index
orchestration/index
```
