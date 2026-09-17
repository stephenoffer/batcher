# BigQuery

This page covers reading BigQuery tables and query results. {py:meth}`bt.read.bigquery(...) <batcher.api.io_namespace.reader.Reader.bigquery>` reads through the BigQuery Storage Read API, which serves a table as parallel Arrow streams, and pushes both your filters and your column selection into the read session so BigQuery scans only what the query uses.

The following table summarizes the connector:

| | |
| --- | --- |
| Read | `bt.read.bigquery(query, project=...)` or `bt.read.bigquery(table=..., project=...)` |
| Write | No sink. Write Parquet to Cloud Storage and load it with `bq load`. |
| Extra | `pip install 'batcher-engine[bigquery]'` |
| Parallelism | One split per Storage Read API stream, `max_streams=8` by default |
| Pushdown | Predicates become the session's `row_restriction`. Projection becomes `selected_fields`. |
| Statistics | Exact row count and size from `__TABLES__` for a table read |
| Credentials | The ambient `google.auth` environment |

The extra installs `google-cloud-bigquery-storage` and `google-cloud-bigquery`. Credentials come from `GOOGLE_APPLICATION_CREDENTIALS`, an application-default login, or the service account attached to the node. Nothing credential-bearing is passed as a keyword or logged.

## Read a table

A table read needs a billing project and a fully-qualified `project.dataset.table`:

```python
# docs: skip
import batcher as bt
from batcher import col

events = bt.read.bigquery(
    table="acme-data.analytics.events",
    project="acme-billing",
    max_streams=32,
)
purchases = events.filter(col("event_type") == "purchase").select("user_id", "ts").collect()
```

`project` is the project the read session is billed and rate-limited against, which is often not the project that owns the data. The `filter` and the `select` both reach BigQuery: the session opens with `row_restriction` set to the pushed predicate and `selected_fields` set to the columns the plan needs, so the other columns and rows never leave the service.

A table read is also cheap to plan. The schema comes back in the `create_read_session` response before any stream is read, and the row count and byte size come from the dataset's `__TABLES__` metadata view, which scans no bytes. Kyber gets an exact cardinality for the table before execution starts.

## Read a query

The positional argument is the query:

```python
# docs: skip
top = bt.read.bigquery(
    """
    SELECT user_id, COUNT(*) AS n
    FROM `acme-data.analytics.events`
    WHERE _PARTITIONDATE >= '2026-01-01'
    GROUP BY user_id
    """,
    project="acme-billing",
    max_streams=16,
)
```

The Storage Read API reads tables, not queries. A query read therefore runs the SQL as a job into BigQuery's anonymous destination table and opens a read session on that table. A pushable predicate is wrapped around your SQL before the job runs.

:::{important}
A query read runs its job more than once. Building the dataset opens a read session to learn the schema, which for a query means running the query, and `collect()` runs it again. Read a table wherever you can, or materialize the result into a table with your own job and read that.
:::

## How it parallelizes

A {py:class}`Source <batcher.io.Source>` divides into {py:class}`Split <batcher.io.Split>` objects, and a split is the unit of read parallelism. One `create_read_session(data_format=ARROW, max_stream_count=N)` call returns up to `N` independent streams, and `splits()` returns one split per stream. Each split carries only the stream name, so it ships to any worker, which builds its own read client and pulls Arrow pages straight from the API.

The server decides how many streams to return based on table size and capacity, so it may return fewer than you ask for. Near your worker count is the right target for a large scan, and more streams than readers buys nothing. Streams read roughly equal shares of the table, not exactly equal ones.

## Choose the columns yourself

Kyber's projection covers the common case. When you want to pin the column set explicitly, or pass a filter in BigQuery's own syntax, set the session options on the source:

```python
# docs: skip
narrow = bt.read.bigquery(
    table="acme-data.analytics.events",
    project="acme-billing",
    selected_fields=("user_id", "ts"),
    row_restriction="event_type = 'purchase'",
)
```

An explicit `selected_fields` replaces the pushed projection rather than intersecting with it. An explicit `row_restriction` is combined with a pushed predicate using `AND`. Nested and repeated fields arrive as Arrow structs and lists. Use the {py:class}`.struct <batcher.plan.expr_ir.namespaces.collections._StructNamespace>` and {py:class}`.list <batcher.plan.expr_ir.namespaces.collections._ListNamespace>` accessors to reach into them.

## Requirements and limitations

The Storage Read API is metered per project, on read throughput and concurrent streams. A job that fans out to hundreds of streams across a large cluster can exhaust the quota, and the symptom is `ResourceExhausted` on `ReadRows` inside a worker partway through a scan, not a failure at planning time. Keep `max_streams` proportionate to the workers that consume the streams.

Read sessions expire. Batcher creates the session at planning time, so a scan whose splits wait behind a long queue, on a busy cluster or an autoscaler still warming up, can find its streams gone. A stream that fails mid-read is re-read from its start by the retry, not resumed at the offset it reached.

There's no BigQuery sink. A {py:exc}`BackendError <batcher.BackendError>` at construction means the client libraries are missing, or that neither `query` nor `table` was supplied. A governance policy matches a `table=` read by its table name. A query read has no table name to match.

## See also

- {doc}`Snowflake </integrations/warehouses/snowflake>`: the warehouse connector that also writes.
- {doc}`Databricks </integrations/warehouses/databricks>`: Unity Catalog tables read straight from their Delta files.
- {doc}`Reading data </user-guide/moving-data/reading-data>`: the reader surface.
- {doc}`Multi-source join </cookbook/data-engineering/modeling/multi-source-join>`: a BigQuery table joined against the lake without staging either side.
- {doc}`Incremental ingest </cookbook/data-engineering/ingest/incremental-ingest>`: reading only new partitions.
- {doc}`Reading and writing </api/relational/io>`: the full API reference.
