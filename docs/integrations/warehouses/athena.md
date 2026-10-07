# Athena

This page covers reading Amazon Athena query results. {py:meth}`bt.read.athena(query, region=...) <batcher.api.io_namespace.reader.Reader.athena>` is a connection profile over the DB-API reader: it translates Athena's settings into PyAthena's `connect()` keywords and reads through the same path as any `bt.read.sql` read.

:::{warning}
The Athena profile is not yet verified against a live Athena workgroup. See [`tests/PENDING_VERIFICATION.md`](https://github.com/stephenoffer/batcher/blob/main/tests/PENDING_VERIFICATION.md).
:::

The following table summarizes the connector:

| | |
| --- | --- |
| Read | `bt.read.athena(query, region=..., workgroup=..., output_location=..., database=...)` |
| Write | No sink. Write Parquet or Iceberg to S3 and register it in Glue. |
| Extra | `pip install 'batcher-engine[athena]'` |
| Parallelism | One split, read on one worker |
| Pushdown | Projection and predicate, folded into the submitted SQL |
| Credentials | The ambient AWS credential chain, or `profile_name=` |

## Read a query

Athena writes every query's result to S3, so name the result location, or a workgroup whose configuration enforces one. Naming neither is refused before any query is submitted:

```python
# docs: skip
import batcher as bt

events = bt.read.athena(
    "SELECT user_id, event_type, ts FROM events WHERE dt = '2026-10-01'",
    region="us-east-1",
    workgroup="analytics",
    output_location="s3://acme-athena-results/batcher/",
    database="web",
)
```

The settings map onto PyAthena's keywords as the following table shows:

| `bt.read.athena` | PyAthena `connect()` |
| --- | --- |
| `region=` | `region_name` |
| `workgroup=` | `work_group` |
| `output_location=` | `s3_staging_dir` |
| `database=` | `schema_name` |
| `catalog=` | `catalog_name` |
| `profile_name=` | `profile_name` |

Further keywords such as `batch_size=` go to the DB-API reader. The schema comes from a zero-row `WHERE 1 = 0` probe, which Athena still runs as a query.

## Requirements and limitations

The read is one query and one split. PyAthena returns rows as Python objects, which Batcher converts to Arrow one batch at a time, so this path is slower than reading the underlying Parquet directly. For a large table, read its S3 location with {py:meth}`bt.read.parquet <batcher.api.io_namespace.reader.Reader.parquet>` or its Iceberg table with {py:meth}`bt.read.iceberg <batcher.api.io_namespace.reader.Reader.iceberg>`, and keep Athena for views and SQL you want Athena to run.

Each query is billed by the bytes Athena scans. Filters push into the submitted SQL, so a `filter` on a partition column prunes partitions server-side.

## See also

- {doc}`/integrations/databases/databases`: the DB-API reader this profile sits on.
- {doc}`/integrations/databases/vendor-matrix`: routes and type rules for the databases Batcher states coverage for.
- {doc}`Snowflake </integrations/warehouses/snowflake>`, {doc}`BigQuery </integrations/warehouses/bigquery>` and {doc}`Databricks </integrations/warehouses/databricks>`: the other warehouse connectors.
