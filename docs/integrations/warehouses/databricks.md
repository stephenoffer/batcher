# Databricks

This page covers reading Databricks tables. A Unity Catalog managed table is a Delta table in your own cloud storage, so {py:meth}`bt.read.databricks(table) <batcher.api.io_namespace.reader.Reader.databricks>` skips the SQL warehouse entirely: Unity vends short-lived, table-scoped storage credentials, and Batcher reads the Delta files directly, in parallel, with file skipping. No cluster spins up and nothing waits in a warehouse queue. For SQL that must run on Databricks, a warehouse path streams the result back as Arrow.

The following table summarizes the connector:

| | |
| --- | --- |
| Read | `bt.read.databricks(table, workspace=..., token=...)` direct, or {py:meth}`bt.read.table("databricks", query=...) <batcher.api.io_namespace.reader.Reader.table>` through a SQL warehouse |
| Write | No sink. Write Delta to an external location, or land files for `COPY INTO`. |
| Extra | `pip install 'batcher-engine[databricks]'` |
| Parallelism | Direct: Delta's splits, one per data file. Warehouse: a single split. |
| Pushdown | Direct: predicates prune files, columns are pruned per file. Warehouse: predicates and projection fold into the SQL. |
| Credentials | Direct: Unity vends storage credentials at plan time. Warehouse: an access token. |

The extra installs the Databricks SDK, for credential vending, and `databricks-sql-connector`, for the warehouse path.

## Read a table directly

Pass the fully-qualified `catalog.schema.table`, the workspace URL, and a token:

```python
# docs: skip
import batcher as bt
from batcher import col

orders = bt.read.databricks(
    "main.sales.orders",
    workspace="https://acme.cloud.databricks.com",
    token="dapi...",
)
big = orders.filter(col("amount") > 1_000).select("order_id", "customer_id", "amount")
print(big.count())
```

All three are required for this path. Without them the source checks for a warehouse configuration instead, and raises {py:exc}`BackendError <batcher.BackendError>` if that's incomplete too.

The SDK looks up the table, calls Unity's `temporary_table_credentials` API for `READ` access, and gets back the table's storage location with a cloud-specific credential block: AWS keys, an Azure SAS or AAD token, or a Google Cloud OAuth token. Batcher hands both to its {doc}`Delta reader </integrations/lakehouse/delta-lake>`, and from there it's an ordinary Delta scan. Deletion vectors are applied per file, the same as on a path-addressed Delta read.

The token needs `SELECT` on the table and `EXTERNAL USE SCHEMA` on its schema, the Unity privilege that authorizes reading a table's files from outside Databricks. Use a service principal rather than a personal access token, which expires on its own schedule and breaks a scheduled job when rotated.

### How the direct path parallelizes

A {py:class}`Source <batcher.io.Source>` divides into {py:class}`Split <batcher.io.Split>` objects, the unit of read parallelism. On the direct path the splits come from the Delta transaction log, one per data file, so a large table fans out across a whole cluster.

The log also gives you file skipping. A predicate Kyber can push is compared against the per-file min and max statistics in the log, and files that can't match are never opened. Put the predicate in a `filter`, not in a downstream `map_batches`, and a selective filter on a well-partitioned table reads a fraction of its files. Columns are pruned when each worker reads a file's footer.

## Read through a SQL warehouse

Use the warehouse path for SQL the direct path can't express: a view, a join you want Photon to run, or a table Batcher has no storage access to.

```python
# docs: skip
report = bt.read.table(
    "databricks",
    query="SELECT region, SUM(amount) AS total FROM main.sales.orders GROUP BY region",
    server_hostname="acme.cloud.databricks.com",
    http_path="/sql/1.0/warehouses/abc123",
    access_token="env:DATABRICKS_TOKEN",
)
```

`access_token` accepts a secret reference such as `env:NAME` or `file:PATH`, resolved on the worker when the connection opens. The schema comes from a zero-row `WHERE 1 = 0` probe, so building the dataset doesn't run your query. Pushed predicates and the projected columns fold into the SQL the split carries, so the warehouse filters and prunes before Cloud Fetch returns anything. The result streams back in Arrow chunks of 65,536 rows through `fetchmany_arrow`.

The warehouse path is one split. The query runs on the warehouse and one worker fetches the result, so there's no fan-out on the read. Use it for a dimension table, a lookup, or an aggregate, and use the direct path for anything large. A cold warehouse takes seconds to start.

The following table compares the two paths:

| | Direct | Warehouse |
| --- | --- | --- |
| Reached by | `table` + `workspace=` + `token=` | `query=` + `server_hostname=` + `http_path=` + `access_token=` |
| Splits | One per Delta data file | One |
| Compute | None. Files are read straight from object storage. | The SQL warehouse, cold-starting if idle |
| Good for | Anything large | A dimension table, a lookup, an aggregate |

## Write results back

There's no Databricks sink, and credential vending requests `READ` access only. Get results back into the lakehouse in either of the following ways:

1. Write Delta to an external location. If the target is an external table whose storage you can reach with your own credentials, {py:meth}`ds.write.delta("s3://.../orders") <batcher.api.io_namespace.writer.Writer.delta>` is a normal transactional Delta commit, and Unity sees the new data on its next read. Don't do this for a managed table, whose storage belongs to Unity.
1. Write Parquet or Delta to a landing path and let a `COPY INTO` or Auto Loader job ingest it. The catalog stays the single writer, which is the arrangement Unity is designed for.

## Requirements and limitations

Vended credentials are short-lived. They're vended once at planning time and embedded in the splits sent to workers, and nothing refreshes them mid-query. A multi-hour read, or a job whose splits wait while a cluster autoscales, can start taking 403 errors from the object store partway through. Split a scan that long into several reads over partition ranges.

The direct path reads Delta. For a table Unity fronts in another format, use the warehouse path.

`BackendError: failed to vend Unity Catalog credentials` usually means a missing `EXTERNAL USE SCHEMA` privilege, an expired token, or a workspace URL with a trailing path. The wrapped exception carries the real reason.

## See also

- {doc}`Delta Lake </integrations/lakehouse/delta-lake>`: the format underneath, and the writer.
- {doc}`Snowflake </integrations/warehouses/snowflake>` and {doc}`BigQuery </integrations/warehouses/bigquery>`: the other warehouse connectors.
- {doc}`Lakehouse </user-guide/moving-data/lakehouse>`: time travel, merges, and maintenance.
- {doc}`Cloud storage </user-guide/moving-data/cloud-storage>`: credentials and object-store paths.
- {doc}`Incremental ingest </cookbook/data-engineering/ingest/incremental-ingest>`: reading a Unity table's new partitions on a schedule.
- {doc}`Reading and writing </api/relational/io>`: the full API reference.
