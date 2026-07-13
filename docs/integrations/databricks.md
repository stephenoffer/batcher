# Databricks

`bt.read.databricks(table)` reads a Unity Catalog table. A Databricks managed table *is* a Delta
table in your own cloud storage, so the fast path skips the SQL warehouse entirely: Unity vends
short-lived, table-scoped storage credentials, and Batcher reads the Delta files directly. No
cluster spins up. Nothing queues.

Databricks is read-only in Batcher. There is no Databricks sink; writing is covered at the bottom
of this page.

| | |
| --- | --- |
| **Read** | `bt.read.databricks(table)` direct, or `bt.read.table("databricks", query=...)` via a warehouse |
| **Write** | Not supported. Write Delta to an external location, or land files for `COPY INTO`. |
| **Extra** | `pip install 'batcher-engine[databricks]'` |
| **Parallelism** | Direct: one split per Delta data file. Warehouse: a single split. |
| **Pushdown** | Predicates on both paths. Projection on neither. |
| **Credentials** | Unity vends short-lived, table-scoped storage credentials at plan time |

```
pip install 'batcher-engine[databricks]'
```

That installs the Databricks SDK (for credential vending) and `databricks-sql-connector` (for the
warehouse fallback).

## The two read paths

::::{tab-set}

:::{tab-item} Direct (Delta files)

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

All three arguments are required for this path: the fully-qualified `catalog.schema.table`, the
workspace URL, and a token. Miss one and the source falls through to the warehouse configuration
check, and raises `BackendError` if that is not satisfied either.
:::

:::{tab-item} Warehouse (SQL)

```python
# docs: skip
report = bt.read.table(
    "databricks",
    query="SELECT region, SUM(amount) AS total FROM main.sales.orders GROUP BY region",
    server_hostname="acme.cloud.databricks.com",
    http_path="/sql/1.0/warehouses/abc123",
    access_token="dapi...",
)
```

For SQL the direct path cannot express: a view, a `JOIN` you want Photon to run, or a table
Batcher has no storage access to. It is a single split, and the section below spells out what
that costs you.
:::

::::

Under the hood of the direct path, the SDK looks up the table, calls Unity's
`temporary_table_credentials` API for `READ` access, gets back the table's storage location plus a
cloud-specific credential block (AWS keys, an Azure SAS or AAD token, a GCP OAuth token), and hands
both to the Delta reader as storage options. From there it is an ordinary Delta scan.

The token needs `SELECT` on the table and `EXTERNAL USE SCHEMA` on its schema, which is the Unity
privilege that specifically authorizes reading a table's files from outside Databricks. Without it
the vend call fails, and what comes back is a wrapped `BackendError` from the SDK rather than a
helpful permissions message.

## How it parallelizes

A `Source` divides into `Split`s, and a split is the unit of read parallelism. On the direct path
the splits are Delta's splits: one per data file (or row-group range), taken from the transaction
log. That is real parallelism across a whole cluster, and it is the same machinery
[`bt.read.delta`](delta-lake.md) uses.

The transaction log also gives you file skipping. A predicate that Kyber can push is threaded into
the Delta source, which compares it against the per-file min/max statistics in the log and never
opens the files that cannot match. This is why the predicate belongs in a `filter` rather than in a
downstream `map_batches`: a well-partitioned table with a selective filter reads a fraction of its
files.

## Credential vending, and when it expires

The credentials Unity vends are short-lived, on the order of an hour, and they are vended once, at
planning time, then embedded in the splits that ship to the workers.

For a scan that finishes inside the credential lifetime, that is exactly right. No long-lived cloud
key exists anywhere, and the grant is scoped to one table.

:::{warning}
For a scan that does not, it is a landmine. A multi-hour read, or a job whose splits sit in a queue
while the cluster autoscales, can find its credentials expired and start taking 403s from the object
store partway through. Nothing refreshes them mid-query. If you have a scan that long, split it into
several reads over partition ranges rather than hoping the token outlives the job.
:::

Personal access tokens also expire on their own schedule, and a rotated token silently breaks a
scheduled job. Use a service principal.

## What the warehouse fallback costs

:::{important}
Understand what you are giving up. That read is a **single split**. The query runs on the SQL
warehouse, the whole result is fetched to one worker with `fetchall_arrow` (over Cloud Fetch, so the
transfer itself is Arrow files from cloud storage, not a row cursor), and it lands as one Arrow table
in memory. There is no fan-out. A result that does not fit on one node does not fit at all.
:::

:::{important}
It also runs the query twice, once to infer the schema when the `Dataset` is constructed and once
when you collect it. A cold warehouse takes seconds to start before either.
:::

Use this path for small results: a dimension table, a lookup, an aggregate. Use the direct path for
anything large.

Predicates do get pushed here too. A `filter` that Kyber can push becomes a `WHERE` wrapped around
your query, so the warehouse filters before Cloud Fetch. The projection does not: `select` is applied
after the fetch, so name your columns in the SQL.

## Writing

There is no Databricks sink, and credential vending requests `READ` access only. Two ways to get
results back into the lakehouse:

1. Write Delta to an external location. If the target is an external table whose storage you can
   reach with your own credentials, `ds.write.delta("s3://.../orders")` is a normal transactional
   Delta commit, and Unity sees the new data on its next read. This does not work for a *managed*
   table; do not write into managed storage behind Unity's back.
2. Write Parquet or Delta to a landing path and let Databricks ingest it. A `COPY INTO` or an Auto
   Loader job on the Databricks side keeps the catalog as the single writer, which is the
   arrangement Unity is designed for.

## Failure modes worth knowing

:::{warning}
Deletion vectors and column mapping are Delta reader features, and support depends on what the
underlying Delta reader implements. A table with deletion vectors enabled can return rows a
Databricks query would not. Check before you trust a reconciliation.
:::

Unity can front Iceberg and foreign tables. The direct path assumes Delta and will fail on those; use
the warehouse fallback, or [`bt.read.iceberg`](iceberg.md) with the appropriate catalog.

`BackendError: failed to vend Unity Catalog credentials` is almost always the missing
`EXTERNAL USE SCHEMA` privilege, an expired token, or a workspace URL with a trailing path. The
wrapped exception carries the real reason.

:::{dropdown} Which path am I on?
| | Direct | Warehouse |
| --- | --- | --- |
| Reached by | `workspace=` + `token=` + a table name | `server_hostname=` + `http_path=` + `access_token=` |
| Splits | One per Delta data file | One, always |
| Compute | None. The files are read straight from object storage. | The SQL warehouse, cold-starting if idle |
| Runs the query twice | No query to run | Yes, once for the schema and once for the rows |
| Good for | Anything large | A dimension table, a lookup, an aggregate |
:::

## See also

- [Delta Lake](delta-lake.md): the format underneath, and the writer.
- [Lakehouse](../user-guide/lakehouse.md): time travel, merges, maintenance.
- [Cloud storage](../user-guide/cloud-storage.md): credentials and object-store paths.
- [Incremental ingest](../examples/data-engineering/incremental-ingest.md): reading a Unity
  table's new partitions on a schedule.
- [Reading and writing](../api/io.md): the full reader/writer surface.
- [Iceberg](iceberg.md): for the Unity tables the direct path will not open.
