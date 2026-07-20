# BigQuery

BigQuery is read-only in Batcher. `bt.read.bigquery(...)` pulls a table or a query result
through the Storage Read API as parallel Arrow streams. There is no BigQuery sink; to land
results in BigQuery, write Parquet to GCS and load it, or use `bq load`.

| | |
| --- | --- |
| **Read** | `bt.read.bigquery(query, project=...)` or `bt.read.bigquery(table=..., project=...)` |
| **Write** | Not supported |
| **Extra** | `pip install 'batcher-engine[bigquery]'` |
| **Parallelism** | One split per Storage Read API stream; `max_streams` defaults to 8 |
| **Pushdown** | Predicates become `row_restriction`. Projection only via `selected_fields=`. |
| **Credentials** | The ambient `google.auth` environment. Nothing is passed as a keyword. |

```bash
pip install 'batcher-engine[bigquery]'
```

That brings in `google-cloud-bigquery-storage` and `google-cloud-bigquery`. Credentials come
from the ambient `google.auth` environment: `GOOGLE_APPLICATION_CREDENTIALS`, an
application-default login, or the service account attached to the node. Nothing
credential-bearing is passed as a keyword.

## Reading

The source needs a billing project *and* a table or a query. The reliable spelling passes both
as keywords:

::::{tab-set}

:::{tab-item} A table

```python
# docs: skip
import batcher as bt
from batcher import col

events = bt.read.table(
    "bigquery",
    project="acme-billing",
    table="acme-data.analytics.events",
    selected_fields=("user_id", "event_type", "ts"),
    max_streams=32,
)
recent = events.filter(col("event_type") == "purchase").collect()
```

`table` is the fully-qualified `project.dataset.table` of the data. `project` is the project the
read session is billed and quota'd against, which is often a different one.
:::

:::{tab-item} A query

```python
# docs: skip
top = bt.read.table(
    "bigquery",
    project="acme-billing",
    query="""
        SELECT user_id, COUNT(*) AS n
        FROM `acme-data.analytics.events`
        WHERE _PARTITIONDATE >= '2026-01-01'
        GROUP BY user_id
    """,
    max_streams=16,
)
```

The Storage Read API cannot read a query, only a table. So a query read runs the SQL as a job
into an anonymous destination table, then opens a read session on *that*. You pay for the query
job, then read its result in parallel.
:::

::::

:::{note}
The positional argument to `bt.read.bigquery(...)` is the **query**, so
`bt.read.bigquery("SELECT ...", project="acme-billing")` reads exactly what it appears to read.
To read a whole table instead, pass `table=` and leave the positional slot empty:
`bt.read.bigquery(table="acme-data.analytics.events", project="acme-billing")`.
:::

:::{important}
A query read is paid for more than once. Constructing the `Dataset` needs a schema, and the
source gets one by opening a read session and reading the first stream, which for a query read
means running the query. `collect()` then runs it again. **Read tables, not queries,** wherever
you can.
:::

When you cannot, materialize the result into a real table with your own job and point Batcher
at the table.

## How it parallelizes

A `Source` divides into `Split`s, and a split is the unit of read parallelism. One
`create_read_session(data_format=ARROW, max_stream_count=N)` call returns up to `N` independent
read streams over the table, and `splits()` returns one split per stream. Each split is nothing
but a stream name, a string, so it ships to a worker cleanly, and the worker builds its own read
client and pulls Arrow batches straight from the API.

`max_streams` defaults to 8, which is low for a large table and pointless for a small one.

:::{dropdown} Three things to know before you raise `max_streams`
1. The server may return *fewer* streams than you ask for. It decides based on table size and
   available capacity; asking for a thousand does not get you a thousand.
2. Somewhere near your worker count is the right target for a big scan. 32 to 64 is a sane range
   for a wide table on a real cluster.
3. Streams read approximately equal shares, not exactly equal ones. Expect stragglers.
:::

## Push the projection down, or pay for the columns

Two pushdowns matter, and they behave differently.

Predicates are pushed. A `filter` that Kyber can translate becomes the read session's
`row_restriction`, evaluated server-side before a byte moves. You can also set `row_restriction=`
yourself; the two are combined with `AND`.

Column projection is not pushed automatically. A `select` after the read is applied to the Arrow
table once it arrives. The server-side column prune is `selected_fields=`, which you pass to the
source. It is the difference between scanning three columns and scanning ninety, and BigQuery
charges you for the difference.

| What you write | Where it runs |
| --- | --- |
| `selected_fields=(...)` | The server, before a byte moves |
| `row_restriction="..."`, or a `filter` Kyber can push | The server, as the session's `row_restriction` |
| `.select(...)` after the read | Your process, on the Arrow table that already arrived |

```python
# docs: skip
narrow = bt.read.table(
    "bigquery",
    project="acme-billing",
    table="acme-data.analytics.events",
    selected_fields=("user_id", "ts"),          # scanned server-side
    row_restriction="event_type = 'purchase'",  # filtered server-side
)
```

## Quotas, and the failure modes they produce

The Storage Read API is metered per project, and the quotas that bite are on read throughput and
concurrent streams, not on the number of sessions. A job that fans out to a few hundred streams
across a big Ray cluster can exhaust the project's read quota, and the symptom is
`ResourceExhausted` on `ReadRows` inside a worker, halfway through a scan, rather than a clean
failure at planning time.

Keep `max_streams` proportionate to the workers that will actually consume them. There is no
benefit to more streams than readers.

The other one to know is that a read session expires, and streams belonging to an expired session
fail. Batcher creates the session at planning time and the splits are consumed later, so a scan
whose workers sit behind a long queue, on a busy cluster or an autoscaler that is still warming
up, can find its streams dead by the time they run. Short queues, or a re-plan.

Two smaller ones. A stream that fails mid-read is re-read from the start of that stream by
whatever retry runs it, not resumed at the offset it reached. And nested or repeated fields come
back as Arrow structs and lists, so use the `.struct` and `.list` accessors; nothing is flattened
for you.

A `BackendError` at construction means the client libraries are missing, or that you supplied
neither `query=` nor `table=`.

## See also

- [Reading data](../user-guide/reading-data.md): the guided tour of the reader surface.
- [Multi-source join](../examples/data-engineering/multi-source-join.md): a BigQuery table
  joined against the lake without staging either side.
- [Incremental ingest](../examples/data-engineering/incremental-ingest.md): reading only the
  new partitions, which is the cheapest read there is.
- [Reading and writing](../api/io.md): the full reader/writer surface.
- [Snowflake](snowflake.md): the other big warehouse, and the only one Batcher writes.
- [Databricks](databricks.md): also read-only, also credential-vended, but the read lands on Delta
  files rather than a proprietary API.
