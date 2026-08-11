# Snowflake

Snowflake is one of the two warehouses Batcher both reads and writes. {py:meth}`bt.read.snowflake(query) <batcher.api.io_namespace.reader.Reader.snowflake>`
pulls a query result back as parallel Arrow chunks, and {py:meth}`ds.write.snowflake(table) <batcher.api.io_namespace.writer.Writer.snowflake>` ingests a
dataset into a table.

| | |
| --- | --- |
| **Read** | `bt.read.snowflake(query, connection_kwargs=...)` |
| **Write** | `ds.write.snowflake(table, connection_kwargs=...)`, `mode="append"` (default) or `"overwrite"` |
| **Extra** | `pip install 'batcher-engine[snowflake]'` |
| **Parallelism** | One split per result chunk from `get_result_batches()` |
| **Pushdown** | Predicates, as a `WHERE` around your query. Projection is not pushed. |
| **Credentials** | Everything inside the `connection_kwargs` dict |

```bash
pip install 'batcher-engine[snowflake]'
```

## Reading

Connection credentials travel as one `connection_kwargs` dict, passed verbatim to
`snowflake.connector.connect`:

```python
# docs: skip
import batcher as bt
from batcher import col

conn = {
    "account": "acme-prod",
    "user": "svc_batcher",
    "private_key_file": "/etc/secrets/batcher_rsa.p8",
    "warehouse": "ETL_WH",
    "database": "ANALYTICS",
    "schema": "PUBLIC",
    "role": "BATCHER_ETL",
}

orders = bt.read.snowflake(
    "SELECT order_id, customer_id, amount, ordered_at FROM sales.orders",
    connection_kwargs=conn,
)
recent = orders.filter(col("amount") > 100).collect()
```

:::{warning}
Not `account=`, `user=`, and friends as loose keywords. They belong inside `connection_kwargs`.
:::

Anything the connector accepts works there, including key-pair auth,
`authenticator="externalbrowser"`, and a `session_parameters` dict.

:::{dropdown} The same read, with an SSO login instead of a key
```python
# docs: skip
sso = dict(conn, authenticator="externalbrowser")
sso.pop("private_key_file")

orders = bt.read.snowflake(
    "SELECT order_id, customer_id, amount, ordered_at FROM sales.orders",
    connection_kwargs=sso,
)
```
:::

## How it parallelizes

Snowflake's connector exposes `get_result_batches()`: after one query execution it hands back a
list of `ResultBatch` handles, each a picklable pointer to one chunk of the result set sitting
in cloud storage. That is exactly Batcher's split model. `splits()` returns one split per
chunk, each worker calls {py:meth}`to_arrow() <batcher.Dataset.to_arrow>` on its own handle, and the query is not re-run.

Parallelism is therefore set by Snowflake's chunking of the result, not by anything you
configure. A small result comes back as one chunk and reads on one worker. A large one fans out
across as many workers as there are chunks, pulling from cloud storage in parallel without
going back through the warehouse.

## Cost, which is the whole ballgame here

Three things run up the bill, and all three are avoidable.

:::{important}
**The query is submitted more than once.** `bt.read.snowflake(...)` is not free at
construction: the reader needs a schema, and it gets one by executing the query and inspecting
the first result chunk. Then {py:meth}`collect() <batcher.Dataset.collect>` executes it again. A heavy query behind a
`bt.read.snowflake` call is a heavy query you have paid for at least twice.
:::

Shape the read to be cheap, a table or a narrow view, and do the work in Batcher. Or
materialize an intermediate table in Snowflake first and read *that*.

**Column projection is not pushed down.** A `select` after the read is applied to the Arrow
table *after* the fetch. Every column named in your SQL crosses the network and lands on your
egress bill whether you use it or not. Name the columns you want in the query text. This is the
single highest-leverage thing on this page.

**Row filters are pushed down.** A `filter` that Kyber can push becomes a `WHERE` wrapped
around your query, so the warehouse evaluates it before returning anything. That one works in
your favor, and it means an early `filter` is worth writing even though an early `select` is
not.

| What you write | Where it runs |
| --- | --- |
| Columns named in the query text | The warehouse, before anything is returned |
| `.filter(...)` after the read | The warehouse, as a `WHERE` around your query |
| {py:meth}`.select(...) <batcher.Dataset.select>` after the read | Your process, on the Arrow table that already arrived |

The fourth cost is not Batcher's. A suspended warehouse takes seconds to resume, and that
latency lands on the first query of the run. If you are reading Snowflake in a
latency-sensitive path, keep the warehouse warm or budget for the cold start.

## Writing

```python
import batcher as bt
from batcher import col

orders = bt.from_pydict({"order_id": [1, 2], "amount": [10.0, 5.5]})

# Snowflake folds unquoted identifiers to upper case, but the write creates columns
# exactly as Arrow names them, so name them the way you want to query them.
shaped = orders.select(
    col("order_id").alias("ORDER_ID"),
    col("amount").alias("AMOUNT"),
)
print(shaped.to_pydict())
```

```text
{'ORDER_ID': [1, 2], 'AMOUNT': [10.0, 5.5]}
```

```python
# docs: skip
manifest = shaped.write.snowflake("ORDERS", connection_kwargs=conn)
print(manifest)
```

:::{warning}
The identifier point above is not pedantry. The sink creates the table, and its columns,
through `write_pandas` with `auto_create_table=True`, which quotes what it is given. Feed it
lowercase Arrow column names and you get a table whose columns can only ever be referenced as
`"order_id"`, with the quotes, forever. Alias them to upper case first.
:::

:::{warning}
`mode` defaults to `"append"`, so a re-run adds its rows again rather than replacing them.
Pass `mode="overwrite"` to replace the destination table instead. Overwrite is the
destructive one: the rows that were there are gone.
:::

And it goes through pandas. The Arrow table is converted with {py:meth}`to_pandas() <batcher.Dataset.to_pandas>` and staged by
`write_pandas`, which is a full copy in driver memory. That is why this path is fine for a few
million rows and wrong for a few billion. For bulk loads, write Parquet to a stage and
`COPY INTO` it.

There is no cross-shard transaction. Each shard of a distributed write commits its own rows as
it finishes, and the driver's commit step is a no-op, so a write that dies halfway leaves the
rows that already landed. Plan for it: write to a staging table and swap, or key the data so a
re-run is idempotent.

## Failure modes worth knowing

A result chunk can go stale. `ResultBatch` handles point at cloud storage that Snowflake
garbage-collects, so a split that sits in a queue for a long time before a worker picks it up
can find its chunk gone. Keep the gap between planning and reading short.

Every split opens its own connection. Splits carry credentials and rebuild a connection on the
worker, so a hundred splits means a hundred connections. Watch the account's concurrency
limits.

Numerics widen. `NUMBER(38, x)` maps to an Arrow decimal, so a column declared wider than an
int64 can hold comes back as decimal or float, not int. Cast explicitly if the downstream cares.

Credentials live on the split. They are never logged, but they *are* serialized to every
worker. On a shared cluster that is worth knowing before you reach for a personal token instead
of a service account.

## See also

- {doc}`Reading data </user-guide/moving-data/reading-data>` and {doc}`Writing data </user-guide/moving-data/writing-data>`.
- {doc}`Incremental ingest </cookbook/data-engineering/ingest/incremental-ingest>`: pulling only the
  new rows, which is how you keep the query cost above under control.
- {doc}`Multi-source join </cookbook/data-engineering/modeling/multi-source-join>`: a warehouse table
  joined against the lake, in one plan.
- {doc}`Reading and writing </api/relational/io>`: the full reader/writer surface.
- {doc}`BigQuery </integrations/warehouses/bigquery>`: the other big warehouse. Read-only, with a genuinely parallel
  server-side read path.
- {doc}`Databricks </integrations/warehouses/databricks>`: the third, where the read lands on Delta files.
