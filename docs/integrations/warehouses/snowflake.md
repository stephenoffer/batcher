# Snowflake

This page covers reading from and writing to Snowflake. {py:meth}`bt.read.snowflake(query) <batcher.api.io_namespace.reader.Reader.snowflake>` submits a query once and reads its result chunks in parallel as Arrow. {py:meth}`ds.write.snowflake(table) <batcher.api.io_namespace.writer.Writer.snowflake>` loads a dataset into a table.

The following table summarizes the connector:

| | |
| --- | --- |
| Read | `bt.read.snowflake(query, connection_kwargs=...)` |
| Write | `ds.write.snowflake(table, connection_kwargs=..., mode=...)`, with `mode` set to `"overwrite"` (the default) or `"append"` |
| Extra | `pip install 'batcher-engine[snowflake]'` |
| Parallelism | One split per result chunk from `get_result_batches()` |
| Pushdown | Predicates, as a `WHERE` around your query. Projection isn't pushed. |
| Credentials | One declared `auth=` strategy with `account=`, `user=`, `role=`, `warehouse=`, `database=` and `schema=`, or everything in a `connection_kwargs` dict. Both accept secret references. |

## Read a query

Connection settings travel as one `connection_kwargs` dict, passed to `snowflake.connector.connect` on each worker:

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

Anything the connector accepts works in that dict, including key-pair auth, `authenticator="externalbrowser"`, and a `session_parameters` dict.

Any string value in the dict can be a secret reference instead of the secret itself: `env:NAME`, `file:PATH`, `cmd:NAME`, or a key store such as `vault:`, `aws-sm:`, `gcp-sm:`, or `azure-kv:`. The reference is resolved on the worker when the connection opens, so the password never appears in the plan or in a pickled split. {doc}`/user-guide/trust/secrets` covers the schemes.

```python
# docs: skip
conn = {
    "account": "acme-prod",
    "user": "svc_batcher",
    "password": "env:SNOWFLAKE_PASSWORD",
    "warehouse": "ETL_WH",
}
```

## Declare one authentication strategy

:::{warning}
The `auth=` keywords and the load results below are not yet verified against a live Snowflake account. See [`tests/PENDING_VERIFICATION.md`](https://github.com/stephenoffer/batcher/blob/main/tests/PENDING_VERIFICATION.md).
:::

Instead of assembling the connector's keywords yourself, name the strategy with `auth=` and pass its credentials beside the session settings. Batcher checks the combination once, before any connection opens, and folds it into the `connection_kwargs` every worker connects with. A local run and a distributed one therefore authenticate the same way.

The following table lists the strategies and what each needs besides `account=`:

| `auth=` | Needs | Sent to the connector as |
| --- | --- | --- |
| `"password"` | `user=`, `password=` | `user`, `password` |
| `"key_pair"` | `user=`, `private_key_file=`, optionally `private_key_file_pwd=` | `authenticator="SNOWFLAKE_JWT"` plus the key file and passphrase |
| `"oauth"` | `token=` | `authenticator="oauth"`, `token` |
| `"externalbrowser"` | `user=` | `authenticator="externalbrowser"` |

When you leave `auth=` out, the credential you pass picks it: a key file means `key_pair`, a token means `oauth`, and a password means `password`. A credential that belongs to a different strategy, or a keyword that contradicts the same key in an explicit `connection_kwargs`, is refused rather than guessed at.

```python
# docs: skip
orders = bt.read.snowflake(
    "SELECT order_id, amount FROM sales.orders",
    account="acme-prod",
    user="svc_batcher",
    auth="key_pair",
    private_key_file="/etc/secrets/batcher_rsa.p8",
    private_key_file_pwd="env:SNOWFLAKE_KEY_PASSPHRASE",
    role="BATCHER_ETL",
    warehouse="ETL_WH",
)
```

The key file path must exist on every machine that connects. Browser single sign-on works for a read, distributed or not, because only the driver connects and workers fetch result chunks through pre-signed URLs. A distributed write is different: every shard connects from its own worker, which has no browser, so `auth="externalbrowser"` is refused there.

## How it parallelizes

Snowflake's connector exposes `get_result_batches()`. After one query execution it returns a list of `ResultBatch` handles, each a picklable pointer to one chunk of the result sitting in cloud storage. That is exactly Batcher's split model. `splits()` returns one split per chunk, each worker fetches its own chunk as Arrow, and the query never runs again on the workers.

Snowflake's chunking of the result sets the parallelism, not anything you configure. A small result comes back as one chunk and reads on one worker. A large one fans out across as many workers as there are chunks, each pulling from cloud storage without going back through the warehouse. A streaming read with `iter_batches` walks the chunks in turn, so only one chunk is in memory at a time.

## Keep the bill down

Batcher learns the schema from a zero-row probe, the query wrapped in `WHERE 1 = 0`. Snowflake answers that without scanning data, so constructing the dataset costs a round trip rather than a second run of your query. If Snowflake vends no chunk for the empty probe, the reader falls back to running the query and reading its first chunk.

Predicates push down. A `filter` that Kyber can push becomes a `WHERE` wrapped around your query, so the warehouse filters before returning anything, on the distributed path as well as on one node.

Column projection doesn't push down. A `select` after the read runs on chunks that already arrived, so every column your SQL names crosses the network. Name only the columns you want in the query text. It's the single highest-leverage habit on this page.

The following table shows where each part of a read runs:

| What you write | Where it runs |
| --- | --- |
| Columns named in the query text | The warehouse, before anything is returned |
| `.filter(...)` after the read | The warehouse, as a `WHERE` around your query |
| {py:meth}`.select(...) <batcher.Dataset.select>` after the read | Your process, on chunks that already arrived |

A suspended warehouse takes seconds to resume, and the first query of a run pays that. In a latency-sensitive path, keep the warehouse warm or budget for the cold start.

## Write a table

Snowflake folds unquoted identifiers to upper case, and the write creates columns exactly as Arrow names them. Alias columns to the case you want to query them by before you write:

```python
import batcher as bt
from batcher import col

orders = bt.from_pydict({"order_id": [1, 2], "amount": [10.0, 5.5]})

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
manifest = shaped.write.snowflake("ORDERS", connection_kwargs=conn, mode="append")
print(manifest)
```

Each shard is a staged bulk load: `write_pandas` uploads the shard as Parquet to a temporary stage and runs `COPY INTO`. Its load results come back in every written file's `job`: the chunk count, the rows loaded, and the `COPY INTO` result rows, so `manifest.files[0].job["copy_into"]` shows each staged file's status.

The sink loads through `write_pandas` with `auto_create_table=True`, which quotes what it's given. Lowercase Arrow names produce a table whose columns can only be referenced as `"order_id"`, quotes included, for the life of the table.

:::{warning}
`mode` defaults to `"overwrite"`, which replaces the destination table's rows. Pass `mode="append"` to add rows to an existing table. Overwrite is refused on a distributed write with more than one shard, because each shard would replace the rows the previous shard loaded.
:::

Each shard commits its own rows as it finishes, and there's no transaction across shards. A distributed append that dies halfway leaves the rows that already landed. Write to a staging table and swap, or key the data so a re-run is idempotent.

## Requirements and limitations

The write converts each shard to pandas and stages it through `write_pandas`, a full copy in worker memory. That suits millions of rows. For billions, write Parquet to a stage and run `COPY INTO` yourself.

Every split opens its own connection, so a hundred splits means a hundred connections. Watch the account's concurrency limits. Splits also carry `connection_kwargs` to every worker. The values are never logged, but on a shared cluster use secret references or a service account rather than a personal credential.

A result chunk is a handle to storage that Snowflake cleans up, so a split that waits a long time in a queue can find its chunk gone. Keep the gap between planning and reading short.

A Snowflake read is a query rather than a named table, so a governance policy keyed on a table name can't match it on its own. Pass `governed_as="<table>"` to declare the table the query reads, and the policy on that name is applied to the result, in exactly the spelling the policy uses. Inside a {py:obj}`bt.security() <batcher.security>` block, an undeclared query whose text names a table the catalog governs is refused with {py:exc}`AccessDeniedError <batcher.AccessDeniedError>` rather than read ungoverned. The check matches names written in the query, so a view over a governed table isn't caught by it; declare the name. See {doc}`How a table is named </user-guide/trust/table-names>`.

## See also

- {doc}`BigQuery </integrations/warehouses/bigquery>` and {doc}`Databricks </integrations/warehouses/databricks>`: the other warehouse connectors.
- {doc}`Reading data </user-guide/moving-data/reading-data>` and {doc}`Writing data </user-guide/moving-data/writing-data>`: the reader and writer surface.
- {doc}`Incremental ingest </cookbook/data-engineering/ingest/incremental-ingest>`: pulling only new rows, which keeps query cost under control.
- {doc}`Multi-source join </cookbook/data-engineering/modeling/multi-source-join>`: a warehouse table joined against the lake in one plan.
- {doc}`Reading and writing </api/relational/io>`: the full API reference.
