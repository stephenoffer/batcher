# Writing to a database

This page covers writing rows back into a SQL database: appending a load, and maintaining an operational table one key at a time with upserts, updates, and deletes. It takes the same connection URI as {doc}`reading </integrations/databases/databases>`, so a read and the write that follows it are spelled the same way.

| | |
| --- | --- |
| **Write** | {py:meth}`ds.write.sql(table, uri=..., mode=...) <batcher.api.io_namespace.writer.Writer.sql>` |
| **Modes** | `append`, `overwrite`, `upsert`, `update`, `delete`, `delete_insert` |
| **Backends** | ADBC for a bulk append where a driver exists, any PEP 249 driver otherwise |
| **Extra** | the per-database driver, such as `pip install psycopg` or `pip install pymysql` |
| **Transactions** | one per write call, and one per shard of a distributed write |
| **Credentials** | `password="env:VAR"` or `"file:/path"`, resolved on the worker |

## What each mode does

An analytics table is loaded. An operational table is *maintained*: a batch of orders is upserted onto the keys it already has, a set of expired sessions is deleted, a scored column is updated in place. `mode` is which of those a write is.

| Mode | Effect | Needs `key_columns` |
| --- | --- | --- |
| `append` | Insert every row. Creates the table if it is absent. | No |
| `overwrite` | Replace the table's contents with these rows. | No |
| `upsert` | Insert each row, or update the one already holding its key. | Yes |
| `update` | Update the rows whose keys match. Inserts nothing. | Yes |
| `delete` | Delete the rows whose keys match. | Yes |
| `delete_insert` | Delete these keys, then insert these rows, in one transaction. | Yes |

`append` is the default, because a write that says nothing about keys should add rows rather than replace a table.

```python
import os
import sqlite3
import tempfile

import batcher as bt

db = os.path.join(tempfile.mkdtemp(), "shop.db")
uri = f"sqlite:///{db}"

orders = bt.from_pydict({"id": [1, 2, 3], "amount": [10.0, 20.0, 30.0]})
orders.write.sql("orders", uri=uri, mode="append", key_columns="id")

changes = bt.from_pydict({"id": [2, 4], "amount": [99.0, 40.0]})
changes.write.sql("orders", uri=uri, mode="upsert", key_columns="id")

print(bt.read.sql("SELECT * FROM orders", uri=uri).sort("id").to_pydict())
```

```text
{'id': [1, 2, 3, 4], 'amount': [10.0, 20.0, 30.0, 40.0]}
```

Order `2` was updated in place and order `4` was inserted. That is one statement per chunk of rows, not one round trip per row.

## Pass `key_columns` on the write that creates the table

An upsert detects a conflict through the target table's `PRIMARY KEY` or a `UNIQUE` constraint. It does not use the column list you passed as a filter, because no database works that way.

So a table created by an earlier `append` that named no `key_columns` has no key to conflict on, and a later upsert against it cannot work. PostgreSQL, SQLite and DuckDB raise. **MySQL raises nothing and duplicates every row instead.**

`key_columns` is therefore accepted on every mode, not only the keyed ones. On `append` it does nothing to the write and everything to the table Batcher creates:

```python
import os
import sqlite3
import tempfile

import batcher as bt

db = os.path.join(tempfile.mkdtemp(), "shop.db")
bt.from_pydict({"id": [1], "amount": [10.0]}).write.sql(
    "orders", uri=f"sqlite:///{db}", mode="append", key_columns="id"
)
conn = sqlite3.connect(db)
print(conn.execute("SELECT sql FROM sqlite_master WHERE name = 'orders'").fetchone()[0])
conn.close()
```

```text
CREATE TABLE "orders" ("id" INTEGER NOT NULL, "amount" REAL, PRIMARY KEY ("id"))
```

Writing into a table you created yourself needs none of this. Batcher creates a table only when one is absent, and `create_table=False` turns even that off.

## Three semantics worth knowing before the first run

Each of these decides whether the result is right, and each is invisible from the mode name.

**An upsert of a subset of columns is a column-level merge, not a row replacement.** Write
`id` and `status` into a table that also has `amount`, and only `status` changes; `amount`
keeps the value it had. That is what `ON CONFLICT DO UPDATE SET` does, and it is usually
what you want — two pipelines can maintain different columns of the same key without
reading each other's. It is also the opposite of `ds.write.mongo`, whose upsert *replaces*
the document, so a column absent from the frame is lost there.

**A repeated key inside one write behaves differently per mode.** `upsert`, `update` and
`delete` bind one statement per row, so the last row for a key wins — quietly, and in frame
order, which a distributed write does not fix. `delete_insert` deletes the key once and then
inserts every row, so a repeated key becomes a repeated row and the target's own key
constraint rejects it. Deduplicate first with
{py:meth}`ds.drop_duplicates(subset=...) <batcher.Dataset.drop_duplicates>` when the source
can carry more than one row per key.

**A column the frame does not have is not written.** For `append` that means the column
takes its database default, or `NULL` — the write does not fail. For `upsert` it means the
column is not updated, which is the merge behavior above. Either way nothing warns, because
both are ordinary SQL; if the frame's shape is meant to match the table, check it.

## Which backend serves the write

You do not choose. Two things decide it, and neither is a preference.

Row-level DML is not expressible as an Arrow ingest: `adbc_ingest` appends a table and has no disposition meaning "update the rows holding these keys". And ADBC does not reach most operational databases. It has drivers for PostgreSQL, SQLite, DuckDB, Snowflake, BigQuery and FlightSQL, which leaves MySQL, MariaDB, Oracle and SQL Server with a PEP 249 driver and nothing else.

| Write | Backend |
| --- | --- |
| `append` or `overwrite`, ADBC driver installed | ADBC, Arrow in bulk |
| `append` or `overwrite`, no ADBC driver installed | the PEP 249 driver for the scheme |
| Any keyed mode | the PEP 249 driver for the scheme |
| `module=` or `connection=` given | that PEP 249 driver |

Availability is part of it, not just the scheme. `sqlite3` ships with Python and `adbc_driver_sqlite` is an extra almost nobody has, so a `sqlite://` write takes the DB-API path unless the ADBC driver is actually installed.

Name a backend explicitly by calling the sink rather than the shortcut: `ds.write(table, "adbc", ...)` or `ds.write(table, "dbapi", ...)`.

### Which PEP 249 driver

There is no single Python driver for PostgreSQL or for MySQL, and which one is installed is usually a decision some other part of your stack made years ago. Each scheme carries the drivers that speak it in preference order, and the first one that imports is used.

| Scheme | Drivers tried, in order |
| --- | --- |
| `postgresql` and the PostgreSQL-wire schemes | `psycopg`, `psycopg2`, `pg8000` |
| `mysql` and the MySQL-wire schemes | `pymysql`, `MySQLdb`, `mysql.connector` |
| `oracle` | `oracledb`, `cx_Oracle` |
| `sqlite` | `sqlite3` |
| `duckdb` | `duckdb` |
| `clickhouse` | `clickhouse_driver` |

For anything else, name the driver yourself with `module=` and `connect_kwargs=`. That is also the answer for `pyodbc`, whose DSN names a driver rather than a dialect, so nothing about the dialect can be inferred from it. Pass `dialect=` alongside it so identifier quoting and the upsert spelling are still right.

## How an upsert is spelled

`INSERT`, `UPDATE` and `DELETE` are ANSI SQL. Upsert is not, and three incompatible spellings cover essentially every operational database in use.

| Spelling | Dialects |
| --- | --- |
| `INSERT ... ON CONFLICT (key) DO UPDATE SET ...` | PostgreSQL, SQLite, DuckDB, CockroachDB, TimescaleDB, AlloyDB, YugabyteDB |
| `INSERT ... ON DUPLICATE KEY UPDATE ...` | MySQL, MariaDB, TiDB, SingleStore, Percona |
| `MERGE INTO ... WHEN MATCHED ... WHEN NOT MATCHED ...` | SQL Server, Oracle, Snowflake, BigQuery, Redshift |

A dialect outside those three is refused rather than guessed at, and the error names `mode="delete_insert"`: the same intent built from ANSI SQL alone, run inside one transaction.

One semantic difference is worth knowing because it is invisible in the SQL. `ON DUPLICATE KEY` has no conflict target: MySQL matches on **any** unique index, not on the columns you named. A table with a second unique index will therefore update rows a PostgreSQL `ON CONFLICT (id)` would have inserted. That is MySQL's semantics, not a translation defect.

## Transactions

One `write` call is one transaction. Every chunk of every statement runs, then a single commit. A write that fails rolls back whole, so a partial batch is never visible.

`overwrite` empties the table with `DELETE FROM` rather than `TRUNCATE`, deliberately. Truncation is DDL on several engines and commits the surrounding transaction implicitly, which would publish the empty table before the new rows were written. A crash between the two would have destroyed the table's contents.

A **distributed** write is one transaction per shard, not one across the cluster. That is safe for `append`, `upsert`, `update` and `delete`, because a shard only ever touches the keys its own rows name. `overwrite` is refused past the first shard: every shard would empty the one table they all target, so each would discard the shards before it. It is invisible single-node and appears at cluster scale as missing rows rather than an error.

Where you need cluster-wide atomicity, write to a staging table and swap, or use a {doc}`lakehouse table </user-guide/moving-data/lakehouse>`, whose commit is atomic by construction.

## Streaming into a table

A `trigger=` turns the same call into a streaming query: each micro-batch is one write, in
one transaction, against the same table.

```python
# docs: skip
import batcher as bt

query = bt.read.kafka("orders").write(
    "orders",
    "dbapi",
    uri="postgresql://db/shop",
    mode="upsert",
    key_columns="order_id",
    trigger=bt.Trigger.processing_time("30 seconds"),
    checkpoint="s3://ckpt/orders",
)
```

Use `mode="upsert"`, not `mode="append"`, and the reason is exactly-once. The engine
records a micro-batch's source offset before processing it, so a crash between processing
and committing leaves a batch the next run replays. An append writes those rows a second
time. An upsert writes the same keys to the same values, which makes the replay a no-op —
the same end-to-end guarantee a Delta stream gets from its `(app_id, batch_id)`
transaction, reached by a different route. Batcher does not warn about at-least-once
delivery for a keyed mode, because for a keyed mode it does not apply.

This is the shape Spark spells as `foreachBatch` plus a hand-written `MERGE`.
{py:meth}`ds.write.for_each_batch <batcher.api.io_namespace.writer.Writer.for_each_batch>`
is still there for a write that needs more than one statement per batch.

## Retries

A deadlock, a serialization failure, a lock-wait timeout or a dropped connection is the server saying "run this again", and each is retried with jittered exponential backoff. `retries=` sets how many extra attempts, and `retries=0` turns it off. Anything unrecognized is raised immediately, so a syntax error or a constraint violation fails fast instead of spending the retry budget.

A retry re-runs a transaction the server already rolled back, so it starts from the state the first attempt did. The one case no retry can decide is a connection lost *after* the server committed: an upsert absorbs the repeat, an append duplicates it. Prefer a keyed mode wherever keys exist.

## Bring your own connection

Pass `connection=` to write through a PEP 249 connection you already hold, the way `pandas.to_sql(name, con)` takes one. Batcher never closes it, and by default never commits it either, so the write can be one statement inside a larger unit of work you commit yourself.

```python
import sqlite3
import tempfile
from pathlib import Path

import batcher as bt

db = Path(tempfile.mkdtemp()) / "shop.db"
conn = sqlite3.connect(db)
# The table must already exist on this path -- see the note below the example.
conn.execute('CREATE TABLE orders ("id" INTEGER PRIMARY KEY, "amount" REAL)')

bt.from_pydict({"id": [1], "amount": [10.0]}).write(
    "orders", "dbapi", connection=conn, dialect="sqlite", mode="append", key_columns=("id",)
)
bt.from_pydict({"id": [2], "amount": [20.0]}).write(
    "orders", "dbapi", connection=conn, dialect="sqlite", mode="append", key_columns=("id",)
)
conn.commit()  # both writes become visible together

print(conn.execute("SELECT count(*) FROM orders").fetchone()[0])
conn.close()
```

```text
2
```

Pass `commit_writes=True` to have Batcher commit a borrowed connection instead. A connection cannot be shipped to a worker, so `connection=` is single-node; `uri=` is what scales out.

**The table must already exist** on this path, which is why the example creates it. Batcher creates a missing table by asking whether it exists first, and on PostgreSQL a statement that fails aborts the entire transaction — so asking about a table that is not there would destroy work the caller did before handing the connection over. Batcher will not do that to a connection it does not own. Use `uri=` if you want the table created for you.

## What this costs

The DB-API path materializes every value as a Python object, because a cursor is row-shaped and there is no way to hand a driver a column. The conversion is column-wise and once per chunk rather than once per row, but it is real, and it is why a plain append to a database ADBC covers still goes through ADBC.

`rows_per_statement` is the chunk size and defaults to 1,000. It is not only a throughput knob: several wire protocols cap the parameters one statement may carry, and PostgreSQL's limit of 65,535 means a ten-column insert overflows at 6,554 rows. Raise it for a narrow table, lower it if a driver complains about statement size.

A driver accepts the Python types it knows, and Arrow produces some that it does not. `decimal.Decimal` is the usual one: SQLite has no decimal type and its driver refuses to adapt one. Batcher will not choose a lossy encoding on your behalf, so the write raises naming the columns being bound. Cast the column first, with `ds.cast({"amount": "float64"})`, or register an adapter with your driver.

## Requirements and limitations

Nested Arrow types (list, struct, map) have no portable SQL column type, so `create_table` refuses them rather than picking an encoding for you. Create the table yourself with the encoding you want and write into it with `mode="append"`, or flatten the column first.

A row whose key column is null matches no row on any database, because SQL equality against null is unknown rather than true. Such rows are counted and reported at warning level rather than silently doing nothing.

`update` and `delete` do not report which keys matched nothing. The server's affected-row count is recorded on the manifest as `stats["affected_rows"]` where the driver reports one, which is the closest available signal.

There is no cross-shard transaction, and no cross-*call* transaction. Two `write` calls are two transactions unless you pass your own `connection=` and commit it yourself.

Each write opens and closes its own connection. For a batch job that is one connect; for a streaming query it is one per micro-batch, so a one-second trigger dials the database once a second and a distributed stream does so once per shard. Check the server's connection limit before running a short trigger interval across many workers, or hold the connection yourself with `connection=` on a single-node stream.

The MySQL type map uses `TEXT` for string columns, and MySQL cannot index a `TEXT` column without a prefix length. A table with a string primary key therefore has to be created by hand there. An integer or date key is unaffected.

## The operational stores use the same vocabulary

MongoDB, DynamoDB, Cassandra, Redis and Elasticsearch are maintained the same way a SQL table is, so they take the same `mode` words. What differs is which of them each store can actually express, and a store declines the rest by name rather than approximating it.

| Sink | Modes | Bulk primitive |
| --- | --- | --- |
| {py:meth}`ds.write.mongo <batcher.api.io_namespace.writer.Writer.mongo>` | `upsert`, `append`, `overwrite`, `delete` | one `bulk_write` per batch |
| {py:meth}`ds.write.elasticsearch <batcher.api.io_namespace.writer.Writer.elasticsearch>` | `upsert`, `append`, `overwrite`, `delete` | `_bulk`, 1,000 documents per request |
| {py:meth}`ds.write.dynamodb <batcher.api.io_namespace.writer.Writer.dynamodb>` | `upsert`, `delete` | `BatchWriteItem`, 25 requests per call |
| {py:meth}`ds.write.cassandra <batcher.api.io_namespace.writer.Writer.cassandra>` | `upsert`, `delete` | one prepared statement, run concurrently |
| {py:meth}`ds.write.redis <batcher.api.io_namespace.writer.Writer.redis>` | `upsert`, `delete` | one pipeline per batch |

The refusals are about the stores, not about unfinished work. DynamoDB has no `append`, because `PutItem` replaces the item holding the same key and no batch operation inserts only when the key is absent; Cassandra has none because a CQL `INSERT` is an upsert. Redis has no `overwrite`, because emptying a keyspace means `FLUSHDB`, which discards keys the write knows nothing about; DynamoDB and Cassandra have none because emptying those means a full scan-and-delete or a cluster-wide `TRUNCATE`. Reaching an operation of that reach by passing a string to `mode` is not something a write API should offer.

All five default to `upsert` rather than to `ds.write`'s usual `overwrite`, for the same reason: these stores are maintained rather than replaced, and a destructive default would empty one on a call that never said so.

Each of these APIs reports partial failure inside a success, and each sink reads the response rather than the status code. `BatchWriteItem` returns the requests it did not apply under `UnprocessedItems` with a 200 — those are retried with backoff, and a remainder that survives raises. `_bulk` reports per-document failures inside an HTTP 200. Cassandra's concurrent execution returns a success flag per statement. A sink that trusted the call would have written some of its rows and reported success, which is the quietest kind of data loss there is.

```python
# docs: skip
scores.write.dynamodb("user_scores", region_name="us-east-1")
sessions.write.redis("session", host="cache.internal", ttl_seconds=3600)
features.write.cassandra("features", contact_points=["c1"], keyspace="serving")
```

## See also

- {doc}`SQL databases </integrations/databases/databases>`: the read path, connection URIs, pushdown, and parallel extraction.
- {doc}`Writing data </user-guide/moving-data/writing-data>`: save modes, partitioning, and the manifest, across every sink.
- {doc}`Lakehouse tables </user-guide/moving-data/lakehouse>`: `MERGE INTO`, CDC and time travel against Delta and Iceberg, where the commit is atomic across a cluster.
- {doc}`MongoDB </integrations/databases/mongodb>` and {doc}`Elasticsearch </integrations/databases/elasticsearch>`: the two document stores, with their own read paths and failure modes.
- {doc}`I/O API </api/relational/io>`: the reference.
