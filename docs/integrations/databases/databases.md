# SQL databases

This page covers reading and writing any SQL database from a standard connection URI. One URI vocabulary covers PostgreSQL, MySQL, SQL Server, Oracle, SQLite, DuckDB, Trino, and the rest, and Batcher picks the backend that can serve it. For a database with no Arrow-native driver at all, the DB-API section at the end takes any PEP 249 driver instead.

| | |
| --- | --- |
| **Read** | `bt.read.sql(query, uri=...)`, `bt.read.sql(query, connection=...)`, or `bt.read.table("dbapi", module=..., ...)` |
| **Write** | `ds.write.sql(table, uri=...)`, ADBC schemes only |
| **Extra** | `pip install 'batcher-engine[sql]'` or `[connectorx]`, plus the per-database driver |
| **Parallelism** | FlightSQL server-side partitions, or `partition_on=` range partitions on every other backend |
| **Pushdown** | Projection and predicate, both folded into the submitted SQL |
| **Credentials** | `password="env:VAR"` or `"file:/path"`, resolved on the worker |

The warehouse-specific pages cover the connectors that are not URI-routed: {doc}`Snowflake </integrations/warehouses/snowflake>`, {doc}`BigQuery </integrations/warehouses/bigquery>`, and {doc}`Databricks </integrations/warehouses/databricks>`.

## Which backend serves which scheme

The scheme vocabulary is SQLAlchemy's. The same string that works in `sqlalchemy.create_engine`, pandas' `read_sql`, Polars' `read_database_uri`, or an existing `$DATABASE_URL` works here unchanged. You do not choose the backend. The scheme does.

| Scheme | Backend | Driver loaded |
| --- | --- | --- |
| `postgresql`, `postgres` | ADBC | `adbc_driver_postgresql` |
| `sqlite` | ADBC | `adbc_driver_sqlite` |
| `duckdb` | ADBC | `adbc_driver_duckdb` |
| `snowflake` | ADBC | `adbc_driver_snowflake` |
| `bigquery` | ADBC | `adbc_driver_bigquery` |
| `flightsql`, `grpc`, `grpc+tcp`, `grpc+tls` | ADBC | `adbc_driver_flightsql` |
| `mysql`, `mariadb` | ConnectorX | ConnectorX's own reader |
| `mssql`, `sqlserver` | ConnectorX | ConnectorX's own reader |
| `oracle`, `redshift`, `trino`, `clickhouse` | ConnectorX | ConnectorX's own reader |

ADBC schemes are the ones with a first-class Arrow driver: the server hands back Arrow and nothing is materialized as a Python object on the way. ConnectorX is Arrow-native end to end too. It owns its own connection vocabulary, which is the same URI Batcher already parsed.

A SQLAlchemy `+driver` suffix is accepted and ignored, so `postgresql+psycopg2://` resolves exactly like `postgresql://`. Which DBAPI driver SQLAlchemy would have used is not meaningful here, because the read goes through ADBC or ConnectorX in Arrow. The `grpc+tcp` and `grpc+tls` schemes are the exception: their suffix names a transport rather than a driver, so it is kept.

`sqlite` and `duckdb` address a local file, so their path is a locator and not a database name. Write them with three slashes, `sqlite:///local.db`.

```python
# docs: skip
import batcher as bt
from batcher import col

orders = bt.read.sql(
    "SELECT order_id, customer_id, amount, country FROM orders",
    uri="postgresql://svc@warehouse:5432/shop",
    password="env:PGPASSWORD",
)
big = orders.filter(col("amount") > 100).select("order_id", "country").collect()
```

To check what a URI resolves to before you connect, ask the parser:

```python
from batcher.io.formats.sql.uri import parse_uri, known_schemes

parsed = parse_uri("postgresql+psycopg2://alice@db:5432/app")
print(parsed.backend, parsed.driver, parsed.database)
print(parse_uri("mysql://svc@db/shop").backend)
print(len(known_schemes()))
```

```text
adbc adbc_driver_postgresql app
connectorx
36
```

### Wire-compatible databases

Many databases are not PostgreSQL or MySQL but speak their wire protocol. The wire protocol is the only thing a driver needs to connect, execute, and return Arrow, so each of these routes to the same driver its base protocol uses. A PostgreSQL-wire database goes to `adbc_driver_postgresql`. A MySQL-wire database goes to ConnectorX's MySQL reader. You install the base driver, not a per-database one.

| Scheme | Wire protocol | Backend | Driver loaded |
| --- | --- | --- | --- |
| `cockroachdb`, `cockroach` | PostgreSQL | ADBC | `adbc_driver_postgresql` |
| `timescaledb` | PostgreSQL | ADBC | `adbc_driver_postgresql` |
| `alloydb` | PostgreSQL | ADBC | `adbc_driver_postgresql` |
| `greenplum` | PostgreSQL | ADBC | `adbc_driver_postgresql` |
| `yugabytedb`, `yugabyte` | PostgreSQL | ADBC | `adbc_driver_postgresql` |
| `risingwave` | PostgreSQL | ADBC | `adbc_driver_postgresql` |
| `materialize` | PostgreSQL | ADBC | `adbc_driver_postgresql` |
| `questdb` | PostgreSQL | ADBC | `adbc_driver_postgresql` |
| `crate`, `cratedb` | PostgreSQL | ADBC | `adbc_driver_postgresql` |
| `singlestore`, `memsql` | MySQL | ConnectorX | ConnectorX's MySQL reader |
| `tidb` | MySQL | ConnectorX | ConnectorX's MySQL reader |
| `starrocks` | MySQL | ConnectorX | ConnectorX's MySQL reader |
| `doris` | MySQL | ConnectorX | ConnectorX's MySQL reader |
| `percona` | MySQL | ConnectorX | ConnectorX's MySQL reader |

```python
# docs: skip
events = bt.read.sql(
    "SELECT event_id, ts, kind FROM events",
    uri="cockroachdb://svc@crdb:26257/analytics",
    password="env:CRDB_PASSWORD",
)
```

:::{warning}
Wire compatibility is not SQL compatibility. The connection works, the query runs, and Arrow comes back, but a query that calls a function the target engine does not implement fails on the server with the server's own error. Routing a `cockroachdb://` URI to the PostgreSQL driver does not make CockroachDB understand every PostgreSQL function. Test against the real database, not against PostgreSQL or MySQL.
:::

Presto is deliberately not routed. Presto and Trino diverged after the fork, and ConnectorX ships a Trino reader, so sending Presto to it would be a guess dressed up as support. A `presto://` URI raises a `BackendError` pointing at ODBC, `bt.read.table("odbc", connection_string=...)`.

## Credentials

A connection URI reaches log lines, error messages, and split identities, so a password embedded in its userinfo leaks everywhere the URI is merely mentioned. On the ADBC path Batcher lifts an inline password out of the URI and carries it in a field excluded from every `repr`. `redact_uri` gives you the same treatment for anything you log yourself.

```python
from batcher.io.formats.sql.uri import redact_uri

print(redact_uri("postgresql://alice:hunter2@db:5432/app"))
```

```text
postgresql://alice:***@db:5432/app
```

That separation is not encryption. The password is still carried, and still pickled onto the split that ships to every worker. To keep the secret out of the process image, pass a reference instead of a literal:

- `password="env:PGPASSWORD"` reads the environment variable.
- `password="file:/etc/secrets/pg"` reads the file and strips trailing whitespace.

The reference is what gets pickled. It becomes a secret only on the worker, at connect time, inside the process that opens the connection. A missing variable or an unreadable file raises a `BackendError` that names the reference and never the secret.

:::{warning}
ConnectorX has no separate password channel. It takes credentials inside its URI, so `bt.read.sql(..., uri="mysql://...", password=...)` raises a `BackendError` rather than silently dropping the credential. Embed it in the URI, `mysql://user:pw@host/db`. To keep it out of the driver process entirely, build the source directly and pass the whole URI as a reference, which ConnectorX resolves on the worker:

```python
# docs: skip
ds = bt.read.table("connectorx", query="SELECT * FROM orders", conn_uri="env:MYSQL_URL")
```

`bt.read.sql` parses its `uri=` before routing, so a reference passed there is rejected as a malformed URI. `bt.read.table("connectorx", ...)` is the spelling that works.
:::

## What reaches the server

Kyber pushes both the projection and the predicate into the SQL that is actually submitted, and it does so at split-planning time rather than read time. That timing is the point: a split is a picklable locator a worker rebuilds a reader from, so a pushdown living anywhere but inside the split's own query would leave the worker reconstructing an unfiltered read that the server never hears about.

| What you write | Where it runs |
| --- | --- |
| `.filter(...)` after the read | The database, as a `WHERE` below the projection |
| `.select(...)` after the read | The database, as the submitted `SELECT` column list |
| A predicate the translator cannot express | Your process, in the engine's `Filter` |

The last row is a slowdown and never a wrong answer. An unpushed predicate is re-checked by the engine regardless, so the result is identical either way.

The predicate is applied below the projection on purpose. Kyber pushes the two independently and routinely pushes a projection that omits the column the predicate filters on, so projecting first would produce SQL referencing a column that no longer exists.

### The schema probe

`schema()` does not run your query. Batcher submits `SELECT * FROM (<your query>) AS _bc WHERE 1 = 0`, which every backend here folds to an empty scan before touching storage. The result set is empty and fully typed, which is exactly what the planner needs. On a warehouse that bills per query or per byte scanned, this is the difference between one invoice and two.

If a driver types an empty result set from row data rather than query metadata, the probe comes back with null-typed columns. Batcher checks for that and falls back to the full read, which is slower rather than wrong.

## How the read parallelizes

Parallelism depends on the backend, and none of the three shapes is configured the same way.

FlightSQL drivers partition server-side. Pass `partition=True` and Batcher makes a single `adbc_execute_partitions` submission, then builds one split per opaque descriptor returned. Each split rebuilds a fresh connection on its worker and reads its own slice. This is the only backend with true shippable distributed partitions. A driver that does not implement partitioning falls back to a single streaming split rather than failing.

ConnectorX owns range partitioning itself. Give it an integer column and a partition count and it issues that many balanced sub-queries, computing the bounds internally so Batcher never needs a separate probe. That parallelism is internal to one split, so it fans out across cores rather than across workers.

Everything else range-partitions on a column you name. Without `partition_on=` an ADBC read is a single split that streams the whole query once with `fetch_record_batch`, which keeps memory bounded but reads on one worker.

## Parallel extraction

A single SQL query is a single stream. One connection, one cursor, one core, however large the table. That is the difference between a warehouse extract that finishes in minutes and one that finishes in hours, and it is the reason every bulk-extract tool has some form of this feature.

Name an indexed numeric column and its approximate range, and the read becomes that many independent queries over disjoint slices of the key:

```python
# docs: skip
events = bt.read.sql(
    "SELECT * FROM events",
    uri="postgresql://svc@warehouse:5432/analytics",
    password="env:PGPASSWORD",
    partition_on="event_id",
    lower_bound=1,
    upper_bound=1_000_000,
    num_partitions=8,
)
```

The same four keywords work on `bt.read.table("dbapi", ...)`, so a PEP 249 driver fans out the same way an Arrow-native one does.

### The bounds are cut points, not filters

:::{important}
`lower_bound` and `upper_bound` say where to *cut*, not what to *keep*. The first partition is unbounded below and the last is unbounded above, so a row outside the stated range is still read. It lands in an edge partition instead. Getting the bounds wrong costs skew, never rows.
:::

That is Spark's JDBC behavior too, and it is still the thing readers get wrong. If bounds filtered, a stale `upper_bound` would silently drop every row inserted since you wrote it, and the extract would report success. NULL keys are placed in the first partition explicitly, for the same reason: `col < x` and `col >= x` are both unknown for NULL, so without that placement every NULL-keyed row would match no partition and vanish.

The two properties together give the invariant the implementation is built around. The partitions are disjoint and exhaustive, so concatenating them reproduces the unpartitioned read exactly, for any bounds, including wrong ones. The test suite asserts it directly at 1, 2, 3, 4, 8, and 16 partitions, with NULL and out-of-range keys present, and again after each split is pickled to a worker.

You can see the fragments the reader builds:

```python
from batcher.io.formats.sql.partition import range_predicates

for fragment in range_predicates("id", 1, 1_000_000, 4):
    print(fragment)
```

```text
id < 250000.75 OR id IS NULL
id >= 250000.75 AND id < 500000.5
id >= 500000.5 AND id < 750000.25
id >= 750000.25
```

Two inputs collapse to a single unpartitioned read rather than producing empty queries: `num_partitions=1`, and equal bounds, where every cut point would coincide and N-1 partitions would be empty.

```python
print(range_predicates("id", 1, 1000, 1))
print(range_predicates("id", 5, 5, 8))
```

```text
[None]
[None]
```

### Coming from Spark

The spelling is deliberately Spark's JDBC reader's, and the mapping is near 1:1.

| Spark JDBC option | Batcher keyword | Difference |
| --- | --- | --- |
| `partitionColumn` | `partition_on` | None. Numeric, and should be indexed. |
| `lowerBound` | `lower_bound` | None. A cut point in both. |
| `upperBound` | `upper_bound` | None. A cut point in both. |
| `numPartitions` | `num_partitions` | Bounds the reader's queries, not a cluster-wide connection cap. |
| `url` | `uri` | SQLAlchemy scheme vocabulary rather than a JDBC URL. |
| `dbtable` | `table=` | ADBC path only. A ConnectorX read takes a query. |
| `query` | the first positional argument | Passed to `bt.read.sql(query, ...)`. |
| `fetchsize` | `batch_size` | DB-API path only. Arrow backends stream natively. |

`partition_on` requires both bounds on the ADBC and DB-API paths. Naming the column without them raises a `BackendError` rather than probing for them, exactly as Spark's JDBC reader requires all four options together. There is no auto-probe.

ConnectorX is the exception to that rule. It takes `partition_on` and `num_partitions` but no bounds, because it derives the ranges itself as part of the partitioned read.

### What this costs

The partition column must be numeric, and it should be indexed and reasonably uniform. A partitioned read on an unindexed column makes the database perform N full scans instead of one, which is slower than not partitioning at all. Check the plan on the server before raising `num_partitions`.

Each partition is a separate query on a separate connection, so `num_partitions=32` means 32 concurrent connections and 32 concurrent queries against one server. That is load you are moving from your process onto the database, and the server's connection and concurrency limits are the real ceiling rather than your core count.

Skew is the cost of wrong bounds. Bounds much narrower than the data leave the edge partitions carrying most of the rows, and the extract runs at the speed of its slowest query. Bounds much wider leave the interior partitions empty. Neither loses a row. A cheap `SELECT min(id), max(id)` before the extract is usually all the accuracy needed.

The splits are what the distributed executor schedules across workers, so the fan-out is realized on a `collect(distributed=...)` run. On a single node the partitioned queries still run, and ConnectorX is the backend that parallelizes within one process regardless.

ADBC prefers server-side partitioning when the driver has it. With `partition=True` Batcher tries `adbc_execute_partitions` first, which splits one already-executed result set and is strictly better than N independent queries. Range partitioning is the fallback, reached only once the driver has declined.

## Writing

`ds.write.sql(table, uri=...)` bulk-ingests Arrow into a destination table through ADBC. It takes the same URI and the same `password=` reference as the read, so a read and the write that follows it are spelled identically.

```python
# docs: skip
manifest = orders.write.sql(
    "orders_enriched",
    uri="postgresql://svc@warehouse:5432/shop",
    password="env:PGPASSWORD",
    mode="append",
)
```

`mode` is passed to `adbc_ingest` and takes `"create"`, `"append"`, `"replace"`, or `"create_append"`, which is the default.

There is no cross-shard transaction. Each shard of a distributed write ingests and commits its own rows as it finishes, and the driver's commit step is a no-op, so a write that dies halfway leaves the rows that already landed. Write to a staging table and swap, or key the data so a re-run is idempotent.

Writing is ADBC only. A ConnectorX scheme has no sink, because ConnectorX is a reader.

## Any driver at all, through DB-API

PEP 249 is the one interface essentially every Python database driver implements, including `psycopg`, `pymysql`, `cx_Oracle`, `pyodbc`, `sqlite3`, `ibm_db_dbi`, `pyhive`, and `teradatasql`. The `dbapi` source turns any of them into a Batcher relation, so "my database has a Python driver" is enough to read it.

Reach for it when no Arrow-native driver exists. Otherwise `bt.read.sql(query, uri=...)` is the right answer.

```python
import os
import sqlite3
import tempfile

import batcher as bt
from batcher import col

db = os.path.join(tempfile.mkdtemp(), "shop.db")
conn = sqlite3.connect(db)
conn.execute("CREATE TABLE orders (id INTEGER, amount REAL, country TEXT)")
conn.executemany(
    "INSERT INTO orders VALUES (?, ?, ?)",
    [(1, 10.0, "US"), (2, 250.0, "US"), (3, 7.5, "DE")],
)
conn.commit()
conn.close()

orders = bt.read.table(
    "dbapi",
    module="sqlite3",
    connect_kwargs={"database": db},
    table="orders",
)
print(orders.filter(col("amount") > 9).select("id", "country").to_pydict())
```

```text
{'id': [1, 2], 'country': ['US', 'US']}
```

`module` is the importable driver name and must expose a module-level `connect()`. `connect_kwargs` is passed straight to it, and any string value there may be an `env:` or `file:` reference, resolved on the worker exactly as `password=` is elsewhere. Pass either `query=` or `table=`.

### What this costs

Rows are converted at batch granularity and never one at a time. A DB-API cursor is row-shaped, so the boundary is drawn at `fetchmany(batch_size)`: the driver returns a block of rows, that block is transposed column-wise, and it reaches Arrow in one call. Everything after that is columnar. `batch_size` defaults to 16,384, matching the engine's morsel size so a batch crossing FFI needs no rechunking, and it is the parameter that matters most for throughput here.

That still means paying Python-object materialization for every value, which ADBC does not. Expect this path to be several times slower than the same read over a URI.

It reads on one worker unless you partition it. A live connection cannot be pickled and PEP 249 defines no way to partition a result set, so the only parallelism available is issuing several independent queries. That is what `partition_on=` does here, and each split then opens its own connection on its own worker. The Parallel extraction section above covers the keywords and their cost.

Pushdown still works, and it matters more here than anywhere else on this page. The pushed `WHERE` and column list are folded into the SQL the split carries, so the server filters before any row becomes a Python object at all.

Types are the remaining sharp edge. PEP 249 exposes only four type singletons, `STRING`, `BINARY`, `NUMBER`, and `DATETIME`, and `NUMBER` covers integers, floats, and decimals alike. When the driver's codes do not resolve to one Arrow type, Batcher infers from the first real batch rather than guessing, which means `schema()` executes the query and reads one batch instead of costing nothing. The first batch then fixes the schema for every batch after it, so a column that is all-null in batch two cannot retype the relation. Pass `schema_override=` with an explicit Arrow schema when the types matter and the driver will not say.

If a table has columns but no rows and the driver reports no types, the schema comes back with null-typed columns. That says exactly what is known. Dropping the columns would hand the planner an empty relation for a table with a real shape, and guessing `string` would be a lie the first non-empty read exposes.

## Bring your own connection

You can hand `bt.read.sql` a connection you already have, exactly as `pandas.read_sql(query, con)` does. Pass a live PEP 249 connection, or a SQLAlchemy `Engine` or `Connection`, as `connection=`. A SQLAlchemy handle is unwrapped to the DBAPI connection underneath it, and the unwrapping is duck-typed, so SQLAlchemy is never a required import.

```python
import sqlite3
import tempfile
from pathlib import Path

import batcher as bt
from batcher import col

db = str(Path(tempfile.mkdtemp()) / "shop.db")
conn = sqlite3.connect(db)
conn.execute("CREATE TABLE orders (id INTEGER, amount REAL, country TEXT)")
conn.executemany(
    "INSERT INTO orders VALUES (?, ?, ?)",
    [(1, 10.0, "US"), (2, 250.0, "US"), (3, 7.5, "DE")],
)
conn.commit()

orders = bt.read.sql("SELECT id, amount, country FROM orders", connection=conn)
print(orders.filter(col("amount") > 9).select("id", "country").to_pydict())
```

```text
{'id': [1, 2], 'country': ['US', 'US']}
```

A borrowed connection reads through the DB-API path, so its costs and its type handling are the ones in the DB-API section above. Three properties are specific to passing a live connection, and each is a deliberate choice rather than a limitation to work around:

- It is single-node only. A live connection belongs to the process that opened it and cannot be pickled to a worker, so the read stays on one process. Combining `connection=` with `partition_on=` raises a `BackendError` rather than partitioning, because range partitioning runs one query per worker and there is no connection to give the other workers. Use `uri=` when you need to scale the read out.
- Batcher never closes it. The caller owns a borrowed connection and keeps using it after the read, so Batcher closes only connections it opened itself. The connection in the example above is still open and usable when `collect` returns.
- `uri=` and `connection=` together are refused. They name two databases with no way to tell which you meant, so passing both raises a `BackendError`.

One more property is worth knowing before you rely on learned statistics. A borrowed connection is a live object with no stable identity across runs, so its split identity falls back to the driver name and cannot tell two databases reached through the same driver apart. Pass `bt.read.table("dbapi", module=..., connect_kwargs=...)` when you want a precise, reusable key for the optimizer.

### Coming from pandas

The `connection=` support is there so a `pandas.read_sql` line ports almost verbatim. pandas takes the query first and the connection second; Batcher takes the query first and the connection as a keyword.

| pandas | Batcher |
| --- | --- |
| `pandas.read_sql(query, conn)` | `bt.read.sql(query, connection=conn)` |
| `pandas.read_sql(query, engine)` | `bt.read.sql(query, connection=engine)` |
| `pandas.read_sql(query, sqlalchemy_conn)` | `bt.read.sql(query, connection=sqlalchemy_conn)` |

The result is a lazy `Dataset` rather than an eager DataFrame, so nothing runs until a terminal op such as `collect` or `to_pydict`. When you have a URI rather than a connection object, `bt.read.sql(query, uri=...)` is the better port, because it scales across workers and keeps the password out of the process image.

## Requirements and limitations

The `sql` extra installs `adbc-driver-manager` and `adbc-driver-flightsql` only. Per-database ADBC drivers are separate packages, so PostgreSQL also needs `pip install adbc-driver-postgresql`, SQLite needs `adbc-driver-sqlite`, and so on. ConnectorX schemes need `pip install 'batcher-engine[connectorx]'`.

`table=` works on the ADBC path. A ConnectorX read takes a query.

`row_count()` returns `None` on every backend here, so the optimizer has no row estimate for a SQL source until it reads one.

Every split opens its own connection. A partitioned FlightSQL read with a hundred descriptors means a hundred connections, so check the server's concurrency limits before fanning out.

Credentials live on the split. They are never logged, but they are serialized to every worker, which is worth knowing on a shared cluster before reaching for a personal token instead of a service account.

A scheme Batcher cannot route raises a `BackendError` listing the ones it can. For a driver with no URI scheme at all, construct the source directly, such as `bt.read.table("odbc", connection_string=...)`.

## See also

- {doc}`Reading data </user-guide/moving-data/reading-data>` and {doc}`Writing data </user-guide/moving-data/writing-data>`: the reader and writer surface, splits, and pushdown.
- {doc}`Snowflake </integrations/warehouses/snowflake>`, {doc}`BigQuery </integrations/warehouses/bigquery>`, and {doc}`Databricks </integrations/warehouses/databricks>`: the warehouse connectors with their own read paths.
- {doc}`MongoDB </integrations/databases/mongodb>` and {doc}`Elasticsearch </integrations/databases/elasticsearch>`: the non-relational stores.
- {doc}`Custom connectors </user-guide/moving-data/custom-connectors>`: the protocol, for a system not listed.
- {doc}`I/O API </api/relational/io>`: the reference.
