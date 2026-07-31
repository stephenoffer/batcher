---
name: migrate-from-a-sql-warehouse
description: Port a SQL warehouse or JDBC extract onto Batcher — the connection-URI vocabulary, the near-1:1 Spark JDBC option mapping, pandas read_sql/SQLAlchemy and plain DB-API cursor loops, range partitioning for parallel extracts, and a recipe that ends by proving the ported extract returns the same rows. Invoke when moving a `spark.read.jdbc`, `pandas.read_sql`, SQLAlchemy, or DB-API extract onto Batcher, or when asked how to connect Batcher to Postgres/MySQL/Snowflake/Redshift/Oracle/Trino.
---

# Migrate a SQL warehouse extract to Batcher

Use this when the *source* is a database rather than a file. The query text usually ports
unchanged — it runs on the warehouse, not in Batcher — so a port is almost entirely about
the **connection** and the **parallelism**. Both are deliberately spelled the way you
already spell them: a SQLAlchemy-style URI, and Spark's JDBC partitioning options.

This skill covers the extract boundary. Once you have a `Dataset`, the transforms are
ordinary Batcher and `write-a-batcher-pipeline` takes over. If you are porting the *SQL
itself* to run inside Batcher, that is `migrate-from-duckdb-sql` instead.

## One URI, routed by scheme

`bt.read.sql(query, uri=...)` is the entry point. The scheme picks the backend, and you
don't choose it:

```python
import batcher as bt

ds = bt.read.sql(  # docs: skip
    "SELECT id, region, amount FROM orders WHERE status = 'paid'",
    uri="postgresql://svc@warehouse:5432/shop",
    password="env:PGPASSWORD",
)
```

Schemes with an Arrow-native ADBC driver route to **ADBC**: `postgresql`/`postgres`,
`sqlite`, `duckdb`, `snowflake`, `bigquery`, and the FlightSQL family
(`flightsql`, `grpc`, `grpc+tcp`, `grpc+tls`). The rest route to **ConnectorX**, which is
also Arrow-native end to end: `mysql`, `mariadb`, `mssql`, `sqlserver`, `oracle`,
`redshift`, `trino`, `clickhouse`. That is 18 schemes;
`batcher.io.formats.sql.uri.known_schemes()` is the live list.

A SQLAlchemy `+driver` suffix is accepted and ignored, so `postgresql+psycopg2://…` and
`mysql+pymysql://…` resolve exactly like the bare scheme. An existing `$DATABASE_URL`, a
dbt profile string, or the URI already in your `create_engine` call works unchanged.

For a database with no scheme at all, construct the source directly:
`bt.read.table("odbc", connection_string=...)` or `bt.read.table("dbapi", module=...)`.

## Spark JDBC option mapping

The partitioning options are modeled on Spark's JDBC reader, so this table is nearly 1:1.

| `spark.read.jdbc` / `.option(...)` | Batcher | Note |
|---|---|---|
| `.format("jdbc")` | *(nothing)* | `bt.read.sql(...)` is the reader |
| `url="jdbc:postgresql://host:5432/db"` | `uri="postgresql://host:5432/db"` | drop the `jdbc:` prefix |
| `driver="org.postgresql.Driver"` | *(nothing)* | the scheme selects the backend; no JDBC jar |
| `dbtable="orders"` | `table="orders"` | reads the table in full |
| `dbtable="(SELECT …) t"` | `query="SELECT …"` | no subquery-alias wrapping needed |
| `query="SELECT …"` | `query="SELECT …"` | the first positional argument |
| `user="svc"` | in the URI userinfo | `postgresql://svc@host/db` |
| `password="…"` | `password="env:PGPASSWORD"` | separate channel on ADBC schemes only |
| `partitionColumn="id"` | `partition_on="id"` | numeric, and should be indexed |
| `lowerBound=1` | `lower_bound=1` | a **cut point**, not a filter |
| `upperBound=1000000` | `upper_bound=1_000_000` | a **cut point**, not a filter |
| `numPartitions=8` | `num_partitions=8` | how many parallel queries to issue |
| `fetchsize=10000` | `batch_size=…` | DB-API path only; ADBC/ConnectorX stream Arrow |
| `pushDownPredicate=true` | *(always on)* | Kyber pushes filter and projection into the SQL |
| `.write.jdbc(url, table, mode)` | `ds.write.sql(table, uri=..., mode=...)` | `"create"`, `"append"`, `"replace"`, `"create_append"` |

## Partitioning: bounds cut, they do not filter

This is the single most important behavior to understand, and it differs from the mental
model most people carry over from Spark even though the spelling matches.

`lower_bound` and `upper_bound` describe **where to cut**, not what to keep. The first
partition is unbounded below, the last is unbounded above, and `NULL` keys ride
explicitly in the first partition. So the partitions are disjoint and exhaustive:
concatenating them reproduces the unpartitioned read exactly, **for any bounds, including
wrong ones**. Stale bounds cost you *skew*, never rows.

```python
ds = bt.read.sql(  # docs: skip
    "SELECT * FROM events",
    uri="postgresql://svc@warehouse:5432/app",
    password="env:PGPASSWORD",
    partition_on="event_id",
    lower_bound=1,
    upper_bound=50_000_000,
    num_partitions=16,
)
```

The two partitioning paths are not spelled identically, and the difference is real:

- **ADBC and DB-API** take `partition_on` + `lower_bound` + `upper_bound` +
  `num_partitions`, and Batcher generates the range predicates itself
  (`batcher.io.formats.sql.partition.range_predicates`). Both bounds are **required** when
  `partition_on` is set — omitting one raises `BackendError`.
- **ConnectorX** takes `partition_on` + `num_partitions` only. ConnectorX derives the
  ranges server-side itself, so there are no bounds to give it.

Pick a column that is numeric, indexed, and reasonably uniform. Partitioning on an
unindexed column makes the database do N full scans instead of one, which is slower than
not partitioning at all.

## pandas, SQLAlchemy, and DB-API cursor loops

| Source idiom | Batcher | Note |
|---|---|---|
| `create_engine(url)` + `pd.read_sql(q, engine)` | `bt.read.sql(q, uri=url)` | no engine object, nothing to dispose |
| `pd.read_sql_table("orders", engine)` | `bt.read.sql(uri=url, table="orders")` | |
| `pd.read_sql(q, engine, chunksize=n)` | `ds.iter_batches()` | streams Arrow batches, bounded memory |
| `pd.read_sql(...)` then filtering in pandas | `bt.read.sql(...).filter(...)` | the filter is pushed into the SQL |
| `con.cursor(); cur.execute(q); cur.fetchall()` | `bt.read.table("dbapi", module=..., query=q)` | |
| `cur.fetchmany(n)` loop | `batch_size=n` | rows are transposed to Arrow per block |
| `engine.dialect` / driver choice | *(nothing)* | the scheme decides |

The DB-API source is the universal fallback for any PEP 249 driver — `psycopg`, `pymysql`,
`cx_Oracle`, `pyodbc`, `sqlite3`, `ibm_db_dbi`, `teradatasql`, and the vendor long tail:

```python
ds = bt.read.table(  # docs: skip
    "dbapi",
    module="psycopg",
    connect_kwargs={"host": "warehouse", "dbname": "shop", "password": "env:PGPASSWORD"},
    table="orders",
    batch_size=16_384,
)
```

**Use it only when no Arrow-native driver exists.** It pays Python-object materialization
for every value and is several times slower than `bt.read.sql(query, uri=...)` against the
same database. Row conversion is confined to one `fetchmany(batch_size)` block at a time,
so `batch_size` is the throughput knob that matters here.

## Conceptual shifts that actually bite

- **The query runs on the warehouse, and Batcher makes it smaller.** Kyber pushes the
  projection and the filter into the SQL that is actually submitted, so
  `bt.read.sql("SELECT * FROM orders", ...).select("id", "amount").filter(bt.col("amount") > 100)`
  submits a narrower, filtered query. Do not hand-optimize the SQL string first — write
  the extract broadly and let the pushdown narrow it.
- **`schema()` costs nothing.** Every connector here probes with a zero-row `WHERE 1 = 0`
  query rather than running yours. Planning never executes the extract, so building a plan
  against a huge table is free.
- **Credentials are a reference, not a secret.** Pass `password="env:PGPASSWORD"` or
  `password="file:/run/secrets/pg"`. The *reference* is what gets pickled onto every split
  and resolved to a secret only on the worker that opens the connection. An inline URI
  password is lifted out of the URI automatically, but it still travels in the process
  image, so the reference form is the one to port to.
- **A URI reaches log lines.** `redact_uri` is applied where connection strings surface in
  errors and reprs, which is exactly why the password belongs in its own channel.
- **Nothing is a connection.** There is no engine, no pool, no cursor, and nothing to
  close. `bt.read.sql(...)` returns a lazy `Dataset`; the connection is opened per split,
  on the worker, at read time.
- **Parallelism is the partition count, not a thread pool.** Without `partition_on` a
  DB-API or ADBC read is one query on one worker, however large the table. That is the
  single most common reason a ported extract is slower than the tool it replaced.

## Porting recipe

1. **Inventory the extract.** List every connection string, every query, and every place
   the old code set a fetch size, a partition column, or a thread count. Note which
   queries are full-table reads (those become `table=`) and which are real SQL.
2. **Port the connection first, unpartitioned.** Turn the JDBC URL or SQLAlchemy URL into
   a `uri=`, drop any `jdbc:` prefix and `+driver` suffix concerns, and move the password
   to `password="env:…"`. Run one small query and confirm it returns rows before anything
   else.
3. **Port the query text verbatim.** It executes on the warehouse, so dialect-specific SQL
   is fine here — this is the opposite of `migrate-from-duckdb-sql`. Keep it byte-identical
   to the original so the two extracts are comparable.
4. **Check the schema.** `print(ds.schema)`. On the DB-API path a driver may report types
   too coarsely to resolve (PEP 249's `NUMBER` covers int, float, and decimal alike); pass
   `schema_override=` rather than letting inference guess.
5. **Add partitioning last, and only if the extract is slow.** Map `partitionColumn` /
   `lowerBound` / `upperBound` / `numPartitions` straight across. Approximate bounds are
   fine — they cut, they don't filter. Verify the row count is unchanged from step 2.
6. **Move the filtering into the plan.** Delete post-extract pandas filtering and
   `df = df[df.amount > 100]` lines; write them as `.filter(...)` so they push into the
   SQL. Confirm with `print(ds.explain())`.
7. **Port the sink.** `.write.jdbc(url, table, mode)` becomes
   `ds.write.sql(table, uri=..., mode=...)`, which ingests Arrow in bulk rather than row by
   row.
8. **Verify the ported extract returns the same rows.** Run the original extract and the
   ported one against the same database and compare **order-independently** — a SQL result
   has no order without an `ORDER BY`, and this is where ports quietly differ:

   ```python
   import batcher as bt

   batcher_rows = sorted(map(tuple, zip(*ported.to_pydict().values())))
   original_rows = sorted(tuple(r) for r in cur.execute(sql).fetchall())  # doctest: +SKIP
   assert batcher_rows == original_rows
   ```

   In-repo, mirror `tests/_harness.py::assert_same` — a multiset comparison tolerant of
   int↔float, Decimal→float, and float rounding. Use `assert_same_ordered` only when the
   query ends in an explicit `ORDER BY`. Import both from `_harness` rather than from
   `conftest`, which re-exports them but resolves ambiguously across test directories.

9. **Prove the partitioning did not change the answer.** Re-run step 8 with
   `num_partitions=1` and with your chosen count, and assert both match. The disjoint-and-
   exhaustive invariant is tested in-repo, but a wrong `partition_on` column (non-numeric,
   or one the database silently coerces) is your bug to catch.

## Known gaps — state these plainly

- **ConnectorX schemes have no separate password channel.** MySQL, SQL Server, Oracle,
  Redshift, Trino, and ClickHouse take credentials *inside* the URI. Passing `password=`
  with one of those raises `BackendError` rather than silently dropping it. Keep the whole
  URI in a secret store and load it as one string.
- **Bounds are not auto-probed.** There is no code path that derives min/max for you;
  `partition_on` without both bounds raises `BackendError`. Run the `MIN`/`MAX` query
  yourself, or use approximate values you already know — they are cut points, not
  filters, so being off costs skew rather than rows. Spark's JDBC reader also requires
  `lowerBound`/`upperBound` explicitly, so this is not a gap relative to it.
- **Partitioning needs a numeric, indexed column.** There is no string, hash, or date
  partitioning. A table whose only key is a UUID or a composite cannot be range-partitioned
  by this reader.
- **DB-API is the slow path, on purpose.** Several times slower than ADBC. It exists so
  that "my warehouse has a Python driver" is sufficient, not because it is a good place to
  stay.
- **ADBC and ConnectorX partition differently.** ConnectorX takes no bounds. Don't copy an
  ADBC call to a MySQL URI and expect `lower_bound=` to be accepted.
- **Narrow numerics widen at the FFI boundary.** Int8/16/32 become Int64 and Float16/32
  become Float64, so a ported extract may come back wider than the driver returned it.
  Don't assert exact type identity against the original.

## Gotchas / do-not

- **Do not embed a password in the URI** because it worked in SQLAlchemy. It reaches log
  lines and error messages. Use `password="env:…"` on ADBC schemes, or hold the whole URI
  as one secret on ConnectorX schemes.
- **Do not treat `lower_bound`/`upper_bound` as a filter.** Adding them does not restrict
  the extract. If you want a range, put it in the `WHERE` clause of the query.
- **Do not partition on an unindexed column.** N full scans is slower than one full scan.
  Check for the index before setting `partition_on`.
- **Do not raise `num_partitions` past what the warehouse will tolerate.** Each partition
  is an independent connection and an independent query. This is a load decision on a
  shared system, not a local tuning knob.
- **Do not `collect()` the extract to iterate it in Python.** Use `iter_batches()` — it is
  the port of `chunksize=` and it keeps memory bounded.
- **Do not reach for `bt.read.table("dbapi", ...)` first.** Check `known_schemes()` for an
  Arrow-native route before accepting the slow path.
- **Do not wrap a query in a subquery alias** out of JDBC habit. `dbtable="(SELECT …) t"`
  becomes a plain `query="SELECT …"`.

## See also

- `docs/user-guide/moving-data/reading-data.md` and `docs/api/relational/io.md` — the reader surface.
- `python/batcher/io/formats/sql/` — `uri.py` (scheme routing), `partition.py` (the
  cut-point invariant), `dbapi.py` (the PEP 249 fallback).
- Skills: `read-and-write-data` (the IO boundary generally), `migrate-from-spark` (the rest
  of a PySpark job once the extract is ported), `migrate-from-duckdb-sql` (porting the SQL
  itself to run *in* Batcher), `add-an-io-format-or-connector` (adding a new database
  backend), `run-a-distributed-job` (fanning the extract across a cluster).
