# DB-API (PEP 249)

`batcher.dbapi` is a PEP 249 adapter over a {py:class}`Session <batcher.Session>`. A Python program written against the standard database API, or a library that takes a DB-API connection, runs SQL on Batcher through it and reads rows back as tuples.

## How a connection relates to a session

A *connection* wraps a session. It doesn't own a database. {py:func}`batcher.dbapi.connect` with no argument uses {py:obj}`bt.current_session() <batcher.current_session>`, so a table you register through the Python API is visible to the cursor straight away. Pass a session to scope the connection to it. `conn.cursor()` returns a `Cursor`, which runs one statement at a time and holds its result until the next `execute`. Closing the connection closes its cursors and leaves the session untouched, because the session belongs to you.

```python
import batcher as bt
from batcher import dbapi

session = bt.Session()
session.register("orders", bt.from_pydict({"id": [1, 2, 3], "amount": [5.0, 7.5, 12.0]}))

conn = dbapi.connect(session)
cur = conn.cursor()
cur.execute("SELECT id, amount FROM orders WHERE amount > ? ORDER BY id", [6.0])
print(cur.fetchall())
# [(2, 7.5), (3, 12.0)]
print([d[0] for d in cur.description], cur.description[0][1] == dbapi.NUMBER)
# ['id', 'amount'] True
conn.close()
```

Each `description` entry's type code is the column's pyarrow type, and it compares equal to the PEP 249 type object for its family: `STRING`, `BINARY`, `NUMBER`, or `DATETIME`. Batcher has no row ids, so no column equals `ROWID`.

The module globals say what the adapter is: `apilevel` is `"2.0"`, `threadsafety` is `1`, which means threads may share the module but not a connection, and `paramstyle` is `"qmark"`.

## Bind parameters

A sequence fills `?` placeholders, and a mapping fills `$name` placeholders. Each value is bound into the parsed statement as a typed literal and never spliced into the SQL text, so a string holding a quote is only ever a string. DuckDB's grammar reads `name: expr` in a select list as an alias, so write `$name` rather than `:name` there.

```python
cur = dbapi.connect(session).cursor()
print(cur.execute("SELECT $lo + $hi AS total", {"lo": 1, "hi": 2}).fetchone())
# (3,)
print(cur.execute("SELECT COUNT(*) FROM orders WHERE CAST(id AS VARCHAR) = ?", ["1' OR '1'='1"]).fetchone())
# (0,)
```

The values a parameter accepts, and the SQL type each binds as, are the ones `Session.sql` takes through `params=`. The PEP 249 constructors `Date`, `Time`, `Timestamp`, `DateFromTicks`, `TimeFromTicks`, `TimestampFromTicks`, and `Binary` return those plain Python values.

## Fetch rows

A cursor does no work when `execute` returns. The first fetch starts the query, and rows arrive one Arrow batch at a time, so `fetchone` and `fetchmany` over a large result hold one batch rather than the whole relation. `fetchmany()` without a size returns `arraysize` rows. A cursor is also iterable.

Rows as tuples are what PEP 249 returns, and building them costs a Python object per value. To keep a large result in Arrow, call `fetch_arrow_table`, which DuckDB and ADBC cursors also offer:

```python
cur.execute("SELECT i FROM range(100000) t(i) ORDER BY i")
print(cur.fetchone())
# (0,)
print(cur.fetch_arrow_table().num_rows)
# 99999
```

## Statements that change the session

`CREATE`, `DROP`, `INSERT`, `UPDATE`, `DELETE`, `MERGE`, `ALTER`, `USE`, and `SET` produce no result set, so `description` is `None` after them and a fetch raises `ProgrammingError`. A DML statement with `RETURNING` does produce rows. `executemany` runs a statement once per parameter set and refuses a query, since PEP 249 leaves a result set there undefined.

```python
cur.execute("CREATE TABLE audit AS SELECT 0 AS n")
cur.executemany("INSERT INTO audit VALUES (?)", [[1], [2], [3]])
print(cur.description, cur.execute("SELECT SUM(n) FROM audit").fetchone())
# None (6,)
```

## Errors

Every failure is one of the PEP 249 classes, with the Batcher error chained as `__cause__`. The mapping follows who can fix the problem:

| Batcher raises | The cursor raises |
|---|---|
| `SQLSyntaxError`, `ColumnNotFoundError`, other `PlanError`, `ConfigError` | `ProgrammingError` |
| `SQLUnsupportedError` | `NotSupportedError` |
| `DataQualityError`, `SchemaError`, `FormatError` | `DataError` |
| `ExecutionError` (including a cancelled query), `ResourceError`, `TransportError`, an IO failure | `OperationalError` |
| Any other error | `DatabaseError` |
| Using a closed cursor or connection | `InterfaceError` |

`IntegrityError`, `InternalError`, and `Warning` exist because PEP 249 names them. Batcher enforces no keys, so `IntegrityError` is never raised.

## Requirements and limitations

There are no transactions. Each statement takes effect when it runs. `commit()` does nothing and returns. `rollback()` returns quietly when nothing was written since the last `commit()`, because then there's nothing to undo. When a statement that writes has run since then, `rollback()` raises `NotSupportedError` instead of pretending the write was undone.

```python
conn = dbapi.connect(bt.Session())
conn.cursor().execute("CREATE TABLE t AS SELECT 1 AS x")
try:
    conn.rollback()
except dbapi.NotSupportedError as exc:
    print(type(exc).__name__)
# NotSupportedError
```

`rowcount` is always `-1`, PEP 249's "not determinable". A query's rows are produced as they're fetched, and a DML statement on a session table is a plan rewrite that runs on a later read, so no count exists when `execute` returns. `description` reports each column's name and pyarrow type, and leaves display size, internal size, precision, scale, and nullability as `None`.

A session isn't locked, so share a connection between threads only behind your own lock. A session built with `read_only=True` refuses writes through the cursor as it does through `Session.sql`.

## See also

- {doc}`sqlalchemy`: the SQLAlchemy dialect built on this adapter.
- {doc}`/api/operations/sql-clients`: the reference for every name in `batcher.dbapi`.
