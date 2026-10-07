# SQLAlchemy

Batcher ships a SQLAlchemy 2.0 dialect over {doc}`batcher.dbapi <dbapi>`, registered as the `batcher://` URL scheme. SQLAlchemy Core queries, `text()` statements, and the inspector run against a Batcher session with no glue code.

:::{warning}
Not yet verified against a live SQLAlchemy application beyond this repository's tests; see tests/PENDING_VERIFICATION.md. The tests run Core queries, reflection, and writes on SQLAlchemy 2.0.
:::

## Connect an engine

Install the `sqlalchemy` extra, which pulls in SQLAlchemy 2.0. The extra's entry point registers the dialect, so `create_engine` finds it by URL. `batcher://` and `batcher+dbapi://` name the same dialect.

A URL can't name a session, since a session is a Python object. `create_engine("batcher://")` connects every pooled connection to {py:obj}`bt.current_session() <batcher.current_session>`. To use another session, pass it through `connect_args`. A URL that names a host, database, user, or query option is refused rather than ignored.

```python
# docs: skip
import batcher as bt
import sqlalchemy as sa

session = bt.Session()
session.register("orders", bt.from_pydict({"id": [1, 2, 3], "amount": [5.0, 7.5, 12.0]}))
engine = sa.create_engine("batcher://", connect_args={"session": session})

orders = sa.Table("orders", sa.MetaData(), autoload_with=engine)
query = sa.select(orders.c.id).where(orders.c.amount > sa.bindparam("lo")).order_by(orders.c.id)
with engine.connect() as conn:
    print(conn.execute(query, {"lo": 6.0}).all())
# [(2,), (3,)]
print(sa.inspect(engine).get_table_names())
# ['orders']
```

## How statements and reflection work

SQLAlchemy's compiler emits ANSI SQL with `?` placeholders. Batcher parses it in its default DuckDB dialect and binds each value as a typed literal. An `insert()` executed with a list of rows runs once per row through `executemany`.

The inspector reads `information_schema`. `get_table_names`, `get_view_names`, `has_table`, `get_schema_names`, and `get_columns` answer from it, and a column's Arrow type maps to the nearest SQLAlchemy type: `int64` to `BigInteger`, `double` to `Double`, `string` to `String`, a decimal to `Numeric` with its precision and scale, a timestamp to `DateTime` with `timezone` set when the Arrow type has one. A nested type such as a list or struct reflects as `NullType`, SQLAlchemy's "type unknown". Batcher declares no primary keys, foreign keys, or indexes, so those report none.

## Requirements and limitations

There are no transactions, and the only isolation level is `AUTOCOMMIT`. Each statement takes effect when it runs. A block opened with `engine.begin()` commits, which acknowledges the writes. A read-only `engine.connect()` block closes cleanly, because rolling back when nothing was written is a no-op. A block that writes and then rolls back, explicitly or by closing without a commit, raises `NotSupportedError`, because the write wasn't undone.

`information_schema.tables` lists the session's own tables and views, so a table in an attached catalog isn't reflected. Query it by its qualified name instead. `rowcount` is `-1`, so ORM features that check an UPDATE's row count, such as version counters, can't work.

## See also

- {doc}`dbapi`: the PEP 249 adapter this dialect drives.
- {doc}`/api/relational/sessions-and-catalogs`: sessions and the catalogs they attach.
