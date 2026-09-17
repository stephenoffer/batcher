# Catalogs and tables

This page covers named tables: creating them in a catalog, writing to them with a save mode, reading them back by name from Python and SQL, and attaching catalogs that live in memory, in a directory of Delta tables, or behind an Iceberg catalog service.

## What is a catalog?

A *catalog* maps dotted names such as `sales.orders` to tables and owns their lifecycle. It holds *namespaces*, and each namespace holds tables. Spark calls a namespace a database, and DuckDB calls it a schema. A {py:class}`Catalog <batcher.Catalog>` stores its tables in one backend, chosen when you build it.

Every {py:class}`Session <batcher.Session>` has a {py:class}`SessionCatalog <batcher.api.catalog.SessionCatalog>`, reached as `session.catalog`. It holds the catalogs the session has attached, and a *current* catalog and namespace that an unqualified name resolves into. A fresh session starts with one in-memory catalog named `memory` and the current namespace `main`, which are DuckDB's names for the same two things.

A session also holds *views*, which are what {py:meth}`Session.register <batcher.Session.register>` creates. A view binds a name to a lazy plan in that session and stores nothing. A catalog table stores rows: writing to it runs the plan, and the table keeps the result. When a view and a catalog table share a name, the view wins.

Set up a session to follow along:

```python
import tempfile

import pyarrow.compute as pc

import batcher as bt

session = bt.Session()
print(session.catalog.current_catalog(), session.catalog.current_namespace())
# memory main
```

## Create a table and write to it

{py:meth}`create_table <batcher.api.catalog.SessionCatalog.create_table>` makes a table from a dataset's rows, or an empty table from a pyarrow schema. A namespace has to exist before a table can be created in it, so create it first with {py:meth}`create_namespace <batcher.api.catalog.SessionCatalog.create_namespace>`:

```python
session.catalog.create_namespace("sales")
orders = bt.from_pydict({"id": [1, 2], "region": ["eu", "us"], "amount": [10.0, 20.0]})
table = session.catalog.create_table("sales.orders", orders, properties={"owner": "finance"})
print(table.name, table.properties)
# memory.sales.orders {'owner': 'finance'}
```

The returned {py:class}`Table <batcher.Table>` handle has a fully qualified `name`, a `schema` read from metadata, `properties`, and `read()` for its rows.

After that, every write goes through {py:meth}`ds.write.table <batcher.api.io_namespace.writer.Writer.table>`. `mode` says what to do about the table existing, and the default refuses to overwrite anything:

```python
more = bt.from_pydict({"id": [3], "region": ["eu"], "amount": [5.0]})
more.write.table("sales.orders", mode="append", session=session)
print(session.table("sales.orders").sort("id").to_pydict()["id"])
# [1, 2, 3]
```

The modes are the following, in the order a reader usually needs them:

| Mode | Table missing | Table exists |
|---|---|---|
| `"error"` (default) | Create it | Raise `PlanError` |
| `"append"` | Create it | Add the rows |
| `"overwrite"` | Create it | Replace rows and schema |
| `"overwrite_partitions"` | Raise | Replace only the partitions the rows cover |
| `"replace"` | Raise | Replace rows and schema |
| `"ignore"` | Create it | Do nothing |

Two parameters refine an existing-table write. `by_name=False` matches columns by position rather than by name, which is how SQL `INSERT` and Spark's `insertInto` behave. An append by name fills a table column the rows don't have with NULL. `replace_where=` scopes `mode="overwrite"` to the rows a predicate matches and keeps the rest:

```python
fix = bt.from_pydict({"id": [4], "region": ["eu"], "amount": [99.0]})
fix.write.table(
    "sales.orders", mode="overwrite", replace_where=bt.col("region") == "eu", session=session
)
print(session.table("sales.orders").sort("id").to_pydict())
# {'id': [2, 4], 'region': ['us', 'eu'], 'amount': [20.0, 99.0]}
```

Without `session=`, a write resolves the name in the process-default session, the one {py:func}`bt.sql <batcher.sql>` uses.

## Read a table

{py:meth}`Session.table <batcher.Session.table>` returns a lazy dataset over a view or a catalog table. A name can be as qualified as it needs to be: `"orders"` resolves in the current namespace, `"sales.orders"` in the current catalog, and `"memory.sales.orders"` names the catalog too. SQL resolves the same names:

```python
print(session.sql("SELECT region, sum(amount) AS total FROM sales.orders GROUP BY region ORDER BY region").to_pydict())
# {'region': ['eu', 'us'], 'total': [99.0, 20.0]}
```

A handle from {py:meth}`get_table <batcher.api.catalog.SessionCatalog.get_table>` answers questions about a table without reading its rows:

```python
handle = session.catalog.get_table("sales.orders")
print(handle.schema.names, handle.read().count())
# ['id', 'region', 'amount'] 2
```

## Attach more catalogs

A {py:class}`Catalog <batcher.Catalog>` comes from one of three constructors:

- {py:meth}`Catalog.from_pydict <batcher.Catalog.from_pydict>` holds tables in memory. Its keys are `"table"` or `"namespace.table"`, and a dataset value stays lazy until something writes to that table.
- {py:meth}`Catalog.from_directory <batcher.Catalog.from_directory>` treats a directory as a warehouse of Delta tables laid out as `<path>/<namespace>/<table>`. There is no service to run: listing the catalog is listing the directory, and each table is an ordinary Delta table.
- {py:meth}`Catalog.from_iceberg <batcher.Catalog.from_iceberg>` wraps a pyiceberg catalog, either a live object or the spec `bt.read.iceberg` takes, such as `{"type": "glue", ...}` or `{"type": "rest", ...}`.

{py:meth}`attach <batcher.api.catalog.SessionCatalog.attach>` adds a catalog to a session under its name or an alias, and {py:meth}`use <batcher.api.catalog.SessionCatalog.use>` makes a catalog, a namespace, or `catalog.namespace` current, which is SQL `USE`:

```python
lake = bt.Catalog.from_directory(tempfile.mkdtemp(), name="lake")
session.catalog.attach(lake)
print(session.catalog.list_catalogs(), session.catalog.has_catalog("lake"))
# ['lake', 'memory'] True

orders.write.table("lake.main.orders", partition_by=["region"], session=session)
session.catalog.use("lake")
print(session.catalog.current_catalog(), session.catalog.list_tables())
# lake ['main.orders']
```

A partitioned table can reload some partitions and leave the rest alone. The write below replaces the `eu` partition and keeps `us`:

```python
reload = bt.from_pydict({"id": [7], "region": ["eu"], "amount": [1.0]})
reload.write.table("orders", mode="overwrite_partitions", session=session)
print(session.table("orders").sort("id").to_pydict()["id"])
# [2, 7]
```

{py:meth}`get_catalog <batcher.api.catalog.SessionCatalog.get_catalog>` returns an attached catalog, and {py:meth}`detach <batcher.api.catalog.SessionCatalog.detach>` removes it from the session without touching its tables. The current catalog can't be detached, so switch away first:

```python
session.catalog.use("memory.sales")
print(session.catalog.current_namespace(), session.catalog.get_catalog("lake").list_tables())
session.catalog.detach("lake")
# sales ['main.orders']
```

## Manage namespaces and tables

The session-level calls act on the current catalog. The ones that take a table or namespace name also accept a name qualified by an attached catalog. The same methods exist on a {py:class}`Catalog <batcher.Catalog>` itself.

- {py:meth}`list_namespaces <batcher.api.catalog.SessionCatalog.list_namespaces>` and {py:meth}`list_tables <batcher.api.catalog.SessionCatalog.list_tables>` take an optional glob, such as `"sales.*"`.
- {py:meth}`has_namespace <batcher.api.catalog.SessionCatalog.has_namespace>` and {py:meth}`has_table <batcher.api.catalog.SessionCatalog.has_table>` test for existence.
- {py:meth}`truncate_table <batcher.api.catalog.SessionCatalog.truncate_table>` removes every row and keeps the schema.
- {py:meth}`drop_table <batcher.api.catalog.SessionCatalog.drop_table>` and {py:meth}`drop_namespace <batcher.api.catalog.SessionCatalog.drop_namespace>` take `if_exists=`, and dropping a namespace that still holds tables needs `cascade=True`.

```python
session.catalog.use("main")
print(session.catalog.list_namespaces(), session.catalog.list_tables("sales.*"))
session.catalog.truncate_table("sales.orders")
print(session.table("sales.orders").count(), session.catalog.has_namespace("sales"))
session.catalog.drop_namespace("sales", cascade=True)
print(session.catalog.has_table("sales.orders"))
# ['main', 'sales'] ['sales.orders']
# 0 True
# False
```

## Use catalogs from SQL

The catalog statements follow DuckDB's syntax and result shapes. `CREATE SCHEMA`, `DROP SCHEMA [CASCADE]`, `USE`, `SHOW TABLES [FROM ns]`, `SHOW DATABASES` and `SHOW SCHEMAS` act on the session's catalogs. `CREATE [OR REPLACE] TABLE ns.t AS <select>`, `INSERT INTO ns.t` and `DROP TABLE ns.t` act on catalog tables, and `current_catalog()`, `current_schema()` and `current_database()` report the session's position:

```python
session.sql("CREATE SCHEMA staging")
session.sql("CREATE TABLE staging.events AS SELECT 1 AS id, 'open' AS kind")
session.sql("INSERT INTO staging.events VALUES (2, 'close')")
session.sql("USE staging")
print(session.sql("SELECT current_schema() AS s, count(*) AS n FROM events").to_pydict())
print(session.sql("SHOW TABLES").to_pydict())
# {'s': ['staging'], 'n': [2]}
# {'name': ['events']}
```

An unqualified `CREATE TABLE t AS <select>` still registers a view, as it did before catalogs existed. Qualify the name to create a catalog table.

## Views and functions on a session

{py:meth}`Session.register <batcher.Session.register>` replaces a view of the same name by default. Pass `replace=False` to refuse instead, which is Spark's `createTempView`. Registered SQL functions have the matching pair of checks, {py:meth}`Session.has_function <batcher.Session.has_function>` and {py:meth}`Session.drop_function <batcher.Session.drop_function>`:

```python
session.register("recent", bt.from_pydict({"id": [1]}), replace=False)
session.register_function("double", lambda a: pc.multiply(a, 2), result_type="int64")
print(session.has_function("double"))
session.drop_function("double")
print(session.list_functions())
# True
# []
```

## The default session

{py:func}`bt.current_session <batcher.current_session>` returns the process-default session that `bt.sql` and `ds.write.table` use when no session is passed, and {py:func}`bt.set_session <batcher.set_session>` installs another one:

```python
previous = bt.current_session()
bt.set_session(session)
print(bt.sql("SELECT count(*) AS n FROM events").to_pydict())
bt.set_session(previous)
# {'n': [2]}
```

## Names in other engines

The table below maps each engine's spelling onto Batcher's, alphabetically by the other engine's name:

| Spark, Daft or Polars | Batcher |
|---|---|
| Daft `Catalog.from_iceberg` / `from_glue` / `from_unity` | `bt.Catalog.from_iceberg(spec)` with `type` `"rest"`, `"glue"` or `"unity"` |
| Daft `attach_catalog` / `detach_catalog` | `session.catalog.attach` / `session.catalog.detach` |
| Daft `create_temp_view`, Spark `createOrReplaceTempView` | `session.register(name, ds)` |
| Daft `write_table`, Polars `write_table` | `ds.write.table(name, mode=...)` |
| Spark `DataFrameWriterV2.create` / `createOrReplace` / `replace` | `mode="error"` / `"overwrite"` / `"replace"` |
| Spark `DataFrameWriterV2.overwrite(cond)` / `overwritePartitions` | `mode="overwrite", replace_where=cond` / `mode="overwrite_partitions"` |
| Spark `catalog.listDatabases`, Daft `list_namespaces` | `session.catalog.list_namespaces()` |
| Spark `catalog.truncateTable` | `session.catalog.truncate_table(name)` |
| Spark `catalog.setCurrentDatabase`, Daft `use` | `session.catalog.use(name)` |
| Spark `insertInto` | `ds.write.table(name, mode="append", by_name=False)` |
| Spark `saveAsTable` | `ds.write.table(name, mode=...)` |
| Spark `spark.table`, Daft `read_table` | `session.table(name)` |

## Requirements and limitations

- A directory catalog stores Delta tables and needs the `delta` extra. It is not the Hive warehouse layout (`<db>.db/<table>`), so an existing Spark warehouse directory isn't read as a catalog.
- A Delta table scopes `replace_where` to its partition columns, so an overwrite with a predicate needs a partitioned table there. The in-memory backend takes any predicate.
- `mode="overwrite_partitions"` commits one scoped overwrite per partition the rows cover. Each commit is atomic and the set isn't, so a concurrent reader can see some partitions reloaded before others.
- An Iceberg catalog refuses `partition_by` on create, because Iceberg partitioning is a partition spec declared on the table. Create the table with pyiceberg, then write to it. Dropping an Iceberg table removes its catalog entry and leaves the data files.
- `DELETE`, `UPDATE` and `MERGE` statements act on views only. For a catalog table, use `replace_where` or {py:meth}`ds.write.merge_into <batcher.api.io_namespace.writer.Writer.merge_into>` on the underlying format.
- `information_schema.tables` and `information_schema.columns` list views only. `SHOW TABLES` lists views and the current namespace's catalog tables, and `DESCRIBE` answers for either.

## See also

- {doc}`writing-data`: path writes and the save modes they share.
- {doc}`lakehouse`: Delta and Iceberg tables addressed by path or identifier.
- {doc}`/user-guide/analyze/sql`: the SQL surface a session runs.
- {doc}`/api/complete/governance`: the `Session`, `Catalog` and `Table` reference.
