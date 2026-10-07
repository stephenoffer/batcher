# SQL sessions and catalogs

This page is the reference for the two objects that give a workload its own namespace: a
{py:obj}`Session <batcher.Session>`, which is a SQL execution context, and a
{py:obj}`Catalog <batcher.Catalog>`, which is a set of named tables over one storage backend.

Everything on this page is control-plane metadata. Registering a table executes nothing, attaching a
catalog reads no data, and a session holds plans rather than results. That is what makes them cheap
enough to build per workload instead of sharing one global context.

## SQL sessions


A {py:class}`Session <batcher.Session>` is a SQL execution context: a table catalog, a
Python-function registry, and a dialect. It mirrors DuckDB's `con` and Spark's `SparkSession`.
Build one to scope a workload's tables and functions instead of sharing the default session
that {py:func}`bt.sql <batcher.sql>` and
{py:func}`bt.register_function <batcher.register_function>` fall back on. Everything a
session holds is control-plane metadata, so registering a table executes nothing.

```{eval-rst}
.. currentmodule:: batcher

.. autosummary::
   :toctree: generated
   :nosignatures:

   Session
   current_session
   set_session
```

### Scope a session to a block

{py:meth}`Session.activate <batcher.Session.activate>` makes a session the one {py:func}`bt.sql <batcher.sql>`, {py:func}`bt.register_function <batcher.register_function>` and `ds.write.table` use, for the code inside a `with` block. The scope is held in a `contextvars.ContextVar`, so a nested block wins until it exits, and two asyncio tasks that each activate their own session don't see each other's tables. When the block exits, the previous session is current again. {py:func}`bt.set_session <batcher.set_session>` still sets the process default, which applies wherever no block is active.

```python
import batcher as bt

scratch = bt.Session()
with scratch.activate():
    bt.sql("CREATE TABLE staging AS SELECT 1 AS x")
    print(bt.sql("SELECT x FROM staging").to_pydict())
# {'x': [1]}
print(scratch.list(), "staging" in bt.current_session())
# ['staging'] False
```

### Read-only sessions

`bt.Session(read_only=True)` refuses every SQL statement that creates, drops or changes a table, view or schema, raising {py:exc}`PlanError <batcher.PlanError>` before anything runs. Queries, `SHOW`, `DESCRIBE` and an `EXPLAIN` of a query still run. It's a guard on SQL statements, not a sandbox: Python methods such as `session.register` still work, and a registered Python function runs whatever code it holds.

```python
reader = bt.Session(read_only=True)
reader.register("t", bt.from_pydict({"x": [1, 2]}))
try:
    reader.sql("DROP TABLE t")
except bt.PlanError as err:
    print(err.message)
# this session is read-only and refuses the DROP statement
```

## Catalogs and tables


A {py:class}`Catalog <batcher.Catalog>` holds namespaces of tables over one storage backend:
in memory, a directory of Delta tables, or a pyiceberg catalog. A session reaches the catalogs
it has attached through `session.catalog`, a {py:class}`SessionCatalog
<batcher.api.catalog.SessionCatalog>` that also tracks the current catalog and namespace.
{py:class}`Table <batcher.Table>` is a handle on one table, and a write to it is
{py:meth}`ds.write.table <batcher.api.io_namespace.writer.Writer.table>`.
{doc}`/user-guide/moving-data/catalogs-and-tables` is the worked introduction.

```{eval-rst}
.. currentmodule:: batcher

.. autosummary::
   :toctree: generated
   :nosignatures:

   Catalog
   Table
   api.catalog.SessionCatalog
```

## Two meanings of "table"

Two families of APIs carry the word "table", and they look a name up in different places.

A *catalog table* is a name in a session or catalog: a dataset you registered, a view, or a table that `CREATE TABLE` or `ds.write.table` made. {py:meth}`Session.table <batcher.Session.table>`, {py:func}`bt.sql <batcher.sql>`, and {py:meth}`ds.write.table <batcher.api.io_namespace.writer.Writer.table>` all resolve it, first against the session's registered names and then against its attached catalogs.

`bt.read.table(format, ...)` is *format dispatch*. Its first argument names a source format such as `"delta"`, and it forwards the rest to that reader, so `bt.read.table("delta", uri)` is `bt.read.delta(uri)`. It never consults a session or catalog. Passing it a catalog name fails with an unknown-source error.

```python
import batcher as bt

session = bt.Session()
session.register("orders", bt.from_pydict({"id": [1, 2], "amount": [10, 20]}))
print(session.table("orders").to_pydict())
# {'id': [1, 2], 'amount': [10, 20]}
print(session.sql("SELECT sum(amount) AS total FROM orders").to_pydict())
# {'total': [30]}
```

## Threads and concurrency

The objects on this page differ in what they let threads share. The following list says which can be shared and which can't, and `tests/integration/test_thread_safety_contract.py` checks each statement:

- A {py:class}`Dataset <batcher.Dataset>` is immutable. No method changes it, so threads can share one, build on it, and collect it at the same time.
- A {py:class}`Session <batcher.Session>` and its catalogs are plain mutable registries with no locking. Don't register, drop, or `CREATE` in one session from several threads at once. Give each concurrent workload its own `bt.Session()` and call its `sql` and `table` directly.
- The default session behind {py:func}`bt.sql <batcher.sql>` is process-global. {py:func}`bt.set_session <batcher.set_session>` in one thread changes it for every thread, so a multi-threaded program should pass sessions around rather than swap the default.
- {py:func}`config_context <batcher.config_context>` is scoped to the calling thread's context, so a block in one thread doesn't affect a query running in another.
- {py:func}`set_config <batcher.set_config>` also sets the calling context only. A thread started afterwards begins from the defaults, so call `set_config` or open a `config_context` inside each worker thread that needs the setting.
- `iter_batches` and `iter_rows` return Python generators. Each belongs to one consumer, so don't advance one from two threads. Call `iter_batches` once per consumer instead, which runs the query once for each.

## See also

- {doc}`sql`: the SQL a session runs, and the constructs it supports.
- {doc}`/user-guide/moving-data/catalogs-and-tables`: catalogs, namespaces, and table writes, worked through.
- {doc}`/api/operations/governance`: attaching policy to the tables a catalog holds.
- {doc}`/api/symbols/readers-and-writers`: `ds.write.table`, which resolves through a catalog, and `bt.read.table`, which dispatches on a format name.
