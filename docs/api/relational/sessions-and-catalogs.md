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

## See also

- {doc}`sql`: the SQL a session runs, and the constructs it supports.
- {doc}`/user-guide/moving-data/catalogs-and-tables`: catalogs, namespaces, and table writes, worked through.
- {doc}`/api/operations/governance`: attaching policy to the tables a catalog holds.
- {doc}`/api/symbols/readers-and-writers`: `bt.read.table` and `ds.write.table`, which resolve through a catalog.
