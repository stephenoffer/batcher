# Governance and sessions

Row and column policy, the principals it is evaluated against, query control, and the
SQL session that gives a workload its own catalog.

## Governance

Policy is a plan rewrite rather than a runtime check, so a principal who may not read a
column never causes it to be read.

{py:func}`bt.security <batcher.security>` is a context manager, not a setter. Policy
attaches when a table is *read*. A dataset built inside the block stays governed for its whole life, including
terminal operations that run after the block has exited. Read a table outside every block
and it is ungoverned. {doc}`/api/operations/governance` is the fuller reference, with row
filters, column masks and data residency.

```{eval-rst}
.. currentmodule:: batcher.governance

.. autosummary::
   :nosignatures:

   SecurityCatalog
   Principal
   GovernanceEvent
```

```{eval-rst}
.. currentmodule:: batcher

.. autosummary::
   :toctree: generated
   :nosignatures:

   security
   authenticate
   set_verifier
   current_verifier
   cancel_query
   running_queries
```

Two of those names are query control rather than policy.
{py:func}`running_queries <batcher.running_queries>` lists the ids executing in this
process, one per terminal operation, and {py:func}`cancel_query <batcher.cancel_query>`
asks one of them to stop. Cancellation is cooperative. The engine checks the flag between
morsels, between operators, and between spill merge passes, so a query part-way through
building a hash table notices when that build finishes rather than the instant you ask.
Column-level lineage is not here at all: it hangs off the dataset, at
{py:meth}`ds.lineage() <batcher.Dataset.lineage>`.

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

- {doc}`/api/operations/governance`: the same types with the enforcement model explained, plus residency and the verifiers.
- {doc}`/api/relational/sql`: what a `Session` runs once you've registered a table on it.
- {doc}`/user-guide/moving-data/catalogs-and-tables`: catalogs, namespaces and table writes, worked through.
- {doc}`/user-guide/trust/governance`: the worked introduction, from a first policy to an audit trail.
