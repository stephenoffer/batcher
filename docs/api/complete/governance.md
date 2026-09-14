# Governance and sessions

Row and column policy, the principals it is evaluated against, query control, and the
SQL session that gives a workload its own catalog.

## Governance

Policy is a plan rewrite rather than a runtime check, so a principal who may not read a
column never causes it to be read.

{py:func}`bt.security <batcher.security>` is a context manager and not a setter, which is
the one thing about this surface worth reading twice: policy attaches when a table is
*read*, so a dataset built inside the block stays governed for its whole life, including
terminal operations that run after the block has exited. Read a table outside every block
and it is ungoverned. {doc}`/api/operations/governance` is the fuller reference, with row
filters, column masks and data residency.

```{eval-rst}
.. autoclass:: batcher.SecurityCatalog
   :members:

.. autoclass:: batcher.Principal
   :members:

.. autoclass:: batcher.GovernanceEvent
   :members:

.. autofunction:: batcher.security

.. autofunction:: batcher.authenticate

.. autofunction:: batcher.set_verifier

.. autofunction:: batcher.current_verifier

.. autofunction:: batcher.cancel_query

.. autofunction:: batcher.running_queries
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
.. autoclass:: batcher.Session
   :members:
   :member-order: groupwise
```

## See also

- {doc}`/api/operations/governance`: the same types with the enforcement model explained, plus residency and the verifiers.
- {doc}`/api/relational/sql`: what a `Session` runs once you've registered a table on it.
- {doc}`/user-guide/trust/governance`: the worked introduction, from a first policy to an audit trail.
