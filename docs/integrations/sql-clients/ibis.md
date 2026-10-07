# Ibis

The Ibis bridge runs an Ibis expression on Batcher. Ibis compiles the expression to SQL with its public `ibis.to_sql`, and Batcher runs that SQL through `Session.sql`, so the result is a lazy {py:class}`Dataset <batcher.Dataset>` and nothing passes through pandas.

:::{warning}
Not yet verified against a live Ibis installation; see tests/PENDING_VERIFICATION.md. The tests run SQL written the way Ibis's DuckDB compiler writes it, not SQL produced by a given Ibis release.
:::

## Run an expression

Install the `ibis` extra. {py:func}`batcher.integrations.ibis.table` describes a session table to Ibis as an unbound table with the same name and schema. {py:func}`batcher.integrations.ibis.to_dataset` compiles an expression over such tables and runs it on the session, where each table resolves by name:

```python
# docs: skip
import batcher as bt
from batcher.integrations import ibis as bt_ibis

session = bt.Session()
session.register("orders", bt.from_pydict({"g": ["a", "b", "a"], "v": [5, 7, 9]}))
orders = bt_ibis.table("orders", session)

expr = orders.group_by("g").aggregate(total=orders.v.sum()).order_by("g")
result = bt_ibis.to_dataset(expr, session)
print(result.to_arrow().to_pydict())
# {'g': ['a', 'b'], 'total': [14, 7]}
```

## Why a bridge rather than a backend

An Ibis backend, the kind `ibis.duckdb.connect()` reaches, subclasses Ibis's backend base classes. Those are internal to Ibis and change between releases. The bridge uses only `ibis.to_sql`, `ibis.table`, and `ibis.Schema.from_pyarrow`, which are documented, so it doesn't break when those internals move. The cost is that there is no `ibis.batcher.connect()`. You call `to_dataset` instead.

## Requirements and limitations

The SQL Ibis emits must be SQL Batcher translates. Projections, filters, computed columns, `group_by` with `aggregate` over the common reductions, `order_by`, `limit`, joins, `distinct`, and unions are the documented subset. A construct outside it raises `SQLUnsupportedError` naming the construct. Each table an expression names must be registered in the session under that name.

## See also

- {doc}`/api/relational/sql`: the SQL Batcher translates.
- {doc}`/api/operations/sql-clients`: the reference for the bridge.
