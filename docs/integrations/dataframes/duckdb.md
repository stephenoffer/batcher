# DuckDB

This page covers reading a DuckDB relation or query into Batcher, running the two engines side by side over the same Arrow, and the role DuckDB plays in Batcher's own test suite.

DuckDB is the engine Batcher is measured against most often, and it is also the one it agrees with by construction: every relational change here ships with a differential test that runs the same query on both and compares the rows.

| | |
| --- | --- |
| **Read** | {py:obj}`bt.from_duckdb(relation) <batcher.from_duckdb>`, or {py:obj}`bt.from_duckdb(connection, query) <batcher.from_duckdb>` |
| **Write** | Hand DuckDB the Arrow table from {py:obj}`ds.to_arrow() <batcher.Dataset.to_arrow>` and query it by variable name |
| **Extra** | `duckdb` |
| **Cost** | Zero-copy through Arrow, both directions |

## Read a relation or a query

Pass a relation and Batcher reads it:

```python
import batcher as bt
import duckdb

con = duckdb.connect()
con.execute(
    "CREATE TABLE orders AS SELECT id, region, CAST(amount AS DOUBLE) AS amount "
    "FROM (VALUES (1,'eu',120.0),(2,'us',80.0),(3,'eu',45.0)) v(id, region, amount)"
)

orders = bt.from_duckdb(con.sql("SELECT * FROM orders"))
print(orders.sort("id").to_pydict())
# {'id': [1, 2, 3], 'region': ['eu', 'us', 'eu'], 'amount': [120.0, 80.0, 45.0]}
```

Or pass a connection plus a query, which is the form to use when you want DuckDB to do the first step:

```python
totals = bt.from_duckdb(
    con, "SELECT region, SUM(amount) AS total FROM orders GROUP BY region ORDER BY region"
)
print(totals.to_pydict())
# {'region': ['eu', 'us'], 'total': [165.0, 80.0]}
```

## The same query, either engine

Batcher has its own SQL front end, and it builds the same plan the DataFrame verbs build. So the choice of engine for a given query is a runtime decision, not a rewrite:

```python
print(
    bt.sql(
        "SELECT region, SUM(amount) AS total FROM ds GROUP BY region ORDER BY region", ds=orders
    ).to_pydict()
)
# {'region': ['eu', 'us'], 'total': [165.0, 80.0]}
```

Going the other way, DuckDB queries a Python variable holding an Arrow table by name, so a Batcher result is a DuckDB table with no load step:

```python
tbl = orders.to_arrow()
print(con.sql("SELECT region, SUM(amount) AS total FROM tbl GROUP BY region ORDER BY region").fetchall())
# [('eu', 165.0), ('us', 80.0)]
```

That symmetry is the reason to keep both installed. Neither direction costs a copy, so you can move a step to whichever engine handles it better and measure the difference on your own data.

## What Batcher's SQL does and doesn't cover

Batcher parses SQL with `sqlglot` and lowers it to the same `LogicalPlan` the DataFrame API builds, so it supports what the plan supports rather than what a SQL standard says. {doc}`/api/relational/sql` is the honest list of supported and unsupported constructs, and {doc}`/user-guide/analyze/sql` walks the surface with runnable queries.

Two differences are worth knowing before you port a query. A Batcher `Dataset` is lazy, so {py:obj}`bt.sql(...) <batcher.sql>` returns a plan rather than a result, and nothing runs until a terminal call. And there is no connection: tables are passed as keyword arguments, or registered on a {py:obj}`Session <batcher.Session>` when you want a catalog that outlives one call.

## DuckDB as the oracle

[`tests/differential/`](https://github.com/stephenoffer/batcher/tree/main/tests/differential) runs relational queries on both engines and compares the results, and the benchmark harness refuses to record a timing for a query whose result doesn't match. That is why the comparison pages carry no numbers for cases where the answers disagree: a missing figure means a wrong answer was caught, not that a case was slow.

If you find a query where the two disagree, that is a bug worth reporting rather than a semantic choice.

## See also

- {doc}`/user-guide/analyze/sql`: the SQL guide, with sessions and registered Python functions.
- {doc}`/api/relational/sql`: the SQL surface, and what it lowers to.
- {doc}`/benchmarks/comparisons/vs-duckdb`: the measured comparison, including where DuckDB wins.
- {doc}`index`: the other in-process libraries and the zero-copy contract they share.
