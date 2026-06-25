# SQL

Batcher runs SQL through the same engine as the DataFrame API. {py:obj}`bt.sql(query, ...) <batcher.sql>`
parses a query, binds each named table to a Dataset, and returns a new Dataset.
Because the result is a Dataset, you can keep chaining DataFrame operations onto a
SQL query, or feed a DataFrame pipeline into SQL.

{py:obj}`bt.sql <batcher.sql>` reads DuckDB syntax by default; pass `dialect=` to
read another sqlglot dialect. For a reusable catalog of tables and Python
functions, build a {py:obj}`bt.Session <batcher.Session>` (the DuckDB-connection /
SparkSession analogue) — `bt.sql` and `bt.register_function` use a shared default session.

```python
import batcher as bt

ds = bt.from_pydict(
    {
        "category": ["a", "b", "a", "b", "a", "c"],
        "price": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    }
)
```

## Running a query

Pass the query string and bind each table name in the query to a Dataset (or a
pyarrow Table) as a keyword argument.

```python
out = bt.sql(
    "SELECT category, COUNT(*) AS n FROM t GROUP BY category ORDER BY category",
    t=ds,
)
print(out.to_pydict())
# {'category': ['a', 'b', 'c'], 'n': [3, 2, 1]}
```

The keyword name (`t` above) is the table identifier used in the `FROM` clause.

## Supported subset

A query may use:

- `SELECT` with column references, scalar expressions, and aggregates
- `WHERE` filters
- `GROUP BY` with `HAVING` (and `ROLLUP` / `CUBE` / `GROUPING SETS`)
- `ORDER BY`, `LIMIT` / `OFFSET`
- `INNER` / `LEFT` / `RIGHT` / `FULL` / `CROSS JOIN` (equi-keys; an extra non-equi
  `AND` condition is applied as a filter)
- `UNION` / `INTERSECT` / `EXCEPT`, `WITH` (CTEs), and subqueries
- Window functions, including explicit `ROWS BETWEEN …` frames
- `CASE` expressions and `CAST`

You can also register Python functions and call them from SQL, and define tables
and views with `CREATE`/`DROP` — see [Sessions and Python functions](#sessions-tables-and-python-functions).

## Filtering and projection

```python
out = bt.sql("SELECT category, price FROM t WHERE price >= 30 ORDER BY price", t=ds)
print(out.to_pydict())
# {'category': ['a', 'b', 'a', 'c'], 'price': [30.0, 40.0, 50.0, 60.0]}
```

## Aggregation with HAVING

```python
out = bt.sql(
    "SELECT category, SUM(price) AS total FROM t "
    "GROUP BY category HAVING SUM(price) > 60 ORDER BY category",
    t=ds,
)
print(out.to_pydict())
# {'category': ['a'], 'total': [90.0]}
```

## CASE and CAST

```python
out = bt.sql(
    "SELECT category, "
    "CASE WHEN price >= 40 THEN 'high' ELSE 'low' END AS tier, "
    "CAST(price AS BIGINT) AS price_int "
    "FROM t ORDER BY price",
    t=ds,
)
print(out.to_pydict())
# {'category': ['a', 'b', 'a', 'b', 'a', 'c'], 'tier': ['low', 'low', 'low', 'high', 'high', 'high'], 'price_int': [10, 20, 30, 40, 50, 60]}
```

## Joining tables

Bind one Dataset per table named in the query.

```python
dim = bt.from_pydict({"category": ["a", "b"], "region": ["west", "east"]})
out = bt.sql(
    "SELECT t.category, t.price, d.region "
    "FROM t INNER JOIN d ON t.category = d.category "
    "ORDER BY t.price",
    t=ds,
    d=dim,
)
print(out.to_pydict())
# {'category': ['a', 'b', 'a', 'b', 'a'], 'price': [10.0, 20.0, 30.0, 40.0, 50.0], 'region': ['west', 'east', 'west', 'east', 'west']}
```

## Mixing SQL and the DataFrame API

A SQL result is an ordinary Dataset, so you can continue with DataFrame methods.

```python
totals = bt.sql("SELECT category, SUM(price) AS total FROM t GROUP BY category", t=ds)
out = totals.filter(bt.col("total") >= 90).sort("category")
print(out.to_pydict())
# {'category': ['a'], 'total': [90.0]}
```

Both paths build the same logical plan, run through the same optimizer, and execute
on the same Rust data plane. There is no separate SQL engine.

(sessions-tables-and-python-functions)=

## Sessions, tables, and Python functions

A {py:obj}`bt.Session <batcher.Session>` holds a catalog of tables, registered
Python functions, and a dialect. Register a dataset as a table, then query it by
name:

```python
s = bt.Session()
s.register("t", ds)
print(s.sql("SELECT COUNT(*) AS n FROM t").to_pydict())
# {'n': [6]}
```

Register a Python function and call it from SQL. A scalar function is vectorized
(it receives an Arrow array); it lowers to the same `map_batches` path as the
DataFrame API, so Python and SQL share one plan:

```python
import pyarrow.compute as pc

s.register_function("discount", lambda a: pc.multiply(a, 0.9))
print(s.sql("SELECT discount(price) AS net FROM t ORDER BY price").to_pydict())
# {'net': [9.0, 18.0, 27.0, 36.0, 45.0, 54.0]}
```

`CREATE TABLE/VIEW AS` and `DROP TABLE` register and unregister a lazy table in the
session (no materialization until a terminal op):

```python
s.sql("CREATE VIEW cheap AS SELECT category, price FROM t WHERE price < 30")
print(s.sql("SELECT * FROM cheap ORDER BY price").to_pydict())
# {'category': ['a', 'b'], 'price': [10.0, 20.0]}
```

`ds.sql("... FROM self")` binds the current dataset directly:

```python
print(ds.sql("SELECT category FROM self WHERE price >= 50 ORDER BY price").to_pydict())
# {'category': ['a', 'c']}
```
