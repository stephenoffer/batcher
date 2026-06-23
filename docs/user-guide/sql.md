# SQL

Batcher runs SQL through the same engine as the DataFrame API. `bt.sql(query, ...)`
parses a query, binds each named table to a Dataset, and returns a new Dataset.
Because the result is a Dataset, you can keep chaining DataFrame operations onto a
SQL query, or feed a DataFrame pipeline into SQL.

`bt.sql` is single-dialect. It supports a focused subset of SQL, not a multi-dialect
translation layer.

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
- `GROUP BY` with `HAVING`
- `ORDER BY` and `LIMIT`
- `INNER JOIN` and `LEFT JOIN`
- `CASE` expressions and `CAST`

Anything outside this subset (window syntax, set operations, subqueries) is better
expressed with the DataFrame API, which exposes those operators directly.

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
on the same Rust data plane. There is no separate SQL engine and no `SQLContext`
object to manage.
