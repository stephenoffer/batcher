# Data scientist learning path

For interactive analysis: shape data with expressions, ask questions with SQL or the
DataFrame API, and summarize with aggregations. The API is lazy and immutable, so you
compose a query and materialize it with a terminal operation when you want the
answer.

## Reading order

1. [Getting started](../getting-started/index.md): install and run a first query.
2. [Concepts](../getting-started/concepts/index.md): datasets, laziness, expressions.
3. [Expressions](../user-guide/expressions.md): column math, conditionals, string
   and date accessors.
4. [Filtering](../user-guide/filtering.md): predicates and `is_in` / `between`.
5. [Aggregations](../user-guide/aggregations.md): `group_by`, `.agg`, quantiles.
6. [SQL](../user-guide/sql.md): query a dataset with {py:obj}`bt.sql <batcher.sql>`.
7. [Window functions](../user-guide/window-functions.md): ranking and rolling
   aggregates.
8. [Expression API reference](../api/expressions.md) and
   [SQL API reference](../api/sql.md).

## Example: derive and summarize

```python
import batcher as bt

sales = bt.from_pydict(
    {
        "category": ["a", "b", "a", "b", "a", "c"],
        "price": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    }
)

summary = (
    sales.with_columns(bucket=bt.when(bt.col("price") > 35.0).then(bt.lit("high")).otherwise(bt.lit("low")))
    .group_by("bucket")
    .agg(avg_price=bt.col("price").mean(), n=bt.count())
    .sort("bucket")
)
print(summary.to_pydict())
# {'bucket': ['high', 'low'], 'avg_price': [50.0, 20.0], 'n': [3, 3]}
```

## Example: ask the same question in SQL

{py:obj}`bt.sql <batcher.sql>` runs a query against a dataset bound to a table name and returns a new
dataset.

```python
counts = bt.sql(
    "SELECT category, COUNT(*) AS n FROM t GROUP BY category ORDER BY category",
    t=sales,
)
print(counts.to_pydict())
# {'category': ['a', 'b', 'c'], 'n': [3, 2, 1]}
```
