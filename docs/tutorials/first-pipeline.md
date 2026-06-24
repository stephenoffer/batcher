# Your first pipeline

This tutorial builds a complete pipeline from an in-memory dataset: derive a column,
group and aggregate, sort, and collect the result. Everything here runs as written.
The last step shows the same pipeline reading from a file, which needs a real path
and so is not executed.

## Build a dataset

A `Dataset` is a lazy, immutable handle to a query plan. {py:obj}`bt.from_pydict <batcher.from_pydict>` builds one
from a column-oriented dict. No work runs until a terminal operation.

```python
import batcher as bt

ds = bt.from_pydict(
    {
        "category": ["a", "b", "a", "b", "a"],
        "price": [10.0, 20.0, 30.0, 40.0, 50.0],
        "qty": [1, 2, 3, 4, 5],
    }
)

print(ds.columns)
# ['category', 'price', 'qty']
```

## Derive a column

Column work is expressed with `Expr`. `with_columns` adds or replaces columns and
keeps the rest. The arithmetic runs in the Rust data plane, not in Python.

```python
priced = ds.with_columns(total=bt.col("price") * bt.col("qty"))
print(priced.to_pydict())
# {'category': ['a', 'b', 'a', 'b', 'a'], 'price': [10.0, 20.0, 30.0, 40.0, 50.0], 'qty': [1, 2, 3, 4, 5], 'total': [10.0, 40.0, 90.0, 160.0, 250.0]}
```

## Group and aggregate

`group_by(*keys)` returns a `GroupBy`; finalize it with `.agg(**named_aggs)`.
Aggregates are passed as keyword arguments where the name becomes the output column.

```python
summary = priced.group_by("category").agg(
    revenue=bt.col("total").sum(),
    orders=bt.count(),
)
print(summary.to_pydict())
# {'category': ['a', 'b'], 'revenue': [350.0, 200.0], 'orders': [3, 2]}
```

## Sort and collect

`sort` orders rows; `descending=True` reverses it. A terminal operation executes the
plan: `to_pydict` returns a column dict, and `collect` returns a pyarrow `Table`.

```python
ranked = summary.sort("revenue", descending=True)
print(ranked.to_pydict())
# {'category': ['a', 'b'], 'revenue': [350.0, 200.0], 'orders': [3, 2]}

table = ranked.collect()
print(table.num_rows)
# 2
```

The whole pipeline reads as one expression because every step returns a new
`Dataset`:

```python
result = (
    bt.from_pydict(
        {
            "category": ["a", "b", "a", "b", "a"],
            "price": [10.0, 20.0, 30.0, 40.0, 50.0],
            "qty": [1, 2, 3, 4, 5],
        }
    )
    .with_columns(total=bt.col("price") * bt.col("qty"))
    .group_by("category")
    .agg(revenue=bt.col("total").sum(), orders=bt.count())
    .sort("revenue", descending=True)
)
print(result.to_pydict())
# {'category': ['a', 'b'], 'revenue': [350.0, 200.0], 'orders': [3, 2]}
```

## Inspect the plan

`explain()` returns the optimized plan as text without executing it, which is useful
when checking what the optimizer did.

```python
print(isinstance(result.explain(), str))
# True
```

## The same pipeline over files

Only the source changes when the data lives in files or object storage. The
transformations and terminal operations are identical. This block needs a real file,
so it is shown but not run.

```python
# docs: skip
import batcher as bt

(
    bt.read.parquet("s3://bucket/orders.parquet")
    .with_columns(total=bt.col("price") * bt.col("qty"))
    .group_by("category")
    .agg(revenue=bt.col("total").sum(), orders=bt.count())
    .sort("revenue", descending=True)
    .write.parquet("output/revenue_by_category.parquet")
)
```

## Next steps

- [Batch inference](batch-inference.md): run a model over batches.
- [Synthetic data generation](synthetic-data-generation.md): build larger test
  inputs.
- [Aggregations](../user-guide/aggregations.md) and
  [Expressions](../user-guide/expressions.md) in the user guide.
