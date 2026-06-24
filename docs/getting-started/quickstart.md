# Quickstart

This page walks through a complete pipeline: build a dataset, transform it,
aggregate it, and read the result. The examples use small in-memory data so they
run anywhere; the same API applies at any scale.

## Import and build a dataset

The conventional alias is `bt`. Build an in-memory dataset from a column-oriented
dictionary with `from_pydict`.

```python
import batcher as bt

ds = bt.from_pydict(
    {
        "id": [1, 2, 3, 4, 5],
        "name": ["ann", "bob", "cy", "dan", "eve"],
        "category": ["a", "b", "a", "b", "a"],
        "price": [10.0, 20.0, 30.0, 40.0, 50.0],
        "qty": [1, 2, 3, 4, 5],
    }
)

print(ds.columns)
# ['id', 'name', 'category', 'price', 'qty']
```

A `Dataset` is lazy: each operation returns a new `Dataset` describing a plan, and
no work runs until a terminal operation such as `to_pydict` or `collect`.

## Filter rows

Filters are expressions built from {py:obj}`bt.col(...) <batcher.col>`. Combine conditions with `&`
(and), `|` (or), and `~` (not).

```python
filtered = ds.filter(bt.col("price") >= 30.0)
print(filtered.to_pydict())
# {'category': ['a', 'b', 'a'], 'id': [3, 4, 5], 'name': ['cy', 'dan', 'eve'], 'price': [30.0, 40.0, 50.0], 'qty': [3, 4, 5]}
```

## Select and transform columns

`select` chooses or derives the full output. `with_columns` adds or replaces
columns and keeps the rest. Derived columns are passed as keyword arguments.

```python
projected = ds.select("name", total=bt.col("price") * bt.col("qty"))
print(projected.to_pydict())
# {'name': ['ann', 'bob', 'cy', 'dan', 'eve'], 'total': [10.0, 40.0, 90.0, 160.0, 250.0]}

enriched = ds.with_columns(total=bt.col("price") * bt.col("qty"))
print(enriched.columns)
# ['id', 'name', 'category', 'price', 'qty', 'total']
```

## Aggregate

Group with `group_by` and finalize with `agg`. Each aggregate is a keyword whose
value is an aggregate expression; {py:obj}`bt.count() <batcher.count>` is `COUNT(*)`.

```python
summary = (
    ds.with_columns(total=bt.col("price") * bt.col("qty"))
    .group_by("category")
    .agg(revenue=bt.col("total").sum(), orders=bt.count())
    .sort("revenue", descending=True)
)
print(summary.to_pydict())
# {'category': ['a', 'b'], 'revenue': [350.0, 200.0], 'orders': [3, 2]}
```

## Join

Join two datasets on a shared key. The default is an inner join.

```python
dim = bt.from_pydict({"category": ["a", "b"], "region": ["west", "east"]})
joined = ds.join(dim, on="category").select("id", "category", "region").sort("id")
print(joined.to_pydict())
# {'id': [1, 2, 3, 4, 5], 'category': ['a', 'b', 'a', 'b', 'a'], 'region': ['west', 'east', 'west', 'east', 'west']}
```

## Execute and inspect

Terminal operations run the plan. `to_pydict` returns columns, `to_pylist`
returns rows, `count` returns the row count, and `collect` returns a
`pyarrow.Table`.

```python
print(ds.count())
# 5

table = ds.select("name", "price").collect()
print(table.num_rows)
# 5
```

`explain` shows the optimized plan without executing it:

```python
print(ds.filter(bt.col("price") > 25.0).explain())
```

## Reading and writing files

File readers and writers use the same API; only the source or sink changes. These
need real files, so they are shown but not run here.

```python
# docs: skip
ds = bt.read("s3://bucket/events.parquet")
ds.filter(bt.col("status") == "active").write.parquet("output/active.parquet")
```

## Next steps

- [Core concepts](concepts.md): the lazy, immutable execution model.
- [Transformations](../user-guide/transformations.md) and
  [Aggregations](../user-guide/aggregations.md) cover the operators in depth.
