# Quickstart

This page builds a complete pipeline, from an in-memory dataset through to a file on
disk. The data is small so every example runs anywhere. The API is the one you would
point at a terabyte of Parquet.

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

A `Dataset` is lazy. Each operation returns a new `Dataset` describing a plan, and no
work runs until a terminal operation such as `to_pydict` or `collect`. That one idea
explains most of the API, and {doc}`concepts/lazy` unpacks it.

## Filter rows

Filters are expressions built from {py:obj}`bt.col(...) <batcher.col>`. Combine conditions with `&`
(and), `|` (or), and `~` (not).

```python
filtered = ds.filter(bt.col("price") >= 30.0)
print(filtered.to_pydict())
# {'id': [3, 4, 5], 'name': ['cy', 'dan', 'eve'], 'category': ['a', 'b', 'a'], 'price': [30.0, 40.0, 50.0], 'qty': [3, 4, 5]}
```

Null handling, `is_in`, and sampling are in {doc}`/user-guide/transform/rows/filtering`.

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

Column work is expressed rather than looped, and the expression language has typed
accessors for strings, dates, lists, and structs. See {doc}`/user-guide/transform/rows/transformations`
and {doc}`/user-guide/transform/columns/expressions`.

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

{doc}`/user-guide/analyze/aggregations` covers the full aggregate list, pivots, and rollups, and
{doc}`/user-guide/analyze/window-functions` covers ranking and running totals, which aggregate
without collapsing rows.

## Join

Join two datasets on a shared key. The default is an inner join.

```python
dim = bt.from_pydict({"category": ["a", "b"], "region": ["west", "east"]})
joined = ds.join(dim, on="category").select("id", "category", "region").sort("id")
print(joined.to_pydict())
# {'id': [1, 2, 3, 4, 5], 'category': ['a', 'b', 'a', 'b', 'a'], 'region': ['west', 'east', 'west', 'east', 'west']}
```

Left, outer, semi, anti, and as-of joins take the same shape. See {doc}`/user-guide/analyze/joins`.

## Write SQL instead, if you prefer

The same plan comes out either way, so you can mix the two spellings in one pipeline.

```python
revenue = bt.sql(
    "SELECT category, SUM(price * qty) AS revenue FROM sales GROUP BY category",
    sales=ds,
)
print(revenue.sort("category").to_pydict())
# {'category': ['a', 'b'], 'revenue': [350.0, 200.0]}
```

{doc}`/user-guide/analyze/sql` lists the supported surface, and {doc}`/tutorials/foundations/sql-to-dataframe`
translates a SQL query into DataFrame verbs step by step.

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

`explain` shows the optimized plan without executing it, which is how you check that a
filter reached the scan instead of running after it:

```python
print(ds.filter(bt.col("price") > 25.0).explain())
```

Reading a plan is a skill worth ten minutes: {doc}`/user-guide/operate/tuning/explain-plans`.

Some questions never need a scan at all. A `count()` on a Parquet source is answered from
file metadata, and {doc}`/user-guide/analyze/metadata-shortcuts` lists the rest.

## Read and write files

Readers and writers use the same API. Only the source or the sink changes. The
snippet below needs real files, so it's shown rather than run.

```python
# docs: skip
ds = bt.read("s3://bucket/events.parquet")
ds.filter(bt.col("status") == "active").write.parquet("output/active.parquet")
```

Local paths work the same way, and this one runs:

```python
ds.write.parquet("sales.parquet")
back = bt.read.parquet("sales.parquet")
print(back.count())
# 5
```

Every format, glob, and credential path is in {doc}`/user-guide/moving-data/reading-data` and
{doc}`/user-guide/moving-data/writing-data`, and object stores are in
{doc}`/user-guide/moving-data/cloud-storage`.

## What you have now

You built a dataset, filtered and derived columns, aggregated, joined, ran the same query
as SQL, inspected a plan, and round-tripped a file. That is the whole shape of a Batcher
pipeline: **read, chain lazy verbs, collect once at the end.** Scaling it up changes the
source and the machine, not the code.

## See also

- {doc}`/tutorials/foundations/first-pipeline`: the same shape again, on a realistic dataset.
- {doc}`concepts/index`: lazy evaluation, expressions, scaling, and the adaptive loop, one short page each.
- {doc}`../user-guide/index`: every operator, with runnable examples.
- {doc}`/getting-started/migration/index`: the verb-by-verb table if you already know pandas, Polars, Spark, or SQL.
- {doc}`../api/reference`: a cheat sheet to keep open while you work.
- {doc}`/user-guide/operate/running/troubleshooting`: what to read when the first query misbehaves.
