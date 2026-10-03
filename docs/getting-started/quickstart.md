# Quickstart

This page builds a complete pipeline in five minutes. You'll filter and derive columns, aggregate, join, switch to SQL, read the plan and write Parquet. The data is five rows. The same code runs unchanged on a terabyte of Parquet or on a Ray cluster.

You need Batcher installed. If `import batcher` fails, follow {doc}`install/packages-and-extras` first.

## Build a dataset

Import Batcher as `bt`. {py:func}`from_pydict <batcher.from_pydict>` builds an in-memory dataset from a dictionary of columns.

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

The schema is known before anything runs:

```python
print(ds.schema)
# id: int64
# name: string
# category: string
# price: double
# qty: int64
```

A {py:class}`Dataset <batcher.Dataset>` is *lazy*. Each step returns a new plan, and nothing runs until a terminal call such as `to_pydict` or `collect`. Because Batcher sees the whole plan first, it can push filters into the scan, drop columns you never use and pick join strategies for you. {doc}`concepts/lazy` has the details.

![Two stacked panels. In the top panel, labeled lazy, a chain of calls (read.parquet, filter, with_columns, group_by then agg) describes a query plan made of scan, filter, project, and aggregate, and nothing runs because each step returns a new Dataset. An amber arrow labeled collect, to_pydict, or write leads into the bottom panel, where the plan runs once: the optimizer pushes filters and prunes columns, hands the plan to the Rust engine, which works over whole Arrow batches, and the rows come back as a Table, a dict, or files.](/_static/diagrams/quickstart_lazy_plan.svg)

## Filter rows and derive columns

Build filters from {py:obj}`bt.col(...) <batcher.col>` and combine them with `&` (and), `|` (or) and `~` (not).

```python
filtered = ds.filter((bt.col("price") >= 30.0) & (bt.col("category") == "a"))
print(filtered.to_pydict())
# {'id': [3, 5], 'name': ['cy', 'eve'], 'category': ['a', 'a'], 'price': [30.0, 50.0], 'qty': [3, 5]}
```

Membership tests read like Python:

```python
print(ds.filter(bt.col("name").is_in(["ann", "eve"])).select("id").to_pydict())
# {'id': [1, 5]}
```

`select` picks or derives the full output. `with_columns` adds or replaces columns and keeps the rest. Either way, a derived column is a keyword argument.

```python
projected = ds.select("name", total=bt.col("price") * bt.col("qty"))
print(projected.to_pydict())
# {'name': ['ann', 'bob', 'cy', 'dan', 'eve'], 'total': [10.0, 40.0, 90.0, 160.0, 250.0]}

enriched = ds.with_columns(total=bt.col("price") * bt.col("qty"))
print(enriched.columns)
# ['id', 'name', 'category', 'price', 'qty', 'total']
```

Conditional columns use `when/then/otherwise`:

```python
tier = bt.when(bt.col("price") > 25).then(bt.lit("high")).otherwise(bt.lit("low"))
print(ds.select("name", tier=tier).to_pydict())
# {'name': ['ann', 'bob', 'cy', 'dan', 'eve'], 'tier': ['low', 'low', 'high', 'high', 'high']}
```

Strings and dates have typed accessors such as `.str`, and so does nested data:

```python
print(ds.select(upper=bt.col("name").str.upper(), chars=bt.col("name").str.len_chars()).to_pydict())
# {'upper': ['ANN', 'BOB', 'CY', 'DAN', 'EVE'], 'chars': [3, 3, 2, 3, 3]}
```

Nulls have their own verbs:

```python
gaps = bt.from_pydict({"v": [1, None, 3]})
print(gaps.select(v=bt.col("v").fill_null(0)).to_pydict())
# {'v': [1, 0, 3]}
```

You never write a loop. Every expression runs in Rust over whole Arrow batches. More in {doc}`/user-guide/transform/rows/filtering`, {doc}`/user-guide/transform/rows/transformations`, and {doc}`/user-guide/transform/columns/expressions`.

## Aggregate

Group with `group_by`, then name each aggregate as a keyword in `agg`. {py:obj}`bt.count() <batcher.count>` is `COUNT(*)`.

```python
summary = (
    enriched.group_by("category")
    .agg(revenue=bt.col("total").sum(), orders=bt.count())
    .sort("revenue", descending=True)
)
print(summary.to_pydict())
# {'category': ['a', 'b'], 'revenue': [350.0, 200.0], 'orders': [3, 2]}
```

Window functions aggregate without collapsing rows. This one keeps a running total per category:

```python
running = bt.col("price").sum().over(partition_by="category", order_by="id")
print(ds.select("id", "category", running=running).sort("id").to_pydict())
# {'id': [1, 2, 3, 4, 5], 'category': ['a', 'b', 'a', 'b', 'a'], 'running': [10.0, 20.0, 40.0, 60.0, 90.0]}
```

Top-N is a sort and a limit:

```python
print(ds.sort("price", descending=True).limit(2).select("name", "price").to_pydict())
# {'name': ['eve', 'dan'], 'price': [50.0, 40.0]}
```

See {doc}`/user-guide/analyze/aggregations` and {doc}`/user-guide/analyze/window-functions`.

## Join

Join two datasets on a shared key. The default is an inner join.

```python
regions = bt.from_pydict({"category": ["a", "b"], "region": ["west", "east"]})
joined = ds.join(regions, on="category").select("id", "category", "region").sort("id")
print(joined.to_pydict())
# {'id': [1, 2, 3, 4, 5], 'category': ['a', 'b', 'a', 'b', 'a'], 'region': ['west', 'east', 'west', 'east', 'west']}
```

Left, outer, semi, anti and as-of joins all take the same shape ({doc}`/user-guide/analyze/joins`).

## Switch to SQL whenever you like

SQL builds the same plan the DataFrame verbs build, so one pipeline can use both. Pass datasets to {py:func}`bt.sql <batcher.sql>` by keyword, then query them by that name.

```python
revenue = bt.sql(
    "SELECT category, SUM(price * qty) AS revenue FROM sales GROUP BY category",
    sales=ds,
)
print(revenue.sort("category").to_pydict())
# {'category': ['a', 'b'], 'revenue': [350.0, 200.0]}
```

{doc}`/user-guide/analyze/sql` lists the supported SQL. To translate a query into DataFrame verbs step by step, see {doc}`tutorials/foundations/sql-to-dataframe`.

## Run the plan and inspect it

Terminal operations run the plan. You get columns from {py:meth}`to_pydict <batcher.Dataset.to_pydict>` and rows from {py:meth}`to_pylist <batcher.Dataset.to_pylist>`. {py:meth}`count <batcher.Dataset.count>` gives the row count, and {py:meth}`collect <batcher.Dataset.collect>` hands back a `pyarrow.Table`.

```python
print(ds.count())
# 5

table = ds.select("name", "price").collect()
print(table.num_rows)
# 5
```

To see the optimized plan without running it, call {py:meth}`explain <batcher.Dataset.explain>`. Look for `pushed[price > 25.0]` on the scan line. The filter moved into the read, so a Parquet reader skips any row group whose statistics rule it out.

```python
print(ds.filter(bt.col("price") > 25.0).explain())
```

{doc}`/user-guide/operate/tuning/explain-plans` walks through the output. Some questions skip the scan entirely: a `count()` on Parquet reads only file metadata ({doc}`/user-guide/analyze/metadata-shortcuts`).

## Read and write files

Readers and writers share one API:

```python
ds.write.parquet("sales.parquet")
back = bt.read.parquet("sales.parquet")
print(back.count())
# 5
```

Object stores work the same way:

```python
# docs: skip
events = bt.read("s3://<your-bucket>/events/*.parquet")
events.filter(bt.col("status") == "active").write.parquet("s3://<your-bucket>/active.parquet")
```

Replace `<your-bucket>` with your bucket and install the `cloud` extra. Formats and save modes are in {doc}`/user-guide/moving-data/reading-data` and {doc}`/user-guide/moving-data/writing-data`. Credentials are in {doc}`/user-guide/moving-data/cloud-storage`.

## Scale it out

You've now written the shape every Batcher pipeline has. Read, chain lazy steps, collect once. Scaling out is one more argument on the terminal call:

```python
# docs: skip
summary.collect(distributed=True, num_workers=8)
```

Next, {doc}`tutorials/foundations/first-pipeline` builds the same shape on a realistic dataset. The {doc}`tour` runs one example of everything Batcher does, and {doc}`concepts/index` gives lazy evaluation, expressions, scaling and the adaptive loop one short page each. If you already know pandas, Polars, Spark, DuckDB or Daft, start from {doc}`migration/index` instead.

For reference, {doc}`/user-guide/index` covers every operator with runnable examples, and {doc}`/api/reference` is a cheat sheet to keep open. When a first query misbehaves, read {doc}`/user-guide/operate/running/troubleshooting`.
