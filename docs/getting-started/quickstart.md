# Quickstart

In the next five minutes you'll build a complete pipeline: filter rows, derive columns, aggregate, join, switch to SQL, inspect the plan, and write Parquet. The data is five rows so every example runs anywhere. The API is the one you'd point at a terabyte of Parquet or a Ray cluster, and none of the code below would change.

You need Batcher installed. If `import batcher` fails, follow {doc}`install/packages-and-extras` first.

## Build a dataset

The conventional alias is `bt`. {py:func}`from_pydict <batcher.from_pydict>` builds an in-memory dataset from a dictionary of columns.

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

A {py:class}`Dataset <batcher.Dataset>` is *lazy*. Each operation returns a new `Dataset` that describes a plan, and no work runs until you ask for a result with a call such as `to_pydict` or `collect`. Because Batcher sees the whole plan before it runs anything, it can push filters into the scan, prune unused columns, and pick join strategies for you. {doc}`concepts/lazy` explains the model in a page.

Every example below has two halves. The steps you chain only describe a plan, and one terminal call optimizes that plan and runs it:

![Two stacked panels. In the top panel, labeled lazy, a chain of calls (read.parquet, filter, with_columns, group_by then agg) describes a query plan made of scan, filter, project, and aggregate, and nothing runs because each step returns a new Dataset. An amber arrow labeled collect, to_pydict, or write leads into the bottom panel, where the plan runs once: the optimizer pushes filters and prunes columns, hands the plan to the Rust engine, which works over whole Arrow batches, and the rows come back as a Table, a dict, or files.](/_static/diagrams/quickstart_lazy_plan.svg)

## Filter rows and derive columns

Filters are expressions built from {py:obj}`bt.col(...) <batcher.col>`. Combine conditions with `&` for and, `|` for or, and `~` for not.

```python
filtered = ds.filter((bt.col("price") >= 30.0) & (bt.col("category") == "a"))
print(filtered.to_pydict())
# {'id': [3, 5], 'name': ['cy', 'eve'], 'category': ['a', 'a'], 'price': [30.0, 50.0], 'qty': [3, 5]}
```

`select` chooses or derives the full output. `with_columns` adds or replaces columns and keeps the rest. Derived columns are keyword arguments.

```python
projected = ds.select("name", total=bt.col("price") * bt.col("qty"))
print(projected.to_pydict())
# {'name': ['ann', 'bob', 'cy', 'dan', 'eve'], 'total': [10.0, 40.0, 90.0, 160.0, 250.0]}

enriched = ds.with_columns(total=bt.col("price") * bt.col("qty"))
print(enriched.columns)
# ['id', 'name', 'category', 'price', 'qty', 'total']
```

You never write a loop. Each expression is handed to the Rust engine, which evaluates it over whole Arrow batches, and typed accessors such as `.str`, `.dt`, and `.list` cover strings, dates, and nested data. {doc}`/user-guide/transform/rows/filtering` covers nulls, {py:meth}`is_in <batcher.plan.expr_ir.core.Expr.is_in>`, and sampling. {doc}`/user-guide/transform/rows/transformations` and {doc}`/user-guide/transform/columns/expressions` cover the rest of the column vocabulary.

## Aggregate

Group with `group_by` and finish with `agg`. Each aggregate is a keyword whose value is an aggregate expression, and {py:obj}`bt.count() <batcher.count>` is `COUNT(*)`.

```python
summary = (
    enriched.group_by("category")
    .agg(revenue=bt.col("total").sum(), orders=bt.count())
    .sort("revenue", descending=True)
)
print(summary.to_pydict())
# {'category': ['a', 'b'], 'revenue': [350.0, 200.0], 'orders': [3, 2]}
```

{doc}`/user-guide/analyze/aggregations` lists every aggregate, plus pivots and rollups. {doc}`/user-guide/analyze/window-functions` covers ranking and running totals, which aggregate without collapsing rows.

## Join

Join two datasets on a shared key. The default is an inner join.

```python
regions = bt.from_pydict({"category": ["a", "b"], "region": ["west", "east"]})
joined = ds.join(regions, on="category").select("id", "category", "region").sort("id")
print(joined.to_pydict())
# {'id': [1, 2, 3, 4, 5], 'category': ['a', 'b', 'a', 'b', 'a'], 'region': ['west', 'east', 'west', 'east', 'west']}
```

Left, outer, semi, anti, and as-of joins take the same shape. See {doc}`/user-guide/analyze/joins`.

## Switch to SQL whenever you like

SQL and DataFrame code build the same plan, so you can mix them in one pipeline. Pass datasets to {py:func}`bt.sql <batcher.sql>` by keyword and query them by that name.

```python
revenue = bt.sql(
    "SELECT category, SUM(price * qty) AS revenue FROM sales GROUP BY category",
    sales=ds,
)
print(revenue.sort("category").to_pydict())
# {'category': ['a', 'b'], 'revenue': [350.0, 200.0]}
```

{doc}`/user-guide/analyze/sql` lists the supported SQL, and {doc}`tutorials/foundations/sql-to-dataframe` translates a query into DataFrame verbs step by step.

## Run the plan and inspect it

Terminal operations run the plan. {py:meth}`to_pydict <batcher.Dataset.to_pydict>` returns columns, {py:meth}`to_pylist <batcher.Dataset.to_pylist>` returns rows, {py:meth}`count <batcher.Dataset.count>` returns the row count, and {py:meth}`collect <batcher.Dataset.collect>` returns a `pyarrow.Table`.

```python
print(ds.count())
# 5

table = ds.select("name", "price").collect()
print(table.num_rows)
# 5
```

{py:meth}`explain <batcher.Dataset.explain>` shows the optimized plan without running it. Look for `pushed[price > 25.0]` on the scan line: the optimizer moved the filter into the read, so on a large Parquet file the reader skips every row group whose statistics rule the predicate out instead of decoding it.

```python
print(ds.filter(bt.col("price") > 25.0).explain())
```

{doc}`/user-guide/operate/tuning/explain-plans` shows how to read every column.

Some questions never need a scan at all. A `count()` on a Parquet source is answered from file metadata, and {doc}`/user-guide/analyze/metadata-shortcuts` lists the other shortcuts.

## Read and write files

Readers and writers share one API, so only the source or the sink changes. This local round trip runs as written:

```python
ds.write.parquet("sales.parquet")
back = bt.read.parquet("sales.parquet")
print(back.count())
# 5
```

Object stores work the same way. The next snippet needs a real bucket, so it's shown rather than run:

```python
# docs: skip
events = bt.read("s3://<your-bucket>/events/*.parquet")
events.filter(bt.col("status") == "active").write.parquet("s3://<your-bucket>/active.parquet")
```

Replace `<your-bucket>` with a bucket you can read and write, and install the `cloud` extra. {doc}`/user-guide/moving-data/reading-data` and {doc}`/user-guide/moving-data/writing-data` cover every format, glob, and save mode, and {doc}`/user-guide/moving-data/cloud-storage` covers credentials.

## Next steps

Every Batcher pipeline has the shape you just wrote: read, chain lazy steps, and collect once at the end. Growing it changes the source and the machine, not the code. Continue with {doc}`tutorials/foundations/first-pipeline`, which builds the same shape on a realistic dataset, or read the {doc}`concepts/index` to learn why the engine behaves the way it does. To see how far the same model stretches, {doc}`tour` runs one small example of each thing Batcher does, from streaming to media to models, on a single page.

## See also

- {doc}`tour`: one runnable example per capability, on one page.
- {doc}`concepts/index`: lazy evaluation, expressions, scaling, and the adaptive loop, one short page each.
- {doc}`migration/index`: the verb-by-verb mapping if you already know pandas, Polars, Spark, DuckDB, or Daft.
- {doc}`/user-guide/index`: every operator, with runnable examples.
- {doc}`/api/reference`: a cheat sheet to keep open while you work.
- {doc}`/user-guide/operate/running/troubleshooting`: what to read when the first query misbehaves.
