# Your first pipeline

Build a complete pipeline from an in-memory dataset: a five-row sales table, a derived
`total` column, a per-category revenue rollup, and a sorted result. Everything here runs
as written.

You need `pip install batcher-engine`. No cluster, no GPU, no files on disk. The closing
block swaps the in-memory source for a Parquet path and changes nothing else, which is
the point of the whole tutorial. It needs a real file, so it's shown rather than run.

## Build a dataset

A {py:class}`Dataset <batcher.Dataset>` is a lazy, immutable handle to a query plan. {py:obj}`bt.from_pydict <batcher.from_pydict>` builds one
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

Column work is expressed with {py:class}`Expr <batcher.plan.expr_ir.core.Expr>`. {py:meth}`with_columns <batcher.Dataset.with_columns>` adds or replaces columns and
keeps the rest. The arithmetic runs in the Rust data plane, not in Python.

```python
priced = ds.with_columns(total=bt.col("price") * bt.col("qty"))
print(priced.to_pydict())
# {'category': ['a', 'b', 'a', 'b', 'a'], 'price': [10.0, 20.0, 30.0, 40.0, 50.0], 'qty': [1, 2, 3, 4, 5], 'total': [10.0, 40.0, 90.0, 160.0, 250.0]}
```

## Group and aggregate

`group_by(*keys)` returns a {py:class}`GroupBy <batcher.GroupBy>`; finalize it with `.agg(**named_aggs)`.
Aggregates are passed as keyword arguments where the name becomes the output column.

::::{tab-set}
:::{tab-item} DataFrame
```python
summary = priced.group_by("category").agg(
    revenue=bt.col("total").sum(),
    orders=bt.count(),
)
print(summary.to_pydict())
# {'category': ['a', 'b'], 'revenue': [350.0, 200.0], 'orders': [3, 2]}
```
:::

:::{tab-item} SQL
```python
same = bt.sql(
    "SELECT category, SUM(total) AS revenue, COUNT(*) AS orders FROM t GROUP BY category",
    t=priced,
)
print(same.sort("category").to_pydict())
# {'category': ['a', 'b'], 'revenue': [350.0, 200.0], 'orders': [3, 2]}
```
:::
::::

Both spellings build the same plan, run through the same optimizer, and execute on the same
Rust engine. Pick whichever reads better.

## Sort and collect

`sort` orders rows; `descending=True` reverses it. A terminal operation executes the
plan: `to_pydict` hands back a column dict, `collect` a pyarrow `Table`.

```python
ranked = summary.sort("revenue", descending=True)
print(ranked.to_pydict())
# {'category': ['a', 'b'], 'revenue': [350.0, 200.0], 'orders': [3, 2]}

table = ranked.collect()
print(table.num_rows)
# 2
```

Each call you chained added one node to the plan, and only the terminal call ran it:

![Code on the left beside the plan node each line adds on the right. bt.from_pydict adds a scan of source 0. with_columns adds a project that computes total. group_by then agg adds an aggregate by category with sum and count_star. sort by revenue, descending, adds a sort. Data flows down that column from scan to project to aggregate to sort. The last line, to_pydict, adds no node: it executes the plan and returns a column dict. explain prints the same tree upside down, with sort on top and scan at the bottom.](/_static/diagrams/first_pipeline_plan.svg)

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

`explain()` gives you the optimized plan as text without executing it. Use it to check
what the optimizer did.

```python
print(isinstance(result.explain(), str))
# True
```

:::{dropdown} What a plan looks like, and how to read one
The plan renders as a tree with the scan at the bottom and the terminal operator at the top. Each line carries the operator's estimated row count and where the estimate came from. This is the plan for `result`:

```text
query plan (planned)                                   4 operators
──────────────────────────────────────────────────────────────────
OPERATOR                                       ESTIMATE  NOTES
sort  [revenue]                                   est≈1  (default)
└─ aggregate  [by category · sum, count_star]     est≈1  (default)
   └─ project                                     est≈5  (exact)
      └─ scan  [source 0]                         est≈5  (exact)
```

`(exact)` means the optimizer knows the count, because an in-memory source has five rows. `(default)` is a prior, used because it has no statistics on how many distinct categories there are. A filter, when there is one, shows up pushed into the scan's notes. {doc}`Optimizing a slow query </getting-started/tutorials/foundations/optimizing-a-slow-query>` teaches you to read a plan properly.
:::

## The same pipeline over files

Only the source changes when the data lives in files or object storage. Every transform
and terminal op below it is identical. This block needs a real file, so it is shown but
not run.

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

:::{warning}
The one thing that trips people up on their first pipeline is expecting a transform to *do*
something. It does not. `with_columns`, `filter`, `group_by`, and `sort` all return a new
`Dataset` and run nothing at all. The work happens at the terminal op ({py:meth}`to_pydict <batcher.Dataset.to_pydict>`,
`collect`, `count`, `write`). If your timing shows a transform taking no time, that is
because it took no time.
:::

:::{tip}
Column work belongs in an expression, not in a Python callback. `bt.col("price") * bt.col("qty")` runs in Rust, and the optimizer can see through it. The same arithmetic in a `map_batches` blocks predicate pushdown and costs throughput on every batch. The {doc}`slow query tutorial </getting-started/tutorials/foundations/optimizing-a-slow-query>` shows exactly that.
:::

## Where to go next

Pick by what you are building:

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`code;1.1em` From SQL to DataFrames
:link: /getting-started/tutorials/foundations/sql-to-dataframe
:link-type: doc
The same query both ways, and the proof they compile to one plan.
:::

:::{grid-item-card} {octicon}`meter;1.1em` Optimizing a slow query
:link: /getting-started/tutorials/foundations/optimizing-a-slow-query
:link-type: doc
Read the plan, measure the operators, fix the one that costs.
:::

:::{grid-item-card} {octicon}`beaker;1.1em` Synthetic data
:link: /getting-started/tutorials/pipelines/synthetic-data-generation
:link-type: doc
Build a larger input to try this at a real size.
:::

:::{grid-item-card} {octicon}`zap;1.1em` Batch inference
:link: /getting-started/tutorials/ml/batch-inference
:link-type: doc
Run a model over Arrow batches with the `.ml` accessor.
:::
::::

## See also

- {doc}`Expressions </user-guide/transform/columns/expressions>`: the column language, in full.
- {doc}`Aggregations </user-guide/analyze/aggregations>`: every aggregate, and `GroupBy`.
- {doc}`Reading data </user-guide/moving-data/reading-data>`: the sources the last block reached for.
- {doc}`Lazy evaluation </getting-started/concepts/lazy>`: why nothing ran until the terminal
  op.
- {doc}`Dataset API </api/relational/dataset>`: the reference for every method on this page.
