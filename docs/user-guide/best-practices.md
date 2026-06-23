# Best practices

These patterns get the most out of the engine. They follow from one fact: Python
builds and optimizes a plan, and Rust runs it over Arrow. The closer your code
stays to describing a plan, the more the engine can do for you.

```python
import batcher as bt

ds = bt.from_pydict(
    {
        "category": ["a", "b", "a", "b", "a"],
        "price": [10.0, 20.0, 30.0, 40.0, 50.0],
        "qty": [1, 2, 3, 4, 5],
    }
)
```

## Build one lazy chain, collect once

A Dataset is lazy and immutable. Each operation returns a new Dataset and runs no
work; the plan executes only at a terminal operation such as `collect`,
`to_pydict`, or a write. Chain the whole transformation, then collect once. The
optimizer sees the entire pipeline and can reorder and fuse it.

```python
out = (
    ds.filter(bt.col("price") >= 20)
    .with_columns(total=bt.col("price") * bt.col("qty"))
    .group_by("category")
    .agg(revenue=bt.col("total").sum())
    .sort("category")
)
print(out.to_pydict())
# {'category': ['a', 'b'], 'revenue': [340.0, 200.0]}
```

Avoid collecting in the middle of a pipeline. Materializing intermediate results
forces work the optimizer could have skipped and pulls rows into Python.

## Express column work as expressions, not Python loops

Use the `Expr` API for every per-row computation. Expressions lower to Rust and run
vectorized over Arrow batches. Iterating rows in Python is the one thing the design
is built to avoid: it is slow and it crosses the control-plane boundary.

```python
# Good: a single expression, evaluated in Rust.
out = ds.with_columns(total=bt.col("price") * bt.col("qty"))
print(out.to_pydict()["total"])
# [10.0, 40.0, 90.0, 160.0, 250.0]
```

Do not pull data into Python to compute a column. If you reach for `to_pylist`
inside a loop to build a new field, rewrite it as an expression instead. When a
computation genuinely needs Python, use `map_batches`, which hands you a whole
Arrow batch rather than one row at a time.

## Write pushdown-friendly filters

Filter early and filter on raw columns. A predicate over a stored column can be
pushed down to the scan, so the engine reads fewer rows or skips files entirely.
Wrapping a column in a function before comparing it can block that pushdown.

```python
# Pushdown-friendly: the comparison is on the column itself.
early = ds.filter(bt.col("category") == "a").select("category", "price")
print(early.to_pydict())
# {'category': ['a', 'a', 'a'], 'price': [10.0, 30.0, 50.0]}
```

Select only the columns you need as early as possible. Projection pushdown then
keeps unused columns from being read at all.

## Read the plan with explain

`explain()` prints the optimized plan and its row estimates. Use it to confirm a
filter landed near the scan or that a projection trimmed the columns.

```python
plan = ds.filter(bt.col("price") > 20).select("category").explain()
print(plan)
# Project  (≈... rows, default)
#   Filter  (≈... rows, default)
#     Scan  (≈5 rows, exact)
```

## Use distributed and spill deliberately

`collect` runs single-node and in-memory by default, which is the fastest path for
data that fits. Reach for the flags when the workload calls for them:

- `distributed=True` (with `num_workers=`) spreads execution across Ray workers.
  Use it when one machine cannot hold or process the data in reasonable time. The
  result is identical to single-node execution because the same mergeable operators
  run in both modes.
- `spill=True` lets stateful operators (aggregation, join, sort) spill to disk
  under memory pressure instead of failing. Use it when an in-memory run risks
  running out of memory.

```python
# docs: skip
out = (
    ds.group_by("category")
    .agg(revenue=bt.col("price").sum())
    .collect(distributed=True, num_workers=8, spill=True)
)
```

Do not turn these on by default. Distribution adds scheduling and shuffle overhead
that hurts small queries, and spill trades memory for disk I/O. Both earn their cost
only at scale.
