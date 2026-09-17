# Best practices

This page collects the habits that get the most out of Batcher. They all follow from one fact: Python builds and optimizes a plan, and Rust runs it over Arrow. The closer your code stays to describing a plan, the more the engine can do for you.

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
work. The plan executes only at a terminal operation such as `collect`,
{py:meth}`to_pydict <batcher.Dataset.to_pydict>`, or a write. Chain the whole transformation, then collect once. The
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

Use the {py:class}`Expr <batcher.plan.expr_ir.core.Expr>` API for every per-row computation. Expressions lower to Rust and run
vectorized over Arrow batches. Iterating rows in Python is the one thing the design
is built to avoid: it is slow and it crosses the control-plane boundary.

```python
# Good: a single expression, evaluated in Rust.
out = ds.with_columns(total=bt.col("price") * bt.col("qty"))
print(out.to_pydict()["total"])
# [10.0, 40.0, 90.0, 160.0, 250.0]
```

Don't pull data into Python to compute a column. If you reach for {py:meth}`to_pylist <batcher.Dataset.to_pylist>`
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
keeps unused columns from being read at all. {doc}`pushdown` lists which predicate shapes each
source accepts.

## Read the plan with explain

`explain()` prints the optimized plan and its row estimates. Use it to confirm a
filter landed near the scan or that a projection trimmed the columns.

```python
plan = ds.filter(bt.col("price") > 20).select("category").explain()
print(plan)
```

The filter sits directly above the scan, the projection above it, and `pushed[price > 20]` on the scan says the source applies the predicate itself. That's the shape you want. Each line carries the row estimate and its provenance: `exact` from the source, `default` from a heuristic, `learned` from a previous run.

```text
query plan (planned)                               3 operators
──────────────────────────────────────────────────────────────
OPERATOR                 ESTIMATE  NOTES
project                     est≈4  (default)
└─ filter  [price > 20]     est≈4  (default)
   └─ scan  [source 0]      est≈5  (exact)  pushed[price > 20]
```

{doc}`explain-plans` covers the rest of the output, including `explain(analyze=True)`.

## Leave distribution and spilling on their defaults

`collect()` decides both for you. With `distributed="auto"` it runs on Ray on a multi-node cluster and single-node otherwise, and spilling engages on its own under memory pressure. The explicit flags are overrides:

- `distributed=True`, with `num_workers=`, forces execution across Ray workers. The result is the same as single-node execution, because the same mergeable operators run in both modes.
- `spill=True` forces stateful operators such as aggregation, join, and sort onto the out-of-core path even without pressure.

```python
# docs: skip
out = (
    ds.group_by("category")
    .agg(revenue=bt.col("price").sum())
    .collect(distributed=True, num_workers=8, spill=True)
)
```

Don't force either one by default. Distribution adds scheduling and shuffle overhead that hurts small queries, and spilling trades memory for disk I/O. Both earn their cost only on a big job. To bound memory, set `memory.max_memory_bytes` to the real ceiling instead, as {doc}`performance` shows.

## See also

- {doc}`Performance and memory </user-guide/operate/tuning/performance>`: caching, spill, and the adaptive knobs.
- {doc}`explain-plans`: reading the plan and its measurements in full.
- {doc}`caching`: reusing a result without collecting it into Python.
- {doc}`Data quality </user-guide/trust/data-quality>`: validate and enforce a contract on inputs.
- {doc}`Distributed fault tolerance </architecture/fault-tolerance>`: how the engine
  recovers from node and task failures.
- {doc}`/cookbook/index`: runnable recipes grouped by domain, each asserting on its own output.
