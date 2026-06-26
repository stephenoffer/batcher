# Troubleshooting

This page collects the errors you are most likely to hit and how to fix them
against the current API. Most stem from one of three things: referencing a column
that does not exist, passing a string where an expression is expected, or calling a
method that lives in a different place than you remembered.

```python
import batcher as bt

ds = bt.from_pydict({"x": [1, 2, 3], "y": [10, 20, 30]})
```

## Nothing happened when I called a transformation

A Dataset is lazy. Transformations build a plan and return a new Dataset; they do no
work. If you expected output, call a terminal operation.

```python
filtered = ds.filter(bt.col("x") > 1)  # builds a plan, runs nothing
print(filtered.to_pydict())  # this executes
# {'x': [2, 3], 'y': [20, 30]}
```

Terminal operations are `collect`, `to_pydict`, `to_pylist`, `count`,
`iter_batches`, `show`, and the write methods.

## Unknown column

Referencing a column that is not in the input raises `PlanError` with the available
names. Check for typos and confirm earlier steps did not drop or rename the column.

```python
try:
    ds.select("nope").to_pydict()
except Exception as exc:
    print(type(exc).__name__, "-", exc)
# PlanError - projection 'nope' references unknown column(s) ['nope']; available: ['x', 'y']
```

`.columns` shows the current schema names at any point in the chain.

```python
print(ds.with_columns(z=bt.col("x") + bt.col("y")).columns)
# ['x', 'y', 'z']
```

## filter needs an expression, not a string

`filter` takes an `Expr`. A raw string is not a predicate; build the condition with
{py:obj}`bt.col <batcher.col>`.

```python
try:
    ds.filter("x > 1")
except Exception as exc:
    print(type(exc).__name__, "-", exc)
# PlanError - filter() requires an expression, e.g. col('x') > 0

ok = ds.filter(bt.col("x") > 1)
print(ok.to_pydict())
# {'x': [2, 3], 'y': [20, 30]}
```

To filter with SQL syntax instead, use {py:obj}`bt.sql <batcher.sql>`.

## Aggregates are keyword arguments

`agg` takes named aggregates as keywords (`out_name=agg_expr`). Passing an
aggregate positionally fails, and there is no `.alias()` on an aggregate; the
keyword is the output name.

```python
try:
    ds.group_by("x").agg(bt.col("y").sum())
except Exception as exc:
    print(type(exc).__name__, "-", exc)
# TypeError - GroupBy.agg() takes 1 positional argument but 2 were given

ok = ds.group_by("x").agg(total=bt.col("y").sum())
print(ok.sort("x").to_pydict())
# {'x': [1, 2, 3], 'total': [10, 20, 30]}
```

## Boolean operators need parentheses

`&`, `|`, and `~` bind tighter than comparison, so combine compared expressions
with explicit parentheses on each side.

```python
ok = ds.filter((bt.col("x") > 1) & (bt.col("y") < 30))
print(ok.to_pydict())
# {'x': [2], 'y': [20]}
```

Writing `bt.col("x") > 1 & bt.col("y") < 30` parses as `bt.col("x") > (1 &
bt.col("y")) < 30` and will not do what you want.

## I cannot find a method I expected

The surface is deliberately small and some operations live in a specific place:

- There is no `ds.sql(...)` method. Use the top-level {py:obj}`bt.sql(query, table=ds) <batcher.sql>`.
- There is no dataset-level `.cast`, `.fill_null`, or `.drop_nulls`. Cast and
  fill nulls on an expression: `ds.with_columns(x=bt.col("x").cast("Float64"))`,
  `ds.with_columns(x=bt.col("x").fill_null(0))`.
- There is no `.unique()`; use `.distinct()`.
- `collect()` returns a pyarrow Table. To get a pandas DataFrame, call
  `.to_pandas()` on that Table, not on the Dataset.

```python
table = ds.collect()
print(type(table).__module__, type(table).__name__)
# pyarrow.lib Table
```

## A large query runs out of memory

Stateful operators hold state in memory by default. Pass `spill=True` to let
aggregation, join, and sort spill to disk under pressure, and `distributed=True`
with `num_workers=` to spread the work across machines. Both are off by default
because they add overhead that only pays off at scale.

```python
# docs: skip
out = ds.group_by("x").agg(total=bt.col("y").sum()).collect(spill=True)
```

## See also

- [Performance and memory](performance.md): caching, spill tuning, and reading a
  query plan.
- [Distributed fault tolerance](../architecture/fault-tolerance.md): diagnosing
  task, shuffle, and node failures.
- [Configuration options](../configuration/options.md): every tunable and its
  default.
