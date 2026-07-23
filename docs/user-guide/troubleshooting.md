# Troubleshooting

The errors below are the ones you are most likely to hit. Nearly all of them come
from the same few causes: a column that does not exist, a string passed where an
expression belongs, or a method that lives somewhere other than where you remembered
it.

```python
import batcher as bt

ds = bt.from_pydict({"x": [1, 2, 3], "y": [10, 20, 30]})
```

## Nothing happened when I called a transformation

A Dataset is lazy. Transformations build a plan and return a new Dataset without doing
any work. If you expected output, call a terminal operation.

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

## The keyword to `agg` is the output name

A positional aggregate is *self-naming*: it keeps the source column's name. Pass a keyword
when you want to choose the output name, which is usually what you want, because the
self-named column shadows the input it was computed from.

```python
selfnamed = ds.group_by("x").agg(bt.col("y").sum())
print(selfnamed.sort("x").to_pydict())
# {'x': [1, 2, 3], 'y': [10, 20, 30]}

named = ds.group_by("x").agg(total=bt.col("y").sum())
print(named.sort("x").to_pydict())
# {'x': [1, 2, 3], 'total': [10, 20, 30]}
```

There is no `.alias()` on an aggregate. The keyword is the output name.

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

The surface is deliberately small, and a few operations sit somewhere other than where
you might reach first:

- `collect()` returns a pyarrow Table. To get a pandas DataFrame, call `.to_pandas()` on
  that Table, not on the Dataset.
- `distinct()` and `unique()` are the same operator under two names, so use whichever
  spelling your background makes natural.
- `ds.sql(query)` binds the current dataset as the table `self`. The top-level
  {py:obj}`bt.sql <batcher.sql>` is what you want when a query names more than one table.
- `ds.cast`, `ds.fill_null`, and `ds.drop_nulls` work over whole columns. When you want
  the same thing on one derived value, the expression methods do it:
  `ds.with_columns(x=bt.col("x").cast("float64"))`.

```python
table = ds.collect()
print(type(table).__module__, type(table).__name__)
# pyarrow.lib Table
```

## Catching errors by type

Every Batcher failure subclasses `bt.BatcherError`, so one `except` catches them all
without importing anything internal:

```python
try:
    ds.select("nope").to_pydict()
except bt.BatcherError as exc:
    print(type(exc).__name__, "-", exc)
# PlanError - projection 'nope' references unknown column(s) ['nope']; available: ['x', 'y']
```

Catch a narrower type when you want to react differently. The catchable types are all
reachable as `bt.<Name>`:

| Type | Catch it for | Also a |
|------|--------------|--------|
| `bt.PlanError` | an invalid plan or schema, raised eagerly at build time | `ValueError` |
| `bt.ColumnNotFoundError` | a reference to a column that isn't there (carries `.column`) | `KeyError` |
| `bt.ConfigError` | an out-of-range or inconsistent configuration value | `ValueError` |
| `bt.MissingDependencyError` | an optional extra that isn't installed (carries `.install`) | `ImportError` |
| `bt.AccessDeniedError` | a governed table or column the principal can't read | `PermissionError` |
| `bt.ExecutionError` | an operator failing at runtime in the engine | |
| `bt.OptimizationError` | the optimizer failing to produce a physical plan | |
| `bt.CompileError` | JIT compilation failing (the interpreter still runs) | |
| `bt.ResourceError` | the resource manager unable to grant memory or credit | |
| `bt.IOError` | a source or sink failing to read, write, list, or open | |
| `bt.FormatError` | an unknown format, or a file malformed for its format | |
| `bt.CommitError` | an atomic write commit failing (a concurrent-writer conflict) | |
| `bt.SchemaError` | schemas that can't be reconciled across files or against an expected one | |
| `bt.DataQualityError` | a `ds.dq...fail()` expectation with violating rows (carries the counts) | `ValueError` |
| `bt.BackendError` | a specific execution backend failing | |
| `bt.TransportError` | the distributed data plane (shared memory / Flight) failing | |

Because several also subclass a builtin, existing `except ValueError` /
`except ImportError` handlers keep working unchanged.

## A large query runs out of memory

Stateful operators hold state in memory by default. Pass `spill=True` to let
aggregation, join and sort spill to disk under pressure, and `distributed=True` with
`num_workers=` to spread the work across machines. Both are off by default. They add
overhead that only pays for itself on a big job.

```python
# docs: skip
out = ds.group_by("x").agg(total=bt.col("y").sum()).collect(spill=True)
```

## See also

- [Performance and memory](performance.md): caching, spill tuning, reading a query
  plan.
- [Distributed fault tolerance](../architecture/fault-tolerance.md): diagnosing a
  failed task, shuffle, or node.
- [Configuration options](../configuration/options.md): every tunable and its default.
- [Agent skills](../agents/index.md): `debug-a-batcher-query` is the triage tree a
  coding agent follows, organized by symptom, with the bisect procedure against DuckDB.
