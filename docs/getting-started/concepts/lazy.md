# Lazy, immutable datasets

A `Dataset` does not hold data. It is a handle to a logical plan plus its bound
inputs. Every operation returns a *new* `Dataset`; nothing is mutated in place, and
no work happens until you ask for results.

```python
import batcher as bt

ds = bt.from_pydict({"x": [1, 2, 3, 4], "g": ["a", "b", "a", "b"]})

filtered = ds.filter(bt.col("x") > 1)     # ds is unchanged
projected = filtered.select("x")          # filtered is unchanged

print(ds.columns)
# ['x', 'g']
```

Because datasets are immutable, you can branch a pipeline from any intermediate
handle and reuse it without copying data — two queries that share a prefix share the
plan, and the optimizer sees the whole thing.

## Terminal operations trigger execution

The plan builds up as you chain calls; the optimizer runs, and the engine executes,
only when you call a *terminal* operation.

![The query lifecycle: reading and transforming build a lazy LogicalPlan; a terminal operation triggers optimization and execution, returning an Arrow result.](../../_static/diagrams/lifecycle.png)

The common terminals:

- `to_pydict()` — a column-oriented dict.
- `to_pylist()` — a list of row dicts.
- `collect()` — a `pyarrow.Table`.
- `count()` — the row count.
- `iter_batches()` — an iterator of Arrow record batches.
- `write.parquet(...)`, `write.csv(...)`, `write.json(...)`, `write(...)`.

```python
plan = ds.filter(bt.col("x") >= 2).select("x")   # nothing runs yet
print(plan.to_pydict())                            # runs here
# {'x': [2, 3, 4]}
```

`explain()` returns the optimized plan as text without executing it — useful for
confirming what the optimizer did.

```python
print(plan.explain())
```
