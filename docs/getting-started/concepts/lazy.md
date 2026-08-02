# Lazy, immutable datasets

A `Dataset` holds no data. It's a handle to a logical plan plus the inputs bound to
it. Every operation returns a *new* `Dataset`. Nothing is mutated in place, and no
work happens until you ask for results.

```python
import batcher as bt

ds = bt.from_pydict({"x": [1, 2, 3, 4], "g": ["a", "b", "a", "b"]})

filtered = ds.filter(bt.col("x") > 1)     # ds is unchanged
projected = filtered.select("x")          # filtered is unchanged

print(ds.columns)
# ['x', 'g']
```

Immutability is what lets you branch a pipeline from any intermediate handle and reuse
it without copying data. Two queries that share a prefix share the plan, so the
optimizer sees the whole thing at once.

## Terminal operations trigger execution

Chaining calls only grows the plan. The optimizer runs, and the engine executes, when
you call a *terminal* operation.

![The query lifecycle: reading and transforming build a lazy LogicalPlan; a terminal operation triggers optimization and execution, returning an Arrow result.](/_static/diagrams/lifecycle.svg)

The common terminals:

- `to_pydict()` gives you a column-oriented dict; `to_pylist()` gives you a list of
  row dicts.
- `collect()` returns a `pyarrow.Table`, and `count()` returns only the row count.
- `iter_batches()` streams Arrow record batches instead of materializing everything.
- `write.parquet(...)`, `write.csv(...)`, `write.json(...)`, and the generic
  `write(...)` send the result to a sink.

```python
plan = ds.filter(bt.col("x") >= 2).select("x")   # nothing runs yet
print(plan.to_pydict())                            # runs here
# {'x': [2, 3, 4]}
```

`explain()` returns the optimized plan as text without executing it. Reach for it when
you want to confirm what the optimizer actually did.

```python
print(plan.explain())
```


## See also

- {doc}`expressions`: what goes inside a plan once you have one.
- {doc}`adaptive`: how a lazy plan gets re-planned on measured row counts.
- {doc}`/user-guide/operate/tuning/explain-plans`: reading what the optimizer decided.
