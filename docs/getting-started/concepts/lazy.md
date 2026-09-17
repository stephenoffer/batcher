# Lazy, immutable datasets

A {py:class}`Dataset <batcher.Dataset>` holds no data. It's a handle to a logical plan plus the inputs bound to it. Every operation returns a *new* `Dataset`, nothing is mutated in place, and no work happens until you ask for results.

```python
import batcher as bt

ds = bt.from_pydict({"x": [1, 2, 3, 4], "g": ["a", "b", "a", "b"]})

filtered = ds.filter(bt.col("x") > 1)  # ds is unchanged
projected = filtered.select("x")  # filtered is unchanged

print(ds.columns)
# ['x', 'g']
```

Immutability means you can branch a pipeline from any intermediate handle and reuse it without copying data or worrying that a later step changed it.

## Why waiting pays off

Because nothing runs early, the optimizer sees your whole query before it touches a byte. It can move a filter you wrote last down into the scan, read only the columns the final `select` needs, and choose a join strategy knowing what feeds the join. An eager library runs each step as you write it and can't take any of those back.

## Terminal operations trigger execution

Chaining calls only grows the plan. The optimizer runs, and the engine executes, when you call a *terminal* operation.

![The query lifecycle: reading and transforming build a lazy LogicalPlan; a terminal operation triggers optimization and execution, returning an Arrow result.](/_static/diagrams/lifecycle.svg)

The common terminal operations are the following:

- {py:meth}`to_pydict() <batcher.Dataset.to_pydict>` returns a column-oriented dict, and {py:meth}`to_pylist() <batcher.Dataset.to_pylist>` returns a list of row dicts.
- {py:meth}`collect() <batcher.Dataset.collect>` returns a `pyarrow.Table`, and {py:meth}`count() <batcher.Dataset.count>` returns only the row count.
- {py:meth}`iter_batches() <batcher.Dataset.iter_batches>` streams Arrow record batches instead of materializing everything.
- `write.parquet(...)`, `write.csv(...)`, `write.json(...)`, and the generic {py:obj}`write(...) <batcher.Dataset.write>` send the result to a sink.

```python
plan = ds.filter(bt.col("x") >= 2).select("x")  # nothing runs yet
print(plan.to_pydict())  # runs here
# {'x': [2, 3, 4]}
```

A dataset doesn't hold on to its result, so a second terminal call executes the plan again. If you'll ask several questions of one expensive intermediate result, mark it with {py:meth}`cache() <batcher.Dataset.cache>` and the first materializing call stores it.

`explain()` returns the optimized plan as text without executing it. Reach for it when you want to confirm what the optimizer did:

```python
print(plan.explain())
```

## See also

- {doc}`expressions`: what goes inside a plan once you have one.
- {doc}`adaptive`: how the plan improves from measured row counts.
- {doc}`/user-guide/operate/tuning/explain-plans`: reading what the optimizer decided.
- {doc}`/user-guide/operate/tuning/caching`: when to cache an intermediate result.
