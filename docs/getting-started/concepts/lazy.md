# Lazy, immutable datasets

A {py:class}`Dataset <batcher.Dataset>` holds no data. It's a handle to a logical plan plus the inputs bound to it. Every operation returns a *new* `Dataset`, nothing is mutated in place, and no work happens until you ask for results.

```python
import batcher as bt

ds = bt.from_pydict({"x": [1, 2, 3, 4], "g": ["a", "b", "a", "b"]})

wide = ds.with_columns(y=bt.col("x") * 2)  # a new Dataset
print(ds.columns, wide.columns)  # ds is unchanged
# ['x', 'g'] ['x', 'g', 'y']
```

Branch a pipeline from any intermediate handle and reuse it freely. Nothing is copied, and no later step can change it:

```python
evens = ds.filter(bt.col("x") % 2 == 0)
print(evens.select("x").to_pydict(), evens.select("g").to_pydict())
# {'x': [2, 4]} {'g': ['b', 'b']}
```

## Why waiting pays off

Because nothing runs early, the optimizer sees your whole query before it touches a byte. It moves a filter you wrote last down into the scan, reads only the columns the final `select` needs, and picks a join strategy knowing what feeds the join.

## Terminal operations trigger execution

Chaining calls only grows the plan. The engine runs it when you call a *terminal* operation.

![The query lifecycle: reading and transforming build a lazy LogicalPlan; a terminal operation triggers optimization and execution, returning an Arrow result.](/_static/diagrams/lifecycle.svg)

```python
plan = ds.filter(bt.col("x") >= 2).select("x")  # nothing runs yet
print(plan.to_pydict())  # runs here
# {'x': [2, 3, 4]}
```

Each terminal returns the result in a different shape:

```python
print(plan.count())
# 3
print(plan.to_pylist())
# [{'x': 2}, {'x': 3}, {'x': 4}]
print(plan.collect().num_rows)  # a pyarrow.Table
# 3
```

{py:meth}`iter_batches() <batcher.Dataset.iter_batches>` streams Arrow record batches instead of materializing everything, and `write.parquet(...)`, `write.csv(...)`, `write.json(...)`, and {py:obj}`write(...) <batcher.Dataset.write>` send the result to a sink:

```python
for batch in plan.iter_batches():
    print(batch.num_rows)
# 3
```

## Cache and explain

A dataset doesn't keep its result, so a second terminal call runs the plan again. Mark an expensive intermediate with {py:meth}`cache() <batcher.Dataset.cache>` and the first materializing call stores it:

```python
cached = plan.cache()
print(cached.count(), cached.to_pydict())
# 3 {'x': [2, 3, 4]}
```

`explain()` returns the optimized plan as text without executing it. Here the filter is already pushed into the scan:

```python
print("pushed" in plan.explain())
# True
```

## See also

- {doc}`expressions`: what goes inside a plan once you have one.
- {doc}`adaptive`: how the plan improves from measured row counts.
- {doc}`/user-guide/operate/tuning/explain-plans`: reading what the optimizer decided.
- {doc}`/user-guide/operate/tuning/caching`: when to cache an intermediate result.
