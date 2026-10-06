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

## What runs when

Not every call is either free or a full run. The following table labels the common calls by what they do when you make them, from cheapest to most far-reaching:

| Label | What it does | Calls |
|---|---|---|
| Plan-building | Extends or records a plan. Reads no data and runs nothing. | transformations such as `filter`, `select`, `join`, and `group_by(...).agg(...)`; `cache()`; `columns`; `explain()`; `Session.register`; a SQL `SELECT`, `CREATE VIEW`, or `CREATE TABLE AS` on a session table |
| Metadata-reading | Touches storage or resolves types, but scans no rows. | `bt.read.*` constructors, which check the path and read the source's schema; `schema` |
| Executing | Runs the whole plan and returns a result. | `collect`, `to_pydict`, `to_pylist`, `to_pandas`, `count`, `len(ds)`, `shape`, `explain(analyze=True)`, and an ML estimator's `fit` |
| Streaming | Runs the plan incrementally as you consume it. | `iter_batches` and `iter_rows`, which start work at the first `next()`, not when called |
| Externally mutating | Runs the plan and changes something outside the process. | every `ds.write.*` sink, and SQL `CREATE TABLE ... AS`, `INSERT`, `DELETE`, or `UPDATE` on a *catalog* table, which write immediately |

Two cases need care. `len(ds)` and `shape` look like attribute reads but execute a `count`, often answered from file metadata and otherwise from a full run. And a Python callable such as `map_batches(fn)` has output types Batcher can't know without running it, so `schema`, or an `INSERT` into a session table built on it, can call `fn` on input data. A SQL `INSERT`, `DELETE`, or `UPDATE` on a session table otherwise only rebinds the name to a new lazy plan.

`iter_batches` doesn't start a query until you ask for the first batch:

```python
stream = plan.iter_batches()  # nothing has run yet
print(next(stream).num_rows)  # runs here
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
