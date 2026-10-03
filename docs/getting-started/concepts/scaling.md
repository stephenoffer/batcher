# One core to a cluster

The query you test on a laptop is the query you run on a cluster. Batcher gets there by writing every stateful operator once, in a form that merges, so scaling out changes where the work runs and never what it means.

## Mergeable operators

Aggregation, join, distinct, and window all carry state across rows. Batcher implements each one exactly once, as a *mergeable* primitive with three steps: `partial` builds partition-local state, `combine` merges those states, and `finalize` produces rows. `combine` is associative and commutative, so the partials merge in any order and the answer doesn't depend on how the work was divided.

![Mergeable algebra: each partition computes a partial state, an associative combine merges them in any order, and finalize produces the result. The same code runs on one core or many machines.](/_static/diagrams/mergeable.svg)

The same implementation serves one core, every core on the machine, and many machines. Distribution is a scheduling decision, not a second set of semantics.

```python
import batcher as bt

ds = bt.from_pydict({"x": [1, 2, 3, 4], "g": ["a", "b", "a", "b"]})

counts = ds.group_by("g").agg(n=bt.count()).sort("g")
print(counts.to_pydict())
# {'g': ['a', 'b'], 'n': [2, 2]}
```

Every built-in aggregate has a mergeable form, including averages and distinct counts:

```python
stats = ds.group_by("g").agg(
    avg=bt.col("x").mean(), hi=bt.col("x").max(), uniq=bt.col("x").count_distinct()
)
print(stats.sort("g").to_pydict())
# {'g': ['a', 'b'], 'avg': [2.0, 3.0], 'hi': [3, 4], 'uniq': [2, 2]}
```

## Scale out with one argument

{py:meth}`collect() <batcher.Dataset.collect>` defaults to `distributed="auto"`: Ray when you're connected to a multi-node cluster, single-node otherwise. Pass `distributed=True` to force it:

```python
# docs: skip
counts.collect(distributed=True)  # same plan, many machines, same rows
```

Ray only schedules the tasks. Bulk Arrow data moves between workers over Arrow Flight with credit-based flow control, never through the Ray object store.

The mergeable form also bounds memory. State lives per partition, and when memory runs short the engine spills to disk, on one machine or many. Spilling needs no flag:

```python
print(counts.collect(spill=True).num_rows)  # force a spill, same answer
# 2
```

## Requirements and limitations

- Distributed execution needs the `ray` extra and a Ray cluster.
- Rows, column names, and column types match a single-node run. Where the query itself leaves the answer open, the result can vary: floating-point sums in the last bits, `row_number()` over tied `ORDER BY` keys, and a `limit` over unordered data. Add a `sort` when you need the same rows every time.

## See also

- {doc}`/integrations/compute/ray`: running the distributed path on a real cluster.
- {doc}`/user-guide/operate/tuning/performance`: measuring and tuning before reaching for more machines.
- {doc}`/architecture/deep-dives/operators/mergeable-algebra`: the `partial`, `combine`, `finalize` contract in full.
- {doc}`adaptive`: how the engine re-plans on measured sizes once a query gets large.
