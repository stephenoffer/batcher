# One core to a cluster

The query you test on a laptop is the query you run on a cluster. Batcher gets there by writing every stateful operator once, in a form that merges, so scaling out changes where the work runs and never what it means.

## Mergeable operators

Aggregation, join, distinct, and window all carry state across rows. Batcher implements each one exactly once, as a *mergeable* primitive with three steps: `partial` builds partition-local state, `combine` merges those states, and `finalize` produces rows. `combine` is associative and commutative, so the partials merge in any order and the answer doesn't depend on how the work was divided.

![Mergeable algebra: each partition computes a partial state, an associative combine merges them in any order, and finalize produces the result. The same code runs on one core or many machines.](/_static/diagrams/mergeable.svg)

That one implementation serves a single core, every core on the machine, and many machines. On one machine the parallel executor splits the input into Arrow batches of up to 16,384 rows, runs the partials on every core, and merges them. On a cluster the distributed path partitions the data, runs the partials on each worker, and combines the results. The operator doesn't know which of the three it's running under.

So distribution is a scheduling decision, not a second set of semantics. There's no separate distributed implementation that could drift from the single-node one.

```python
import batcher as bt

ds = bt.from_pydict({"x": [1, 2, 3, 4], "g": ["a", "b", "a", "b"]})

counts = ds.group_by("g").agg(n=bt.count()).sort("g")
print(counts.to_pydict())
# {'g': ['a', 'b'], 'n': [2, 2]}
```

## Scale out with one argument

{py:meth}`collect() <batcher.Dataset.collect>` defaults to `distributed="auto"`, which uses Ray when you're connected to a multi-node cluster and runs single-node otherwise. Pass `distributed=True` to force it. The plan is the same and so are the rows:

```python
# docs: skip
counts.collect(distributed=True)  # same plan, many machines, same rows
```

Ray only schedules the tasks. Bulk Arrow data moves between workers over Arrow Flight with credit-based flow control, and never passes through the Ray object store.

The mergeable form also bounds memory. State lives per partition, and when memory runs short the engine spills to disk instead of failing, on one machine or many. Spilling needs no flag. `collect(spill=True)` exists only to force it.

## Requirements and limitations

Distributed execution needs the `ray` extra and a Ray cluster. The rows, column names, and column types match a single-node run, with three exceptions where the query itself leaves the answer open:

- Floating-point sums and averages can differ in the last bits, because partitioning changes the order of addition.
- `row_number()` over rows that tie on the `ORDER BY` key can number the tied rows differently.
- A `limit` over data with no defined order, such as an unsorted `group_by`, can keep different rows. Add a `sort` first when you need the same rows every time.

## See also

- {doc}`/integrations/compute/ray`: running the distributed path on a real cluster.
- {doc}`/user-guide/operate/tuning/performance`: measuring and tuning before reaching for more machines.
- {doc}`/architecture/deep-dives/operators/mergeable-algebra`: the `partial`, `combine`, `finalize` contract in full.
- {doc}`adaptive`: how the engine re-plans on measured sizes once a query gets large.
