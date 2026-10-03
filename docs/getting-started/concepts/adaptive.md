# Adaptive re-optimization

Batcher measures what every query actually did and plans the next one from those measurements. For large joined queries it also re-plans in the middle of a run, as soon as a stage has produced real row counts.

![Two feedback loops. Within one query, Batcher plans, executes a stage to a pipeline breaker, measures the real cardinalities, and re-plans the remaining stages, which is stage-boundary re-optimization at Spark AQE's granularity. Across runs, it records what happened as sketches into the MetadataHub so the next run plans better.](/_static/diagrams/adaptive_loop.svg)

## See the measurements

{py:meth}`stats() <batcher.Dataset.stats>` runs the query and reports what each operator really did. These are the numbers the engine feeds back into its planning:

```python
import batcher as bt

ds = bt.from_pydict({"city": ["NYC", "LA", "NYC", "SF"], "amount": [10, 20, 30, 40]})
plan = ds.filter(bt.col("amount") > 15).group_by("city").agg(total=bt.col("amount").sum())

stats = plan.stats()
print(stats.rows)  # rows the query actually produced
# 3
```

Each operator carries its measured row counts next to the estimate the optimizer started from:

```python
for op in stats.ops:
    print(op.kind, op.rows_in, "->", op.rows_out)
# aggregate 3 -> 3
# filter 4 -> 3
# scan 4 -> 4
```

`explain()` shows the plan before it runs. `stats()` shows what happened.

## Learning across runs

After a query runs, Core records row counts, timings, and column sketches, such as distinct-value counts and quantiles, into the `MetadataHub`. Kyber, the optimizer, reads that history the next time it plans a query of the same shape: estimates become measurements, cost coefficients are calibrated from real timings, and a bandit learns which join strategy wins on your data. A repeated query gets faster the more it runs.

The history lives for the life of the process by default. Set the metadata backend to `sqlite` to keep it across restarts, as {doc}`/configuration/options` describes.

## Re-planning inside a query

A *pipeline breaker* is an operator that must finish before the next one starts, such as a sort, an aggregate, or a join build. Batcher executes up to a breaker, measures the real cardinality, and re-plans the rest of the query on that number. If a filter expected to keep millions of rows keeps a handful, the join after it can switch to a broadcast before it starts.

This is stage-boundary re-optimization, the same mechanism as Spark's adaptive query execution, running inside the Python process on a single machine. `collect()` decides when to use it, and you can force it either way:

```python
print(plan.collect(adaptive=False).num_rows, plan.collect(adaptive=True).num_rows)
# 3 3
```

:::{dropdown} When does `adaptive="auto"` engage?
Staging has a cost, so the default turns the loop on only when all of the following hold:

- The query has a join.
- The input clears 5,000,000 rows for each breaker the loop would cut at, about 10,000,000 rows for the simplest joined query.
- At least one join input is a genuine guess. If source statistics or earlier runs already size it well, measuring again changes nothing.

On a cluster, a few plan shapes always run staged whatever their size, such as a join over three or more sources.
:::

## See also

- {doc}`lazy`: why nothing runs until a terminal call, which is what makes re-planning possible.
- {doc}`/user-guide/operate/tuning/explain-plans`: reading the plan and the measured numbers behind it.
- {doc}`/architecture/deep-dives/adaptive/adaptive-reoptimization`: the re-planning loop, breaker by breaker.
