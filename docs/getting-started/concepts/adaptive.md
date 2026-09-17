# Adaptive re-optimization

A static optimizer plans from estimates and never finds out how wrong they were. Batcher measures what every query actually did and plans the next one from those measurements. For large joined queries it also re-plans in the middle of a run, as soon as a stage has produced real row counts. This page explains both loops and how to see the numbers they use.

## Why estimates go wrong

The classic way a query goes wrong is a bad estimate. A filter expected to drop 90% of rows drops 5%. A join's "small" side turns out huge. An optimizer that commits to the plan built from those guesses runs it to the end, and that's how jobs stall or run out of memory.

Batcher closes the gap in two ways. Both are shown below: the inner loop runs inside one query, and the outer loop runs across queries.

![Two feedback loops. Within one query, Batcher plans, executes a stage to a pipeline breaker, measures the real cardinalities, and re-plans the remaining stages, which is stage-boundary re-optimization at Spark AQE's granularity. Across runs, it records what happened as sketches into the MetadataHub so the next run plans better.](/_static/diagrams/adaptive_loop.svg)

## Learning across runs

After a query runs, Core records what each operator really produced into the `MetadataHub`: row counts, timings, and sketches of the data such as distinct-value counts and quantiles. Kyber, the optimizer, reads that history the next time it plans a query of the same shape. Estimates become measurements, cost coefficients get calibrated from measured timings, and a bandit learns which join strategy wins on your data. So a repeated query tends to get faster the more it runs.

Neither DuckDB nor Spark keeps anything like this between queries. By default the history lives for the life of the process. Set the metadata backend to `sqlite` to keep it across restarts, as {doc}`/configuration/options` describes.

## Re-planning inside a query

A *pipeline breaker* is an operator that must finish before the next one starts, such as a sort, an aggregate, or a join build. When the inner loop is on, Batcher executes up to a breaker, measures the real cardinality, and re-plans the rest of the query on that number rather than on the estimate it started with.

Suppose a filter is expected to keep most rows but actually keeps a handful. A plan built for the large estimate might size a hash join for millions of rows. Batcher runs the filter, sees that only a few rows survived, and re-plans the join before it starts, where it can choose a broadcast instead.

This is stage-boundary re-optimization, the same mechanism and granularity as Spark's adaptive query execution. The difference is that Batcher does it on a single machine too, where AQE needs shuffle stages. Staging has a cost, so `collect(adaptive="auto")` turns the loop on only when all of the following hold:

- The query has a join. A query with no join never qualifies.
- The input clears 5,000,000 rows for each breaker the loop would cut at. That's about 10,000,000 rows for the simplest joined query.
- At least one join input is a genuine guess. If source statistics or earlier runs already size it well, measuring again changes nothing.

Pass `adaptive=True` or `adaptive=False` to force it either way. On a cluster, a few plan shapes always run staged whatever their size, such as a join over three or more sources, because staging is the only way to distribute them correctly.

## See the measurements yourself

The measured half of the loop is visible to you. {py:meth}`stats() <batcher.Dataset.stats>` runs the query and reports what each operator really did, including row counts, time, and peak memory. Those are the numbers the engine feeds back into its planning.

```python
import batcher as bt

ds = bt.from_pydict({"city": ["NYC", "LA", "NYC", "SF"], "amount": [10, 20, 30, 40]})
plan = ds.filter(bt.col("amount") > 15).group_by("city").agg(total=bt.col("amount").sum())

print(plan.stats().rows)  # rows the query actually produced
# 3
```

`explain()` shows the plan Kyber chose without running it. `stats()` shows what actually happened.

## See also

- {doc}`lazy`: why nothing runs until a terminal call, which is what makes re-planning possible.
- {doc}`/user-guide/operate/tuning/explain-plans`: reading the plan and the measured numbers behind it.
- {doc}`/architecture/deep-dives/adaptive/adaptive-reoptimization`: the re-planning loop, breaker by breaker.
