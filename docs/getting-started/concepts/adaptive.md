# Adaptive re-optimization

Kyber, the optimizer, doesn't plan once and commit. A *pipeline breaker* is an operator
that must materialize before the next one starts, such as a sort, an aggregate, or a
join build. At each one the engine has already *measured* the data it produced: real row
counts, real memory, real timings. It feeds those numbers back and re-plans the rest of
the query on them, rather than on the static estimates it started with.

The classic way a query goes wrong is a bad estimate. A filter expected to cut 90% of
rows cuts 5%. A join's "small" side turns out huge. A static optimizer commits to the
plan built from those guesses and runs it to the end, which is how jobs stall or run
out of memory. Batcher corrects mid-flight.

For comparison, DuckDB plans once, before execution. Spark AQE re-plans at stage
boundaries, and so does Batcher: it's the same mechanism at the same granularity, with
the difference that Batcher does it single-node too. The loop also stays off below
20,000,000 input rows, so most queries never reach it.

The half with no equivalent elsewhere is what happens between runs. Batcher records what
each query actually did into a sketch-backed store, so the next run plans against
measured history rather than estimates alone.

![Two feedback loops. Within one query, Batcher plans, executes a stage to a pipeline breaker, measures the real cardinalities, and re-plans the remaining stages, which is stage-boundary re-optimization at Spark AQE's granularity and gated off below 20 million input rows. Across runs, it records what happened as sketches into the MetadataHub so the next run plans better.](/_static/diagrams/adaptive_loop.svg)

## A bad estimate, corrected

Suppose a filter is *expected* to keep most rows but actually keeps a handful. A static
plan, built for the large estimate, might pick a hash join sized for millions of rows
and thrash. Batcher runs the filter, measures that only a few rows survived, and
re-plans the join before it starts, often switching to a broadcast.

The measured half of that loop is visible to you. `stats()` runs the query and reports
what each operator really did: row counts, time, peak memory. Those are the same numbers
the engine feeds back into its next planning decision.

```python
import batcher as bt

ds = bt.from_pydict({"city": ["NYC", "LA", "NYC", "SF"], "amount": [10, 20, 30, 40]})
plan = ds.filter(bt.col("amount") > 15).group_by("city").agg(total=bt.col("amount").sum())

print(plan.stats().rows)   # rows the query actually produced
# 3
```

`explain()` shows the plan Kyber chose, without running it. `stats()` shows what
actually happened.


## See also

- {doc}`lazy`: why nothing runs until a terminal call, which is what makes re-planning possible.
- {doc}`/user-guide/operate/explain-plans`: reading the plan and the measured numbers behind it.
- {doc}`/deep-dives/adaptive/adaptive-reoptimization`: the re-planning loop, breaker by breaker.
