# Adaptive re-optimization

Kyber, the optimizer, does not plan once and commit. At a **pipeline breaker** (a sort,
an aggregate, a join build) the engine has just *measured* the data it produced: real
row counts, real memory, real timings. It feeds those numbers back and re-plans the
rest of the query on them, rather than on the static estimates it started with.

The classic way a query goes wrong is a bad estimate. A filter expected to cut 90% of
rows cuts 5%. A join's "small" side turns out huge. A static optimizer commits to the
plan built from those guesses and runs it to the end, which is how jobs stall or run
out of memory. Batcher corrects mid-flight.

For comparison, DuckDB plans once, before execution. Spark AQE does re-plan, but only
at stage boundaries. Continuous re-optimization *inside* a running query is what
neither can retrofit, and it is why a query that starts on a bad estimate can still
finish fast and within memory.

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
