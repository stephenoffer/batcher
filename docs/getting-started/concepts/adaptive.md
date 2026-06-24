# Adaptive re-optimization

This is the idea that sets Batcher apart. The optimizer (Kyber) does not optimize
once and commit. At **pipeline breakers** — sort, aggregate, join build — the engine
has just *measured* the data it produced: real row counts, real memory, real
timings. It feeds those numbers back and re-plans the rest of the query on them
instead of the static estimates it started with.

That matters because the classic way a query goes wrong is a bad estimate: a filter
expected to cut 90% of rows that cuts 5%, or a join whose "small" side turns out
huge. A static optimizer commits to the plan built from those guesses and runs it to
the end — which is how jobs stall or run out of memory. Batcher corrects mid-flight.

For comparison: DuckDB's optimizer is static (it plans once, before execution);
Spark AQE re-plans, but only at stage boundaries. Continuous re-optimization *inside*
a running query is what neither can retrofit, and it is the reason a query that
starts on a bad estimate can still finish fast and within memory.

`explain()` shows the plan the optimizer chose, and `stats()` reports the measured
per-operator rows, time, and peak memory that feed the next decision — the same
signal the engine uses to re-plan.
