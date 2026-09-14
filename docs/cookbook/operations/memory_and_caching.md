# Memory and caching

`cache()` is an execution hint. The result is identical with it, without it, and at every storage level, so the only thing it can change is what the query costs. Read `cache_stats()` as a difference across a span of work rather than as an absolute, because the counters are lifetime figures for the process. Spilling is the same bargain for memory: under a small budget the engine goes out of core instead of failing, and the answer does not move.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/operations/memory_and_caching.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/operations/memory_and_caching.py
```

## See also

- {doc}`inspecting_a_query`: reading a plan, timing a query, and checking what the engine actually ran.
- {doc}`observability`: verbosity, logging, and execution statistics.
- {doc}`/user-guide/operate/tuning/performance`: measuring and tuning a query that is correct but slow.
- {doc}`/user-guide/operate/running/observability`: what the engine records about a run, and where.
