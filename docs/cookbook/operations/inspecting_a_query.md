# Inspecting a query

`explain()` prints the optimized plan, one line per operator with its row estimate. That is where you confirm the filter really runs below the aggregate rather than above it. Guessing is slower. When you want to assert on plan shape in a test, `ds.meta.explain()` returns the plan as a dict instead of a string.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/operations/inspecting_a_query.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/operations/inspecting_a_query.py
```

## See also

- {doc}`error_handling`: catching the failure you meant to catch.
- {doc}`memory_and_caching`: caching a reused branch and spilling under a tight budget.
- {doc}`/user-guide/operate/tuning/performance`: measuring and tuning a query that is correct but slow.
- {doc}`/user-guide/operate/running/observability`: what the engine records about a run, and where.
