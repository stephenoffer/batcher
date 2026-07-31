# Inspecting a query

``explain()`` shows the optimized plan, which is where you confirm a predicate really was pushed into the scan. Reading the plan is faster than guessing, and it is the only way to tell a fused pipeline from three separate passes.

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
- {doc}`/user-guide/operate/performance`: measuring and tuning a query that is correct but slow.
- {doc}`/user-guide/operate/observability`: what the engine records about a run, and where.
