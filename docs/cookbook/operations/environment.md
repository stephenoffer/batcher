# What is installed, what the engine sees, and what to paste into a bug report

Half of "it works on my machine" is an optional extra present in one environment and absent in the other. These calls answer that in one line, and they are the first thing to include when reporting a problem.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/operations/environment.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/operations/environment.py
```

## See also

- {doc}`configuration`: configuring the engine: options, scoped overrides, and profiles.
- {doc}`error_handling`: the exception hierarchy: catching the failure you meant to catch.
- {doc}`../../user-guide/performance`: measuring and tuning a query that is correct but slow.
- {doc}`../../user-guide/observability`: what the engine records about a run, and where.
