# Error handling

Every error the engine raises descends from `BatcherError`, so one handler at the pipeline boundary catches all of them. Catch a subclass when the response differs, because telling them apart is the difference between retrying and giving up. `PlanError` is your mistake. `IOError` is the environment's. `MissingDependencyError` means an optional extra was never installed. `ColumnNotFoundError` is also a `KeyError`, so mapping-style handling works on it.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/operations/error_handling.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/operations/error_handling.py
```

## See also

- {doc}`environment`: what is installed, what the engine sees, and what to paste into a bug report.
- {doc}`inspecting_a_query`: reading a plan, timing a query, and checking what the engine actually ran.
- {doc}`/user-guide/operate/tuning/performance`: measuring and tuning a query that is correct but slow.
- {doc}`/user-guide/operate/running/observability`: what the engine records about a run, and where.
