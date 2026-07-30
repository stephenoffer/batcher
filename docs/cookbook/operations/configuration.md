# Configuring the engine: options, scoped overrides, and profiles

Configuration is a value, not global mutable state you have to remember to undo. ``option_context`` and ``config_context`` scope an override to a block, so a memory-tight step cannot leak its settings into the rest of the program.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/operations/configuration.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/operations/configuration.py
```

## See also

- {doc}`environment`: what is installed, what the engine sees, and what to paste into a bug report.
- {doc}`error_handling`: the exception hierarchy: catching the failure you meant to catch.
- {doc}`../../user-guide/performance`: measuring and tuning a query that is correct but slow.
- {doc}`../../user-guide/observability`: what the engine records about a run, and where.
