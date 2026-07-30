# Window functions: per-row values computed from a window of related rows

The difference from ``group_by`` is that the row count is preserved. That is what you want for a running total, a rank within a partition, or a comparison against the previous row.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/expressions/window_functions.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/expressions/window_functions.py
```

## See also

- {doc}`temporal_truncation`: bucketing timestamps: truncate to a period, or snap to a period boundary.
- {doc}`temporal_timezones`: time zones: converting between them, and the reporting-boundary trap.
- {doc}`../../user-guide/expressions`: how expressions are built, evaluated, and combined.
- {doc}`../../api/expressions`: the complete `Expr` reference.
