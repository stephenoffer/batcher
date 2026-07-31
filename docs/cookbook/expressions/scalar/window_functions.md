# Window functions

The difference from ``group_by`` is that the row count is preserved. That is what you want for a running total, a rank within a partition, or a comparison against the previous row.

The whole script, executed on every test run:

```{literalinclude} ../../../../examples/expressions/window_functions.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/expressions/window_functions.py
```

## See also

- {doc}`/cookbook/expressions/temporal/temporal_truncation`: truncate to a period, or snap to a period boundary.
- {doc}`/cookbook/expressions/temporal/temporal_timezones`: converting between them, and the reporting-boundary trap.
- {doc}`/user-guide/transform/expressions`: what an expression is, and how it is evaluated.
- {doc}`/api/relational/expressions`: the complete `Expr` reference.
