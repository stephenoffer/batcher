# Truncating timestamps

``truncate``/``floor`` round a timestamp down to a unit, which is how you build an hourly or daily rollup key. The ``*_start``/``*_end`` pairs snap to calendar boundaries, which is what a month-over-month report needs.

The whole script, executed on every test run:

```{literalinclude} ../../../../examples/expressions/temporal_truncation.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/expressions/temporal_truncation.py
```

## See also

- {doc}`/cookbook/expressions/temporal/temporal_timezones`: converting between them, and the reporting-boundary trap.
- {doc}`/cookbook/expressions/scalar/window_functions`: per-row values computed from a window of related rows.
- {doc}`/user-guide/transform/columns/expressions`: what an expression is, and how it is evaluated.
- {doc}`/api/relational/expressions`: the complete {py:class}`Expr <batcher.plan.expr_ir.core.Expr>` reference.
