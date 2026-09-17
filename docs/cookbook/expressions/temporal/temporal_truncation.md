# Truncating timestamps

Rollups need a bucket key, and a raw timestamp is never one. `truncate` rounds a timestamp down to a unit, which is how you build an hourly or daily key. The `*_start` and `*_end` pairs snap to calendar boundaries, which is what a month-over-month report needs.

The script truncates to the hour, drops the time of day with `normalize`, snaps to month, quarter, and year boundaries including a leap-year February, and tests boundaries with `is_month_start` and `is_quarter_start`. It ends with the hourly rollup these exist for.

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
