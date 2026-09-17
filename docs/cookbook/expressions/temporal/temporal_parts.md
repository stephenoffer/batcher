# Calendar parts

Every accessor here is a projection, so extracting a year to group by costs one pass and no Python. Each part has one spelling, such as `dayofweek`, `week`, and `monthname`.

The script pulls year, quarter, month, day, and time-of-day parts from one timestamp column, then day-of-week and day-of-year with their names, the ISO calendar, and calendar facts such as `days_in_month` and `is_leap_year` on 2024. It closes by grouping on a derived part.

The whole script, executed on every test run:

```{literalinclude} ../../../../examples/expressions/temporal_parts.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/expressions/temporal_parts.py
```

## See also

- {doc}`/cookbook/expressions/temporal/temporal_differences`: durations between two timestamp columns, and shifting a timestamp.
- {doc}`/cookbook/expressions/temporal/temporal_timezones`: converting between them, and the reporting-boundary trap.
- {doc}`/user-guide/transform/columns/expressions`: what an expression is, and how it is evaluated.
- {doc}`/api/relational/expressions`: the complete {py:class}`Expr <batcher.plan.expr_ir.core.Expr>` reference.
