# Business days

Reports almost always want weekdays only, and almost always want a string at the end. Both are expressions, so the filter runs in the engine and the formatting happens in Rust rather than in a Python `strftime` loop.

The script flags weekends, weekdays, and business days across four days in March 2024 that straddle a weekend, formats timestamps with `strftime`, and filters to weekday traffic. No holiday calendar is applied, so a business day is a weekday and nothing more.

The whole script, executed on every test run:

```{literalinclude} ../../../../examples/expressions/temporal_business_days.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/expressions/temporal_business_days.py
```

## See also

- {doc}`/cookbook/expressions/nested/structs_and_maps`: nested records without flattening the table.
- {doc}`/cookbook/expressions/temporal/temporal_differences`: durations between two timestamp columns, and shifting a timestamp.
- {doc}`/user-guide/transform/columns/expressions`: what an expression is, and how it is evaluated.
- {doc}`/api/relational/expressions`: the complete {py:class}`Expr <batcher.plan.expr_ir.core.Expr>` reference.
