# Business days

Reports almost always want weekdays only, and almost always want a string at the end. Both are expressions, so the filter pushes down toward the scan and the formatting happens in Rust rather than in a Python ``strftime`` loop.

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
