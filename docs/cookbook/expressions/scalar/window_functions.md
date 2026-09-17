# Window functions

A window function computes a value for each row from a set of related rows. The difference from `group_by` is that the row count is preserved, which is what you want for a running total, a rank within a partition, or a comparison against the previous row.

The script uses `.over(...)` to broadcast a partition total back to every row, ranks rows within a region with `row_number` and `dense_rank`, accumulates a running total with `cum_sum`, and reads the previous day's value with `shift` and the change with `diff`. It ends on the classic use of a broadcast aggregate, a share-of-total column.

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
- {doc}`/user-guide/transform/columns/expressions`: what an expression is, and how it is evaluated.
- {doc}`/api/relational/expressions`: the complete {py:class}`Expr <batcher.plan.expr_ir.core.Expr>` reference.
