# Aggregates

Exact aggregates read every row. The `approx_*` family reads sketches instead, trading a bounded error for speed and, more importantly, for bounded memory on a high-cardinality column.

The script runs the aggregate vocabulary over one small table: counts, sums, means, spread, median and quantiles, `first` and `last`, `arg_min` and `arg_max` for the value of one column where another peaks, boolean and bitwise reductions, and `approx_count_distinct`, `approx_median`, and `approx_quantile`. It closes by running the same aggregates per group in a single pass.

The whole script, executed on every test run:

```{literalinclude} ../../../../examples/expressions/aggregates.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/expressions/aggregates.py
```

## See also

- {doc}`/cookbook/expressions/scalar/column_selectors`: naming columns by type or pattern instead of one at a time.
- {doc}`/cookbook/expressions/scalar/conditionals`: when/then/otherwise, and the SQL null helpers.
- {doc}`/user-guide/transform/columns/expressions`: what an expression is, and how it is evaluated.
- {doc}`/api/relational/expressions`: the complete {py:class}`Expr <batcher.plan.expr_ir.core.Expr>` reference.
