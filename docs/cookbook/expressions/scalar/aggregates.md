# Aggregates

Exact aggregates read every row. The ``approx_*`` family reads sketches instead, trading a bounded error for a large constant-factor speedup and, more importantly, bounded memory on a high-cardinality column.

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
