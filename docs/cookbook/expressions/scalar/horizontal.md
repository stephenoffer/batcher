# Horizontal functions

An aggregate like ``sum()`` collapses a column. The ``*_horizontal`` family collapses a *row* across several columns and leaves the row count alone. That is what you want for a per-row total, a "did any check fail" flag, or a coalesce-style fallback.

The whole script, executed on every test run:

```{literalinclude} ../../../../examples/expressions/horizontal.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/expressions/horizontal.py
```

## See also

- {doc}`/cookbook/expressions/scalar/conditionals`: when/then/otherwise, and the SQL null helpers.
- {doc}`/cookbook/expressions/nested/json_columns`: reading JSON held in a string column, without parsing it in Python.
- {doc}`/user-guide/transform/columns/expressions`: what an expression is, and how it is evaluated.
- {doc}`/api/relational/expressions`: the complete {py:class}`Expr <batcher.plan.expr_ir.core.Expr>` reference.
