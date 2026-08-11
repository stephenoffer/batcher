# Nulls and casting

Null is not zero and not empty string, and every aggregate skips it. Casting is where a schema mismatch between two sources gets resolved, and where an unparseable value becomes a null rather than an error.

The whole script, executed on every test run:

```{literalinclude} ../../../../examples/expressions/nulls_and_casting.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/expressions/nulls_and_casting.py
```

## See also

- {doc}`/cookbook/expressions/nested/lists_vectors`: similarity, distance, and normalization.
- {doc}`/cookbook/expressions/scalar/numeric_math`: arithmetic and math functions on numeric columns.
- {doc}`/user-guide/transform/columns/expressions`: what an expression is, and how it is evaluated.
- {doc}`/api/relational/expressions`: the complete {py:class}`Expr <batcher.plan.expr_ir.core.Expr>` reference.
