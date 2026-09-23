# Nulls and casting

Null is not zero and not the empty string, and every column aggregate skips it. With one null present, `col("a").count()` is 2 where {py:obj}`bt.count() <batcher.count>` is 3. Casting is where a schema mismatch between two sources gets resolved, and the strictness is a choice: `cast` raises on a value it cannot parse, while `try_cast` turns that value into a null. The second one fails silently. That is the point, and the risk.

The script pins each of those behaviors with an assertion, uses `coalesce` as a multi-fallback `fill_null`, counts the nulls a `try_cast` introduced, and checks the float edge cases with `is_nan` and `is_finite`.

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
