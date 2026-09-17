# Conditionals

`bt.when(...).then(...).otherwise(...)` is the columnar `if`. Chain `.when()` for more branches and the first matching branch wins, exactly like SQL `CASE`. Because it is an expression it runs in Rust, so a five-way bucketing is still one pass.

The script builds a multi-branch bucket and a two-branch flag, then the SQL null helpers `coalesce` and `nullif` and the row-wise `greatest` and `least`. `.otherwise(...)` is required and takes a real value, so the script also shows the pattern for an unmatched row that should end up null: give it a sentinel, then turn the sentinel into a null with `nullif`.

The whole script, executed on every test run:

```{literalinclude} ../../../../examples/expressions/conditionals.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/expressions/conditionals.py
```

## See also

- {doc}`/cookbook/expressions/scalar/column_selectors`: naming columns by type or pattern instead of one at a time.
- {doc}`/cookbook/expressions/scalar/horizontal`: reducing across columns instead of down rows.
- {doc}`/user-guide/transform/columns/expressions`: what an expression is, and how it is evaluated.
- {doc}`/api/relational/expressions`: the complete {py:class}`Expr <batcher.plan.expr_ir.core.Expr>` reference.
