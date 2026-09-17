# Column selectors

A selector is an `Expr` leaf standing for *every* matching column, so "round every float" is one expression that keeps working when a column is added. Spelling out names is how a pipeline silently stops covering a new column.

The script selects columns by type family and by name pattern, takes everything or everything except, and combines selectors with `|`, `&`, `-`, and `~`. The payoff is a single `with_columns` that rounds every matched column in place and leaves the rest untouched.

The whole script, executed on every test run:

```{literalinclude} ../../../../examples/expressions/column_selectors.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/expressions/column_selectors.py
```

## See also

- {doc}`/cookbook/expressions/scalar/aggregates`: counts, positions, quantiles, and approximations.
- {doc}`/cookbook/expressions/scalar/conditionals`: when/then/otherwise, and the SQL null helpers.
- {doc}`/user-guide/transform/columns/expressions`: what an expression is, and how it is evaluated.
- {doc}`/api/relational/expressions`: the complete {py:class}`Expr <batcher.plan.expr_ir.core.Expr>` reference.
