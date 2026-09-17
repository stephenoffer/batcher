# List set operations

This is the shape behind "which tags do these two documents share" and "what did the user add to the cart since last time". Everything is per row and columnar, so a set operation over a million rows never builds a million Python sets.

The script compares a `before` and an `after` list column with `union`, `intersect`, and `difference`, uses `has_any` and `has_all` when a yes or no is enough, and shows that `concat` keeps duplicates where `union` does not. It builds Jaccard similarity from the set operators and explains why `.list.jaccard` is a different measure.

The whole script, executed on every test run:

```{literalinclude} ../../../../examples/expressions/lists_set_operations.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/expressions/lists_set_operations.py
```

## See also

- {doc}`/cookbook/expressions/nested/lists_basics`: indexing, slicing, joining, and flattening.
- {doc}`/cookbook/expressions/nested/lists_transforms`: transforming inside a list column, without exploding it first.
- {doc}`/user-guide/transform/columns/expressions`: what an expression is, and how it is evaluated.
- {doc}`/api/relational/expressions`: the complete {py:class}`Expr <batcher.plan.expr_ir.core.Expr>` reference.
