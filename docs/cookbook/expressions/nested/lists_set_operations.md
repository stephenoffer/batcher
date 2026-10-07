# List set operations

This is the shape behind "which tags do these two documents share" and "what did the user add to the cart since last time". Everything is per row and columnar, so a set operation over a million rows never builds a million Python sets.

The script compares a `before` and an `after` list column with `union`, `intersect`, and `difference`, uses `has_any` and `has_all` when a yes or no is enough, and shows that `concat` keeps duplicates where `union` does not. It builds Jaccard similarity from the set operators, checks it against `.list.jaccard(mode="set")`, and explains why the default `.list.jaccard` is a different measure.

## Set and multiset semantics

`union`, `intersect` and `difference` treat each list as a *set*: duplicates collapse to one, and the result keeps the order of first appearance in the left list. A null element is a value like any other and equals another null, which is Spark's `array_intersect` rule. DuckDB's `list_intersect` drops nulls instead, so `[1, 1, None, 2]` intersected with `[1, None, 3]` is `[1, None]` here and `[1]` there. Call `drop_nulls()` on both sides first when you want DuckDB's answer.

`multiset_overlap` is the *multiset* counterpart. It counts repeats, clipped to the smaller count on either side, and a null element matches nothing. `jaccard(mode="set")` is the set Jaccard index, the size of the intersection over the size of the union of the distinct non-null values, which is null for two lists with no value at all.

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
