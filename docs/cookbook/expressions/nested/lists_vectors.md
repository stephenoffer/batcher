# Vectors in list columns

A list column of floats is an embedding. Keeping it in the engine means a similarity search is a projection plus a sort rather than a round trip through NumPy, and it stays columnar when the table is larger than memory.

The whole script, executed on every test run:

```{literalinclude} ../../../../examples/expressions/lists_vectors.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/expressions/lists_vectors.py
```

## See also

- {doc}`/cookbook/expressions/nested/lists_transforms`: transforming inside a list column, without exploding it first.
- {doc}`/cookbook/expressions/scalar/nulls_and_casting`: the two places a pipeline quietly changes its answer.
- {doc}`/user-guide/transform/columns/expressions`: what an expression is, and how it is evaluated.
- {doc}`/api/relational/expressions`: the complete {py:class}`Expr <batcher.plan.expr_ir.core.Expr>` reference.
