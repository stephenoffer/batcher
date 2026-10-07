# Vectors in list columns

A list column of floats is an embedding. Keeping it in the engine means a similarity search is a projection plus a sort rather than a round trip through NumPy, and it stays columnar when the table is larger than memory.

The script scores three small vectors against a query with `cosine_similarity`, `cosine_distance`, `dot`, and `euclidean_distance`, and shows why a vector scaled by two has the same cosine similarity but a different L2 distance. It checks `l2_norm`, `len`, and unit norm, normalizes a vector, and finishes with the nearest-neighbor ranking these functions exist for.

The distance functions need two vectors of one dimension. When both columns are fixed-size lists whose declared sizes differ, such as a 2-element and a 3-element embedding, the plan fails with a `PlanError` naming both columns before any row is read. Variable-length lists are checked per row as they are scored.

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
