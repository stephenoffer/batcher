# Embedding vectors as list columns: similarity, distance, and normalization

A list column of floats is an embedding. Keeping it in the engine means a similarity search is a projection plus a sort rather than a round trip through NumPy, and it stays columnar when the table is larger than memory.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/expressions/lists_vectors.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/expressions/lists_vectors.py
```

## See also

- {doc}`lists_transforms`: transforming inside a list column, without exploding it first.
- {doc}`nulls_and_casting`: nulls and type casting: the two places a pipeline quietly changes its answer.
- {doc}`../../user-guide/expressions`: how expressions are built, evaluated, and combined.
- {doc}`../../api/expressions`: the complete `Expr` reference.
