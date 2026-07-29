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
