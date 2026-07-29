# Vector search over an embedding column, in the engine

Keeping vectors as a list column means retrieval is a projection plus a top-N, composable with any other filter. That is what lets you pre-filter by metadata *before* scoring, which is both faster and more correct than scoring everything and filtering after.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/ml/vector_search.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/ml/vector_search.py
```
