# Vector search

Keep embeddings as a list column and retrieval becomes a projection plus a top-k, composable with any other filter.

The script broadcasts a query vector onto the corpus with `cross_join`, scores it with `.list.cosine_similarity`, and retrieves three ways: `top_k`, a similarity threshold, and a metadata filter applied *before* scoring. Pre-filtering scores fewer vectors than filtering afterwards, and it honors the constraint exactly.

The whole script, executed on every test run:

```{literalinclude} ../../../../examples/ml/vector_search.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/ml/vector_search.py
```

## See also

- {doc}`/cookbook/ml/pipelines/text/rag-index`: chunking, embedding, and indexing a corpus to search like this.
- {doc}`/cookbook/metrics/embeddings`: aggregate health checks for an embedding column.
- {doc}`/cookbook/ml/inference/batch_inference`: a model over every row, without a Python loop.
- {doc}`/ml/retrieval/vector-search`: the engine scan, the ANN index, and which corpus size needs which.
