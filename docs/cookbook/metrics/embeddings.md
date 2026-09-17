# Embedding metrics

Per-row similarity is a projection. These are the aggregates over it: mean cosine similarity, cosine and angular distance, dot product, and Euclidean and Manhattan distance across two vector columns, plus the mean norm, unit-norm rate, and zero-vector rate of one.

They make cheap health checks for an embedding job. A zero vector is usually a failed embedding call, and the script ends by measuring exactly that. A drifting mean similarity usually means the upstream text changed rather than the model.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/metrics/embeddings.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/metrics/embeddings.py
```

## See also

- {doc}`/cookbook/metrics/text/text_retrieval`: is the answer actually supported by the retrieved context?
- {doc}`/cookbook/metrics/text/text_overlap`: comparing a generated answer against a reference, without a model.
- {doc}`/ml/evaluation/evaluation`: scoring a model, per segment, in one pass.
- {doc}`/api/models/metrics`: the complete metric vocabulary.
