# Embedding metrics

Per-row similarity is a projection. These are the aggregates over it, and they are the cheap health checks for an embedding job: a drifting mean cosine similarity or a rising zero-vector rate usually means the upstream text changed rather than the model.

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
