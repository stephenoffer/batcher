# Embedding metrics

Per-row similarity is a projection; these are the aggregates over it. They are the cheap health checks for an embedding job: a drifting mean cosine similarity or a rising zero-vector rate usually means the upstream text changed, not the model.

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

- {doc}`/cookbook/metrics/model/diagnostic`: the epidemiology-style view of a binary classifier.
- {doc}`/cookbook/metrics/model/probabilistic_losses`: losses that score a probability or a margin rather than a hard label.
- {doc}`/ml/evaluation/evaluation`: scoring a model, per segment, in one pass.
- {doc}`/api/models/metrics`: the complete metric vocabulary.
