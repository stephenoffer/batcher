# Corpus-level embedding metrics: monitoring a vector column in aggregate

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
