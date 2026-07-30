# Losses that score a probability or a margin rather than a hard label

A classifier that says "0.51" and one that says "0.99" both predict the positive class, but they are not equally right. These losses read the score column, which is what you need to tell a confident model from a lucky one.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/metrics/probabilistic_losses.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/metrics/probabilistic_losses.py
```

## See also

- {doc}`embeddings`: corpus-level embedding metrics: monitoring a vector column in aggregate.
- {doc}`regression_errors`: regression error metrics: absolute, squared, percentage, and robust.
- {doc}`../../ml/evaluation`: scoring a model, per segment, in one pass.
- {doc}`../../api/metrics`: the complete metric vocabulary.
