# Regression errors

Picking the metric is the modeling decision. MAE treats every miss equally, RMSE punishes big misses, MAPE is scale-free but explodes near zero, and Huber sits between MAE and MSE. All of them are aggregates here, so you can compute several in one pass.

The whole script, executed on every test run:

```{literalinclude} ../../../../examples/metrics/regression_errors.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/metrics/regression_errors.py
```

## See also

- {doc}`/cookbook/metrics/model/probabilistic_losses`: losses that score a probability or a margin rather than a hard label.
- {doc}`/cookbook/metrics/text/text_diversity`: repetition, truncation, refusal, and empty output.
- {doc}`/ml/evaluation/evaluation`: scoring a model, per segment, in one pass.
- {doc}`/api/models/metrics`: the complete metric vocabulary.
