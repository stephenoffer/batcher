# Regression errors

Picking the error metric is a modeling decision. MAE treats every miss equally, RMSE punishes big misses, MAPE is scale-free but explodes near zero, and Huber sits between MAE and MSE.

The script's residuals are +1, -1, +2, and -4. Watch the one large miss pull RMSE above MAE while the median absolute error stays at 1.5. Every metric is an aggregate, so all seventeen come out of a single `select`.

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
- {doc}`/cookbook/metrics/model/agreement`: how well a prediction tracks the truth, not just how close.
- {doc}`/ml/evaluation/evaluation`: scoring a model, per segment, in one pass.
- {doc}`/api/models/metrics`: the complete metric vocabulary.
