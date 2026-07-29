# Regression error metrics: absolute, squared, percentage, and robust

Picking the metric is the modelling decision. MAE treats every miss equally, RMSE punishes big misses, MAPE is scale-free but explodes near zero, and Huber sits between MAE and MSE. All of them are aggregates here, so you can compute several in one pass.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/metrics/regression_errors.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/metrics/regression_errors.py
```
