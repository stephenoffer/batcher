# Regularized linear regression: Ridge, Lasso, and ElasticNet

Every estimator follows the same two-step shape: ``fit(ds)`` returns a fitted model and ``predict(ds)`` returns a new Dataset with the prediction column appended. Fitting reads the data through the engine, so the training set never has to fit in memory as a NumPy array.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/ml/linear_models.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/ml/linear_models.py
```
