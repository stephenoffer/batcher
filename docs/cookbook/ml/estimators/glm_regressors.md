# GLM regressors

Ordinary least squares assumes a symmetric, unbounded target. Counts are neither, and insurance-style cost data is a spike at zero plus a long positive tail. Poisson, gamma, and Tweedie regressions carry the right assumption for each.

The script fits `PoissonRegressor`, `GammaRegressor`, and `TweedieRegressor` with `power=1.5`, checks that predictions stay positive, scores the Poisson and Tweedie fits with their matching deviance, and compares against a `DummyRegressor` that predicts the mean.

The whole script, executed on every test run:

```{literalinclude} ../../../../examples/ml/glm_regressors.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/ml/glm_regressors.py
```

## See also

- {doc}`/cookbook/ml/preprocessing/feature_construction`: interactions, ratios, calendar parts, lags, and rolling windows.
- {doc}`/cookbook/ml/validation/imbalance_and_sampling`: measure it, then resample or reweight.
- {doc}`/ml/index`: the ML surface these recipes sit on.
- {doc}`/ml/evaluation/index`: scoring the model once it is fitted, as engine expressions.
