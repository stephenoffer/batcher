# GLM regressors

Ordinary least squares assumes a symmetric, unbounded target. Counts are neither, and insurance-style cost data is a spike at zero plus a long positive tail. Poisson, gamma, and Tweedie regressions carry the right assumption for each.

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
- {doc}`/ml/preparing/preprocessors/index`: the fit and transform steps most pipelines start with.
