# Generalized linear models for counts, costs, and mixed zero-and-positive targets

Ordinary least squares assumes a symmetric, unbounded target. Counts are neither, and insurance-style cost data is a spike at zero plus a long positive tail. Poisson, gamma, and Tweedie regressions carry the right assumption for each.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/ml/glm_regressors.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/ml/glm_regressors.py
```

## See also

- {doc}`feature_construction`: building new features: interactions, ratios, calendar parts, lags, and rolling windows.
- {doc}`imbalance_and_sampling`: class imbalance: measure it, then resample or reweight.
- {doc}`../../ml/index`: the ML surface these recipes sit on.
- {doc}`../../ml/preprocessors/index`: the fit and transform steps most pipelines start with.
