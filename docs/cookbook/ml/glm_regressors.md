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
