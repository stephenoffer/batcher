# Class imbalance

Resampling changes the data; weighting changes the loss. Prefer weights when the model supports them, because oversampling duplicates rows (and any leakage in them) while undersampling throws information away.

The whole script, executed on every test run:

```{literalinclude} ../../../../examples/ml/imbalance_and_sampling.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/ml/imbalance_and_sampling.py
```

## See also

- {doc}`/cookbook/ml/estimators/glm_regressors`: generalized linear models for counts, costs, and mixed zero-and-positive targets.
- {doc}`/cookbook/ml/estimators/linear_models`: Ridge, Lasso, and ElasticNet.
- {doc}`/ml/index`: the ML surface these recipes sit on.
- {doc}`/ml/preparing/preprocessors/index`: the fit and transform steps most pipelines start with.
