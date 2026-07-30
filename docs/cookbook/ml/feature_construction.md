# Building new features: interactions, ratios, calendar parts, lags, and rolling windows

These are the featurizers that turn a raw table into a model-ready one. The time-series ones (``LagFeaturizer``, ``RollingFeaturizer``) need an ``order_by`` and usually a ``partition_by``: forgetting the partition silently leaks one entity's history into another's features.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/ml/feature_construction.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/ml/feature_construction.py
```

## See also

- {doc}`clustering_and_decomposition`: unsupervised models: KMeans, Gaussian mixtures, PCA, and truncated SVD.
- {doc}`glm_regressors`: generalized linear models for counts, costs, and mixed zero-and-positive targets.
- {doc}`../../ml/index`: the ML surface these recipes sit on.
- {doc}`../../ml/preprocessors/index`: the fit and transform steps most pipelines start with.
