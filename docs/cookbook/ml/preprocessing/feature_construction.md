# Feature construction

These are the featurizers that turn a raw table into a model-ready one. The time-series ones (``LagFeaturizer``, ``RollingFeaturizer``) need an ``order_by`` and usually a ``partition_by``: forgetting the partition silently leaks one entity's history into another's features.

The whole script, executed on every test run:

```{literalinclude} ../../../../examples/ml/feature_construction.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/ml/feature_construction.py
```

## See also

- {doc}`/cookbook/ml/estimators/clustering_and_decomposition`: KMeans, Gaussian mixtures, PCA, and truncated SVD.
- {doc}`/cookbook/ml/estimators/glm_regressors`: generalized linear models for counts, costs, and mixed zero-and-positive targets.
- {doc}`/ml/index`: the ML surface these recipes sit on.
- {doc}`/ml/preparing/preprocessors/index`: the fit and transform steps most pipelines start with.
