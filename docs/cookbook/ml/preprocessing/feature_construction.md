# Feature construction

These featurizers turn a raw table into a model-ready one. The time-series ones, `LagFeaturizer` and `RollingFeaturizer`, need an `order_by` and usually a `partition_by`: forget the partition and one entity's history silently leaks into another's features.

The script builds polynomial, interaction, and ratio features, calendar parts and a cyclical hour encoding, per-user lags and rolling means, group statistics, and text statistics. It finishes with `VarianceThreshold` dropping a column that never varies.

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
