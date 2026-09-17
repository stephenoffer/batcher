# Fitting a model in the engine

Batcher's estimators fit from a Dataset and predict into one. Training reads the table through the engine, and `predict` appends a column to a new Dataset, so a fitted model drops into any pipeline next to a filter or a join.

Every estimator shares the same `fit` and `predict` shape. Learn it once on the linear models, then pick by target: counts and costs, class labels, or no label at all.

| Recipe | What it shows |
|---|---|
| {doc}`/cookbook/ml/estimators/linear_models` | Regularized linear regression: Ridge, Lasso, and ElasticNet |
| {doc}`/cookbook/ml/estimators/glm_regressors` | Counts, costs, and mixed zero-and-positive targets |
| {doc}`/cookbook/ml/estimators/classifiers` | Naive Bayes, discriminant analysis, and baselines |
| {doc}`/cookbook/ml/estimators/clustering_and_decomposition` | KMeans, Gaussian mixtures, PCA, and truncated SVD |

## See also

- {doc}`/cookbook/ml/preprocessing/index`: preparing the features these models consume.
- {doc}`/cookbook/ml/validation/index`: checking a fitted model before you trust its score.
- {doc}`/cookbook/metrics/model/index`: the metrics that score a prediction column.
- {doc}`/api/models/ml`: the `batcher.ml` reference.

```{toctree}
:hidden:

linear_models
glm_regressors
classifiers
clustering_and_decomposition
```
