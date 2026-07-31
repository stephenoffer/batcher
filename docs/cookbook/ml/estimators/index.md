# Fitting a model in the engine

Estimators that run as engine operators, so training does not pull the table into Python.

Each page embeds a complete, self-contained script that builds its own in-memory data and asserts on its own output, so a page that stops matching the engine fails the suite instead of drifting.

| Recipe | What it shows |
|---|---|
| {doc}`/cookbook/ml/estimators/linear_models` | Regularized linear regression: Ridge, Lasso, and ElasticNet |
| {doc}`/cookbook/ml/estimators/glm_regressors` | Counts, costs, and mixed zero-and-positive targets |
| {doc}`/cookbook/ml/estimators/classifiers` | Naive Bayes, discriminant analysis, and baselines |
| {doc}`/cookbook/ml/estimators/clustering_and_decomposition` | KMeans, Gaussian mixtures, PCA, and truncated SVD |

```{toctree}
:hidden:

linear_models
glm_regressors
classifiers
clustering_and_decomposition
```
