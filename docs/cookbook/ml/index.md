# Machine learning cookbook

Preprocessors, estimators, model selection, batch inference, and vector search on the `batcher.ml` surface.

Every page here embeds a complete, self-contained script from the
[`examples/ml/`](https://github.com/batcher/batcher/tree/main/examples/ml) directory.
The scripts build their own in-memory data and assert on their own output, and
`tests/docs/test_examples.py` runs all of them, so a page that stops matching the
engine fails the suite instead of drifting.

| Recipe | What it shows |
|---|---|
| {doc}`batch_inference` | Batch inference: a model over every row, without a Python loop |
| {doc}`classifiers` | Classifiers that fit in the engine: naive Bayes, discriminant analysis, and baselines |
| {doc}`clustering_and_decomposition` | Unsupervised models: KMeans, Gaussian mixtures, PCA, and truncated SVD |
| {doc}`feature_construction` | Building new features: interactions, ratios, calendar parts, lags, and rolling windows |
| {doc}`glm_regressors` | Generalized linear models for counts, costs, and mixed zero-and-positive targets |
| {doc}`imbalance_and_sampling` | Class imbalance: measure it, then resample or reweight |
| {doc}`linear_models` | Regularized linear regression: Ridge, Lasso, and ElasticNet |
| {doc}`model_selection` | Cross-validation, learning curves, and feature importance -- all in the engine |
| {doc}`outlier_detection` | Finding outliers: per-column rules and a multivariate distance |
| {doc}`preprocessing_binning` | Discretizing, clipping, and reshaping the distribution of a numeric column |
| {doc}`preprocessing_chain` | Chaining preprocessors into one fitted pipeline |
| {doc}`preprocessing_encoding` | Turning categories into numbers, and picking the encoder by cardinality |
| {doc}`preprocessing_imputation` | Filling missing values, and keeping the fact that they were missing |
| {doc}`preprocessing_scaling` | Scaling numeric features, and why the choice of scaler matters |
| {doc}`text_features` | Turning raw text into model-ready features without a model |
| {doc}`vector_search` | Vector search over an embedding column, in the engine |

```{toctree}
:hidden:

batch_inference
classifiers
clustering_and_decomposition
feature_construction
glm_regressors
imbalance_and_sampling
linear_models
model_selection
outlier_detection
preprocessing_binning
preprocessing_chain
preprocessing_encoding
preprocessing_imputation
preprocessing_scaling
text_features
vector_search
```
