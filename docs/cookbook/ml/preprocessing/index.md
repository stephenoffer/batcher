# Preparing the features

Most of the work in a tabular model happens before the fit. These recipes cover the preprocessors that get a raw table into shape: scalers, encoders, imputers, and binning, each with the same `fit` and `transform` split that keeps training statistics out of your validation set.

Read {doc}`preprocessing_chain` once you know the individual steps. It turns them into one fitted object you apply everywhere. The last two pages manufacture features the source never carried, from timestamps, groups, ratios, and raw text.

| Recipe | What it shows |
|---|---|
| {doc}`/cookbook/ml/preprocessing/preprocessing_scaling` | Scaling numeric features, and why the choice of scaler matters |
| {doc}`/cookbook/ml/preprocessing/preprocessing_encoding` | Turning categories into numbers, and picking the encoder by cardinality |
| {doc}`/cookbook/ml/preprocessing/preprocessing_imputation` | Filling missing values, and keeping the fact that they were missing |
| {doc}`/cookbook/ml/preprocessing/preprocessing_binning` | Discretizing, clipping, and reshaping a numeric distribution |
| {doc}`/cookbook/ml/preprocessing/preprocessing_chain` | Chaining preprocessors into one fitted pipeline |
| {doc}`/cookbook/ml/preprocessing/feature_construction` | Interactions, ratios, calendar parts, lags, and rolling windows |
| {doc}`/cookbook/ml/preprocessing/text_features` | Turning raw text into model-ready features without a model |

## See also

- {doc}`/cookbook/ml/estimators/index`: the models these features feed.
- {doc}`/cookbook/ml/pipelines/features/feature-pipeline`: a complete feature pipeline built from these steps.
- {doc}`/ml/preparing/preprocessors/index`: the preprocessor guide, in full.

```{toctree}
:hidden:

preprocessing_scaling
preprocessing_encoding
preprocessing_imputation
preprocessing_binning
preprocessing_chain
feature_construction
text_features
```
