# Preparing the features

The fit/transform preprocessors, plus the two ways to manufacture a feature the source did not carry.

Each page embeds a complete, self-contained script that builds its own in-memory data and asserts on its own output, so a page that stops matching the engine fails the suite instead of drifting.

| Recipe | What it shows |
|---|---|
| {doc}`/cookbook/ml/preprocessing/preprocessing_scaling` | Scaling numeric features, and why the choice of scaler matters |
| {doc}`/cookbook/ml/preprocessing/preprocessing_encoding` | Turning categories into numbers, and picking the encoder by cardinality |
| {doc}`/cookbook/ml/preprocessing/preprocessing_imputation` | Filling missing values, and keeping the fact that they were missing |
| {doc}`/cookbook/ml/preprocessing/preprocessing_binning` | Discretizing, clipping, and reshaping a numeric distribution |
| {doc}`/cookbook/ml/preprocessing/preprocessing_chain` | Chaining preprocessors into one fitted pipeline |
| {doc}`/cookbook/ml/preprocessing/feature_construction` | Interactions, ratios, calendar parts, lags, and rolling windows |
| {doc}`/cookbook/ml/preprocessing/text_features` | Turning raw text into model-ready features without a model |

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
