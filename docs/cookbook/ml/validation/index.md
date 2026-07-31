# Checking what you fitted

The three questions that decide whether a score is real: did it generalize, was the data balanced, and is anything in it an artifact.

Each page embeds a complete, self-contained script that builds its own in-memory data and asserts on its own output, so a page that stops matching the engine fails the suite instead of drifting.

| Recipe | What it shows |
|---|---|
| {doc}`/cookbook/ml/validation/model_selection` | Cross-validation, learning curves, and feature importance, all in the engine |
| {doc}`/cookbook/ml/validation/imbalance_and_sampling` | Measuring class imbalance, then resampling or reweighting |
| {doc}`/cookbook/ml/validation/outlier_detection` | Per-column rules and a multivariate distance |

```{toctree}
:hidden:

model_selection
imbalance_and_sampling
outlier_detection
```
