# Checking what you fitted

A score is only worth reporting once three questions have answers. Did the model generalize beyond the rows it saw? Was the training data balanced, or did the majority class do the work? And is anything in the data an artifact that the model learned instead of the signal? Each recipe answers one of them, and all three run through the engine.

| Recipe | What it shows |
|---|---|
| {doc}`/cookbook/ml/validation/model_selection` | Cross-validation, learning curves, and feature importance, all in the engine |
| {doc}`/cookbook/ml/validation/imbalance_and_sampling` | Measuring class imbalance, then resampling or reweighting |
| {doc}`/cookbook/ml/validation/outlier_detection` | Per-column rules and a multivariate distance |

## See also

- {doc}`/cookbook/ml/pipelines/features/train-test-split`: a split that does not leak, and stays the same on the next run.
- {doc}`/cookbook/metrics/model/index`: the metrics a cross-validation loop scores with.
- {doc}`/ml/evaluation/model-selection`: cross-validation and hyperparameter search, in full.

```{toctree}
:hidden:

model_selection
imbalance_and_sampling
outlier_detection
```
