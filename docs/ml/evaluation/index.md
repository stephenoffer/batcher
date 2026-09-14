# Measure what the model does

A model is only as trustworthy as the numbers around it, and those numbers are queries.
Every metric here is an expression the engine evaluates, so a report over a billion scored
rows is one pass and the same report *per segment* costs the same. Nothing lands on a driver.

- {doc}`/ml/evaluation/evaluation`: metrics, per-segment scoring, and the diagnostic tables.
- {doc}`/ml/evaluation/model-selection`: cross-validation, grid and random search, learning curves.
- {doc}`/ml/evaluation/splits-and-resampling`: class rebalancing, stratified hold-outs, and the fold assignment underneath them.
- {doc}`/ml/evaluation/statistics-and-drift`: statistical expressions, outliers, hypothesis tests, and input drift.

```{toctree}
:hidden:

evaluation
model-selection
splits-and-resampling
statistics-and-drift
```
