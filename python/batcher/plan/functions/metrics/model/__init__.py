"""Metrics that score a model's predictions against known labels.

Six families, grouped by what the prediction is: `classification` for discrete labels and the
confusion-matrix counts behind them, `diagnostic` for the ratio measures a screening or
medical-test setting reports, `errors` for continuous targets, `agreement` for how well a
predicted series tracks an observed one, `losses` for the training objectives themselves, and
`embedding` for predictions that are vectors.

Every name here is an aggregate `Expr`, so it belongs inside `agg()` — which is what turns
"evaluate the model" into "evaluate the model per segment, per day, per cohort" at no extra cost.
The metrics needing a global ordering (ROC AUC, PR AUC, KS) are Dataset functions in
`batcher.ml.metrics` instead, because a rank is not an aggregate.

The public names are re-exported by the parent `metrics` facade and reachable as `bt.<name>`;
this package is the organization, not a second import path.
"""
