# Measure what the model does

This section covers scoring predictions against labels, comparing and tuning models, building splits you can trust, and watching a deployed model's inputs for drift.

A model is only as trustworthy as the numbers around it, and in Batcher those numbers are queries. Accuracy, F1, log loss and the rest are expressions the engine evaluates inside an aggregate, so a report over a billion scored rows never lands on a driver, and ten metrics cost the same single pass as one. Because they are aggregates, they also group. The question a model review actually asks, "how does it do per region, per month", is the same query with a grouping added.

## One pass, overall and per segment

{py:meth}`ds.ml.evaluate <batcher.api.dataset.ml.DatasetML.evaluate>` runs a task's metric set. With `by=` it returns one row per group instead of a dict:

```python
import batcher as bt

ds = bt.from_pydict(
    {
        "region": ["eu", "eu", "eu", "us", "us", "us"],
        "y": [1, 0, 1, 1, 0, 0],
        "score": [0.9, 0.2, 0.7, 0.3, 0.6, 0.1],
    }
)
print(ds.ml.evaluate("y", y_score="score", metrics=["accuracy", "roc_auc"]))
# {'accuracy': 0.6666666666666666, 'roc_auc': 0.8888888888888888}
print(ds.ml.evaluate("y", y_score="score", by="region", metrics=["accuracy", "roc_auc"]).sort("region").to_pydict())
# {'region': ['eu', 'us'], 'accuracy': [1.0, 0.3333333333333333], 'roc_auc': [1.0, 0.5]}
```

The overall accuracy of 0.67 hides a segment that is perfect and one that is worse than a coin flip. The grouped form is the one to run first. Individual metrics such as {py:func}`bt.accuracy <batcher.accuracy>` also work inside any `agg` or `group_by`, next to the business measures you already compute.

## Splits that don't leak

The random splits assign each row by a hash of its content, or of the `key=` columns you name, rather than by a materialized shuffle. A fold is therefore an ordinary row-wise filter. It streams, it distributes, and a row lands in the same fold however the data is partitioned or reordered. {py:meth}`ds.ml.kfold <batcher.api.dataset.ml.DatasetML.kfold>` takes `stratify=` to hold each label's share constant across folds and `group=` to keep every row of one entity on the same side. For time series, {py:meth}`ds.ml.time_series_split <batcher.api.dataset.ml.DatasetML.time_series_split>` cuts on time instead, so a model never trains on the future.

## Model selection and drift

{py:func}`cross_val_score <batcher.ml.cross_val_score>`, {py:func}`grid_search <batcher.ml.grid_search>` and {py:func}`random_search <batcher.ml.random_search>` take `fit` and `predict` as callables, so a Batcher estimator, a scikit-learn model, or a closure around a whole preprocessing chain all plug in the same way.

Once a model is deployed and labels haven't arrived, the inputs are the only thing you can watch. {py:meth}`ds.ml.drift <batcher.api.dataset.ml.DatasetML.drift>` compares today's data against a reference, with bin edges taken from the reference so a shift shows up as mass moving between bins. `batcher.ml.stats` adds the population stability index, Jensen-Shannon divergence, hypothesis tests with p-values, and outlier detection, each as a pass over the data rather than a sample.

## In this section

The following table lists the pages in this section:

| Page | Covers |
|---|---|
| {doc}`/ml/evaluation/evaluation` | Task metric sets, per-segment scoring, diagnostic tables, operating points, fairness, and calibration. |
| {doc}`/ml/evaluation/model-selection` | Cross-validation, grid and random search, reading a whole search, and learning curves. |
| {doc}`/ml/evaluation/splits-and-resampling` | Rebalancing an imbalanced label, stratified hold-outs, and the fold assignment underneath them. |
| {doc}`/ml/evaluation/statistics-and-drift` | Robust statistics, feature profiling and screening, outliers, drift monitoring, and hypothesis tests. |

## See also

- {doc}`/ml/inference/calibration`: fixing a classifier whose probabilities are miscalibrated.
- {doc}`/ml/retrieval/llm-evaluation`: scoring generated text rather than labels.
- {doc}`/cookbook/ml/validation/index`: short validation recipes.
- {doc}`/api/models/metrics`: the complete metric reference.

```{toctree}
:hidden:

evaluation
model-selection
splits-and-resampling
statistics-and-drift
```
