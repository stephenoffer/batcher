# Measure what the model does

This section covers scoring predictions against labels, comparing and tuning models, building splits you can trust, and watching a deployed model's inputs for drift.

In Batcher model metrics are queries. Accuracy, F1, log loss and the rest are expressions evaluated inside an aggregate, so a report over a billion scored rows never lands on a driver, ten metrics cost one pass, and "how does it do per region" is the same query with a grouping added.

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

The overall accuracy of 0.67 hides a segment that is perfect and one that is worse than a coin flip. Run the grouped form first. Individual metrics such as {py:func}`bt.accuracy <batcher.accuracy>` also work inside any `agg` or `group_by`:

```python
pred = (bt.col("score") > 0.5).cast("int64")
print(ds.agg(acc=bt.accuracy("y", pred)).to_pydict())
# {'acc': [0.6666666666666666]}
```

## Splits that don't leak

The random splits assign each row by a hash of its content, or of the `key=` columns you name, so a fold is an ordinary row-wise filter. It streams, distributes, and puts a row in the same fold however the data is partitioned:

```python
folds = ds.ml.kfold(3, seed=0)
print([(train.count(), test.count()) for train, test in folds])
# [(4, 2), (5, 1), (3, 3)]
```

{py:meth}`ds.ml.kfold <batcher.api.dataset.ml.DatasetML.kfold>` takes `stratify=` to hold each label's share constant and `group=` to keep one entity's rows on the same side. {py:meth}`ds.ml.time_series_split <batcher.api.dataset.ml.DatasetML.time_series_split>` cuts on time, so a model never trains on the future.

## Model selection and drift

{py:func}`cross_val_score <batcher.ml.cross_val_score>`, {py:func}`grid_search <batcher.ml.grid_search>` and {py:func}`random_search <batcher.ml.random_search>` take `fit` and `predict` as callables, so a Batcher estimator, a scikit-learn model, or a whole preprocessing chain plug in the same way.

Before labels arrive, the inputs are what you can watch. {py:meth}`ds.ml.drift <batcher.api.dataset.ml.DatasetML.drift>` compares today's data against a reference, one row per column:

```python
reference = bt.from_pydict({"x": [float(i) for i in range(100)]})
today = bt.from_pydict({"x": [float(i) + 50 for i in range(100)]})
report = today.ml.drift(reference, ["x"]).to_pydict()
print(report["column"], round(report["psi"][0], 2), report["mean_shift"])
# ['x'] 6.65 [50.0]
```

`batcher.ml.stats` adds Jensen-Shannon divergence, hypothesis tests with p-values, and outlier detection, each as a pass over the data rather than a sample.

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
