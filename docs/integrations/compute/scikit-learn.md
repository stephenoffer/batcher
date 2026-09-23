# scikit-learn

This page covers scoring a fitted scikit-learn model over a Batcher dataset, scoring the result, and when to use Batcher's own in-engine estimators instead.

The integration is the scikit-learn *contract* rather than the library: {py:obj}`ds.ml.predict <batcher.api.dataset.ml.DatasetML.predict>` calls `predict`, `predict_proba`, `decision_function`, or `transform` on whatever object you hand it. Anything implementing that contract works, which covers scikit-learn, its pipelines, XGBoost, LightGBM, CatBoost, and most of what wraps them.

| | |
| --- | --- |
| **Score** | {py:obj}`ds.ml.predict(model, features=[...]) <batcher.api.dataset.ml.DatasetML.predict>` |
| **Extra** | `sklearn` |
| **Runs** | In the worker process, batch by batch; never one row at a time |

## Score a fitted model

Fit however you already fit. Batcher enters at the point where the model has to meet more rows than fit in memory:

```python
import batcher as bt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(0)
features = np.vstack([rng.normal(0, 1, (40, 2)), rng.normal(4, 1, (40, 2))])
labels = np.r_[np.zeros(40, int), np.ones(40, int)]
model = make_pipeline(StandardScaler(), LogisticRegression()).fit(features, labels)

rows = bt.from_pydict({"id": [1, 2, 3], "f0": [0.0, 4.0, 0.2], "f1": [0.1, 3.8, 0.0]})
print(rows.ml.predict(model, features=["f0", "f1"], output_column="label").to_pydict())
# {'id': [1, 2, 3], 'f0': [0.0, 4.0, 0.2], 'f1': [0.1, 3.8, 0.0], 'label': [0, 1, 0]}
```

A whole `Pipeline` is one object to the contract, so the scaler travels with the classifier and training-time preprocessing cannot drift from scoring-time preprocessing.

`method=` picks a different call, and `as_list=True` keeps a multi-column output as one list column:

```python
scored = rows.ml.predict(
    model, features=["f0", "f1"], method="predict_proba", output_column="p", as_list=True
)
print(scored.select("id", "p").to_pydict()["p"][1])
# [0.0256762190611185, 0.9743237809388815]
```

## Why this is not `map_batches`

You could call the model inside a batch UDF. `ds.ml.predict` exists because doing that well means solving four problems that have nothing to do with your model: assembling the feature matrix in the right column order, keeping it out of Python object space, sizing the batch against the model's memory, and reusing one loaded model across batches rather than pickling it per call.

It also takes the scaling arguments — `num_workers`, `num_gpus`, `concurrency`, `batch_size` — so moving from one process to a cluster is an argument rather than a rewrite. {doc}`/ml/inference/index` covers them.

## Scoring the scores

Metrics are aggregate expressions, so evaluating a model is a `select`, and evaluating it per segment is the same expression under a `group_by`, still in one pass:

```python
truth = bt.from_pydict({"y": [0, 1, 0, 1, 1], "pred": [0, 1, 1, 1, 0]})
print(truth.select(acc=bt.accuracy("y", "pred"), f1=bt.f1_score("y", "pred")).to_pydict())
# {'acc': [0.6], 'f1': [0.6666666666666666]}
```

{doc}`/api/models/metrics` lists the full vocabulary, and {doc}`/ml/evaluation/evaluation` covers per-segment scoring and the report shape.

## When to use Batcher's own estimators instead

Batcher ships in-engine estimators — {py:obj}`bt.ml.LinearRegression <batcher.ml.LinearRegression>`, {py:obj}`bt.ml.KMeans <batcher.ml.KMeans>`, and the preprocessors — that fit over a `Dataset` without materializing it. Reach for them when the *training* data does not fit in memory, because scikit-learn's `fit` takes an array that does.

Reach for scikit-learn when it has an estimator Batcher does not, when you need its exact numerics, or when the model is already fitted and in production. The two compose: fit in scikit-learn, score with `ds.ml.predict`, or fit in Batcher and export.

{doc}`/api/models/ml-models` lists the in-engine estimators, and {doc}`/api/models/preprocessors` the fit/transform surface.

## See also

- {doc}`gradient-boosting`: XGBoost and LightGBM through the same contract.
- {doc}`/ml/inference/tabular-models`: scoring a tabular model at length, including error handling.
- {doc}`/ml/preparing/preprocessors/index`: feature preparation that stays in the plan.
- {doc}`mlflow`: loading a logged model and scoring it the same way.
