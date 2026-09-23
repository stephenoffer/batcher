# XGBoost, LightGBM, and CatBoost

This page covers scoring a gradient-boosted tree model over a Batcher dataset, and the one argument that decides whether it is fast.

All three libraries expose the scikit-learn contract, so they go through {py:obj}`ds.ml.predict <batcher.api.dataset.ml.DatasetML.predict>` exactly as a scikit-learn estimator does. Nothing about the model is special to Batcher; what is worth reading is how the batches reach it.

| | |
| --- | --- |
| **Score** | {py:obj}`ds.ml.predict(model, features=[...]) <batcher.api.dataset.ml.DatasetML.predict>` |
| **Extras** | `xgboost`, `lightgbm`, `catboost` |
| **Booster objects** | Wrap in the library's scikit-learn class (`XGBClassifier`, `LGBMClassifier`), or pass `method=` naming the call you want |

## Score a booster

```python
import batcher as bt
import numpy as np
import lightgbm as lgb
import xgboost as xgb

rng = np.random.default_rng(0)
features = np.vstack([rng.normal(0, 1, (60, 3)), rng.normal(4, 1, (60, 3))])
labels = np.r_[np.zeros(60, int), np.ones(60, int)]

boosted = xgb.XGBClassifier(n_estimators=20, max_depth=3, verbosity=0).fit(features, labels)
leafy = lgb.LGBMClassifier(n_estimators=20, verbose=-1).fit(features, labels)

rows = bt.from_pydict({"id": [1, 2], "f0": [0.0, 4.0], "f1": [0.1, 3.9], "f2": [0.0, 4.2]})
names = ["f0", "f1", "f2"]

print(rows.ml.predict(boosted, features=names, output_column="label").to_pydict()["label"])
# [0, 1]
print(rows.ml.predict(leafy, features=names, output_column="label").to_pydict()["label"])
# [0, 1]
```

Probabilities come back as a list column, one entry per class:

```python
scored = rows.ml.predict(
    boosted, features=names, method="predict_proba", output_column="p", as_list=True
)
print([round(p[1], 3) for p in scored.to_pydict()["p"]])
# [0.016, 0.989]
```

## Batch size is the whole performance story

A boosted tree scores a batch in native code and returns almost immediately, so the cost is dominated by the per-call overhead rather than by the trees. Scoring 128 rows at a time spends most of its time in Python; scoring 100,000 at a time spends almost none.

`batch_size=` sets it. The default adapts to the batches the scan produces, which is usually right; raise it when the model is small and the rows are narrow, and lower it when a row carries a wide feature vector and memory is the binding constraint.

```python
# docs: skip
scored = rows.ml.predict(boosted, features=names, batch_size=100_000)
```

This is the opposite of the trade for a large neural network, where the batch is bounded by device memory. {doc}`/ml/inference/gpu` covers that case.

## Feature order is the column list

`features=` is the order the matrix columns are assembled in, and it must match the order the model was trained on. A model trained on `[f0, f1, f2]` and scored on `[f0, f2, f1]` returns confident nonsense rather than an error, because nothing in the array carries a name.

Keep the list in one place and reuse it for training and scoring, as the example above does. If the model records its own feature names, assert against them before scoring rather than after.

## A raw `Booster`

`xgb.Booster` and `lgb.Booster` are the native handles and do not implement `predict` the same way. Either keep the scikit-learn wrapper class, which is what `fit` returns anyway, or name the call with `method=`. The wrapper is the simpler answer and costs nothing at score time.

## See also

- {doc}`scikit-learn`: the same contract, and Batcher's own in-engine estimators.
- {doc}`/ml/inference/tabular-models`: tabular scoring at length, with error handling and retries.
- {doc}`/ml/inference/index`: the scaling arguments, from one process to a cluster.
- {doc}`mlflow`: loading a logged booster and scoring it the same way.
