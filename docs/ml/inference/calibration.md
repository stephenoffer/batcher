# Calibrate a classifier's scores

This page describes how to turn a classifier's raw scores into probabilities you can
make decisions against, with
{py:class}`CalibratedClassifierCV <batcher.ml.compose.calibration.CalibratedClassifierCV>`.

## Why separation is not calibration

A classifier that ranks rows well can still be wrong about *how* confident it is. A model whose 0.9 scores come true only half the time orders rows correctly and misprices every decision made against a threshold, which matters as soon as the score meets a cost.

{py:class}`CalibratedClassifierCV <batcher.ml.compose.calibration.CalibratedClassifierCV>` learns the mapping from score to probability. It splits the data into `cv` folds, fits the classifier on all but one, fits the mapping on the fold that model never saw, then averages those mappings and refits the classifier on everything. Calibrating on the training split instead learns the overconfidence the model shows on rows it memorized, which looks perfect in development and is wrong in use.

Pass the estimator as a class, as with `OneVsRestClassifier`, because it fits one per fold:

```python
import batcher as bt
from batcher.ml import CalibratedClassifierCV, LogisticRegression

ds = bt.from_pydict(
    {
        "x": [0.1, 0.4, 0.35, 0.8, 0.9, 0.2, 0.75, 0.6, 0.05, 0.95],
        "z": [1.0, 2.0, 1.5, 3.5, 4.0, 1.2, 3.0, 2.5, 0.5, 4.5],
        "label": [0, 0, 0, 1, 1, 0, 1, 1, 0, 1],
    }
)
model = CalibratedClassifierCV(LogisticRegression, ["x", "z"], "label", cv=2).fit(ds)
print(sorted(model.predict(ds).columns)[:1])
# ['calibrated']
```

`method="sigmoid"` is Platt scaling, a two-parameter fit that suits a small held-out fold; `method="isotonic"` fits a free monotone step function, which corrects shapes a sigmoid cannot but needs considerably more data per fold before it stops fitting noise.

The fitted mapping is part of the model, so `save_model` writes it alongside the classifier and a loaded model calibrates exactly as the saved one did.

## See also

- {doc}`/ml/inference/tabular-models`: fitting and scoring the classifier being calibrated.
- {doc}`/ml/evaluation/evaluation`: the metrics that tell you whether it worked.
