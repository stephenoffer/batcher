# Calibrate a classifier's scores

This page describes how to turn a classifier's raw scores into probabilities you can make decisions against, how to check that the result is actually calibrated, and how to calibrate a score column produced by a model Batcher didn't train.

## Why ranking well is not the same as being calibrated

A classifier that ranks rows well can still be wrong about *how* confident it is. Take a model whose 0.9 scores come true only half the time. It orders rows correctly. It also misprices every decision made against a threshold. That bites as soon as the score meets a cost, such as a fraud review queue sized by expected loss or a bid multiplied by a click probability.

A *calibrator* fixes this by learning a monotone mapping from score to probability. Monotone matters. The mapping never reorders rows, so ranking metrics such as AUC are unchanged and only the numbers attached to the ranks move.

## Calibrate while you fit

{py:class}`CalibratedClassifierCV <batcher.ml.compose.calibration.CalibratedClassifierCV>` wraps a classifier and calibrates it on scores the classifier never trained on. It splits the data into `cv` stratified folds and fits one model per fold on the other folds. Each fold model scores its held-out rows, and the calibrator is fitted once on all of those pooled out-of-fold scores. The model that ships is then refitted on every row, so calibration costs no training data.

Fitting the calibrator on the training rows instead is the mistake this class exists to prevent. A model scores its own training rows better than anything else, so the calibrator learns to correct a distortion that won't be there at serving time. Measured on 300 rows with twenty features, the expected calibration error was 0.102 for the raw scores and 0.146 after calibrating in-sample. That is worse than doing nothing.

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

`predict_proba` appends the calibrated probability, named by `output_column` and `calibrated` by default. `predict` does the same and adds a 0/1 `prediction` column thresholded at 0.5.

Two curves are available through `method`. The default, `method="sigmoid"`, is Platt scaling: a two-parameter logistic fit that is hard to overfit, which makes it the right choice when the folds are small. `method="isotonic"` fits a free-form non-decreasing step function instead. It corrects shapes a sigmoid can't, and it needs considerably more data per fold before it stops fitting noise.

## Check that it worked

Calibration is a claim about held-out data, so measure it there. {py:func}`expected_calibration_error <batcher.ml.metrics.expected_calibration_error>` bins the predictions by confidence and averages the gap between predicted probability and observed positive rate, weighted by how many predictions each bin holds. {py:func}`calibration_curve <batcher.ml.metrics.calibration_curve>` returns the per-bin table behind that number, which is what a reliability diagram plots:

```python
from batcher.ml.metrics import calibration_curve, expected_calibration_error

scored = model.predict_proba(ds)
ece = expected_calibration_error(scored, "label", "calibrated", bins=5)
print(0.0 <= ece <= 1.0)
# True
print("observed_rate" in calibration_curve(scored, "label", "calibrated", bins=5).columns)
# True
```

The example scores the rows it trained on to stay short. In a real pipeline, split first with {py:meth}`ds.ml.train_test_split <batcher.api.dataset.ml.DatasetML.train_test_split>` and measure on the test part. {doc}`/ml/evaluation/evaluation` covers `brier_score` and `log_loss`, the two metrics that reward calibration directly.

## Calibrate a score you already have

Sometimes the model isn't a Batcher estimator at all: an XGBoost booster scored with {py:meth}`ds.ml.predict <batcher.api.dataset.ml.DatasetML.predict>`, or a column that arrived with the data. {py:class}`PlattCalibrator <batcher.ml.preprocessors.PlattCalibrator>` and {py:class}`IsotonicCalibrator <batcher.ml.preprocessors.IsotonicCalibrator>` are the two curves `CalibratedClassifierCV` uses internally, exposed as ordinary preprocessors. Each takes the score column and the label column, learns the mapping in `fit`, and appends the calibrated column in a lazy `transform`:

```python
from batcher.ml.preprocessors import PlattCalibrator

held_out = bt.from_pydict(
    {
        "score": [0.9, 0.8, 0.85, 0.7, 0.95, 0.2, 0.3, 0.6, 0.1, 0.75],
        "label": [1, 0, 1, 0, 1, 0, 0, 1, 0, 0],
    }
)
calibrator = PlattCalibrator("score", "label").fit(held_out)
print(calibrator.transform(held_out).columns)
# ['score', 'label', 'calibrated']
```

Fit the calibrator on rows the model didn't train on, for the same reason as above. A validation split the model never saw is the usual source. `IsotonicCalibrator` takes `n_bins`, default 100, and `PlattCalibrator` takes `max_iter`, default 100.

## Save and reload

The fitted mapping is part of the model. {py:func}`save_model <batcher.ml.save_model>` writes a `CalibratedClassifierCV` with its calibrator nested inside, and a model restored with {py:func}`load_model <batcher.ml.load_model>` calibrates exactly as the saved one did. A standalone calibrator persists like any other preprocessor, as {doc}`/ml/preparing/preprocessors/pipelines` describes.

## Requirements and limitations

- The target is a binary label. `positive` on the standalone calibrators names the positive class, default `1`.
- The wrapped estimator must expose `predict_proba`. A classifier that only emits hard 0/1 labels raises {py:exc}`PlanError <batcher.PlanError>` at construction, because there is no score to calibrate.
- `cv` must be at least 2, so every row is scored by a model that didn't train on it.
- The estimator class must be exported from `batcher.ml`, or a saved model can't name it well enough to rebuild.

## See also

- {doc}`/ml/inference/tabular-models`: fitting and scoring the classifier being calibrated.
- {doc}`/ml/evaluation/evaluation`: the metrics that tell you whether it worked.
- {doc}`/ml/evaluation/splits-and-resampling`: the held-out split a calibrator should be fitted on.
- {doc}`/ml/training/ensembling`: blending several models, whose combined score is often worth calibrating.
