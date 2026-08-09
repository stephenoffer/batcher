"""Probability calibration — turning a model's scores into numbers that mean what they say.

`batcher.ml.metrics` can already tell you a model is miscalibrated (`expected_calibration_error`,
`calibration_curve`, `brier_skill_score`). These are the two standard ways to fix it.

A score between 0 and 1 is not a probability. A boosted tree that says 0.9 is usually right
more often than 90% of the time, and an SVM's decision value is not on a probability scale
at all. It only matters once the number is used for something other than ranking — a
cost-sensitive threshold, an expected-value calculation, a downstream model that reads it as
a rate — and none of those fail loudly when the number is wrong.

Both calibrators must be fitted on a split the model did **not** train on. Calibrating on
the training split measures the model's confidence on data it memorized, which produces a
calibration curve that looks perfect in development and is wrong in use.

`PlattCalibrator`
    Two parameters, so it needs little data, at the cost of assuming the miscalibration is
    sigmoid-shaped. The default choice on a small calibration split.
`IsotonicCalibrator`
    Assumes only that a higher score should not mean a lower probability. More faithful when
    the distortion is not sigmoid, and it wants a few thousand rows rather than a few
    hundred.
"""

from __future__ import annotations

from batcher.ml.preprocessors.calibration.isotonic import IsotonicCalibrator
from batcher.ml.preprocessors.calibration.platt import PlattCalibrator

__all__ = ["IsotonicCalibrator", "PlattCalibrator"]
