"""`CalibratedClassifierCV` — calibrating on scores the model did not train on.

A calibrator has to be fitted on scores paired with labels, and the obvious source is the
training data. That is where it goes wrong: a model's scores on its own rows are better than
its scores on anything else, so the calibrator learns to correct a distortion that will not
be there at serving time.

The failure is not subtle once measured, and it is completely invisible otherwise - the
calibrator fits its own data beautifully either way. The first two tests establish it: raw
held-out calibration error, then the *worse* number after calibrating in-sample. Only then is
there anything for this class to fix.
"""

from __future__ import annotations

import numpy as np
import pytest

import batcher as bt
from batcher._internal.errors import ColumnNotFoundError, PlanError
from batcher.ml import CalibratedClassifierCV, LogisticRegression, PlattCalibrator
from batcher.ml.metrics import expected_calibration_error

pytestmark = pytest.mark.unit

FEATURES = [f"f{i}" for i in range(20)]


@pytest.fixture(scope="module")
def split() -> tuple[bt.Dataset, bt.Dataset]:
    """Twenty features and 300 rows, so the model overfits its own scores."""
    rng = np.random.default_rng(0)
    n, d = 300, 20
    x = rng.normal(size=(n, d))
    weights = np.zeros(d)
    weights[0] = 1.5
    probability = 1.0 / (1.0 + np.exp(-(x @ weights)))
    y = (rng.random(n) < probability).astype(int)
    full = bt.from_pydict(
        {**{name: x[:, i].tolist() for i, name in enumerate(FEATURES)}, "y": y.tolist()}
    )
    return full.ml.train_test_split(test_size=0.4, seed=1, stratify="y")


def _ece(ds: bt.Dataset, column: str) -> float:
    return expected_calibration_error(ds, "y", column, bins=5)


def test_calibrating_in_sample_makes_held_out_calibration_worse(split) -> None:
    """The premise, measured. Without it the rest of this file fixes nothing."""
    train, test = split
    model = LogisticRegression(FEATURES, "y", max_iter=200).fit(train)
    scored = model.predict_proba(test)
    raw = _ece(scored, "prediction")
    naive = PlattCalibrator("prediction", "y").fit(model.predict_proba(train))
    after = _ece(naive.transform(scored), "calibrated")
    assert after > raw, f"expected in-sample calibration to hurt: raw={raw}, after={after}"


def test_out_of_fold_calibration_beats_calibrating_in_sample(split) -> None:
    train, test = split
    model = LogisticRegression(FEATURES, "y", max_iter=200).fit(train)
    naive = PlattCalibrator("prediction", "y").fit(model.predict_proba(train))
    in_sample = _ece(naive.transform(model.predict_proba(test)), "calibrated")

    cross = CalibratedClassifierCV(
        LogisticRegression, FEATURES, "y", cv=5, params={"max_iter": 200}
    ).fit(train)
    assert _ece(cross.predict_proba(test), "calibrated") < in_sample


def test_out_of_fold_calibration_beats_not_calibrating_at_all(split) -> None:
    """The bar that matters: it has to be worth doing, not merely better than the bad way."""
    train, test = split
    raw = _ece(
        LogisticRegression(FEATURES, "y", max_iter=200).fit(train).predict_proba(test), "prediction"
    )
    cross = CalibratedClassifierCV(
        LogisticRegression, FEATURES, "y", cv=5, params={"max_iter": 200}
    ).fit(train)
    assert _ece(cross.predict_proba(test), "calibrated") < raw


def test_the_shipped_model_is_fitted_on_everything(split) -> None:
    """The fold models exist only to make honest scores; the final fit uses all the data."""
    train, _ = split
    cross = CalibratedClassifierCV(
        LogisticRegression, FEATURES, "y", cv=5, params={"max_iter": 200}
    ).fit(train)
    whole = LogisticRegression(FEATURES, "y", max_iter=200).fit(train)
    assert cross.model_.coef_ == pytest.approx(whole.coef_, abs=1e-9)


@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
def test_both_curves_produce_probabilities(split, method: str) -> None:
    train, test = split
    cross = CalibratedClassifierCV(
        LogisticRegression, FEATURES, "y", method=method, cv=5, params={"max_iter": 200}
    ).fit(train)
    values = cross.predict_proba(test).to_pydict()["calibrated"]
    assert all(0.0 <= v <= 1.0 for v in values), f"{method} left a value outside [0, 1]"


def test_the_staging_column_does_not_leak(split) -> None:
    train, test = split
    cross = CalibratedClassifierCV(
        LogisticRegression, FEATURES, "y", cv=5, params={"max_iter": 200}
    ).fit(train)
    columns = cross.predict_proba(test).columns
    assert not any(c.startswith("__bt_") for c in columns), columns


def test_predict_thresholds_the_calibrated_probability() -> None:
    ds = bt.from_pydict(
        {"x": [0.0, 1.0, 2.0, 3.0, 6.0, 7.0, 8.0, 9.0], "y": [0, 0, 0, 0, 1, 1, 1, 1]}
    )
    model = CalibratedClassifierCV(LogisticRegression, ["x"], "y", cv=2).fit(ds)
    assert model.predict(ds).to_pydict()["prediction"] == [0, 0, 0, 0, 1, 1, 1, 1]


def test_the_output_column_is_configurable() -> None:
    ds = bt.from_pydict(
        {"x": [0.0, 1.0, 2.0, 3.0, 6.0, 7.0, 8.0, 9.0], "y": [0, 0, 0, 0, 1, 1, 1, 1]}
    )
    model = CalibratedClassifierCV(LogisticRegression, ["x"], "y", cv=2, output_column="p").fit(ds)
    assert "p" in model.predict_proba(ds).columns


def test_the_folds_are_stratified_so_a_fold_keeps_both_classes() -> None:
    """A fold of one class gives the calibrator a constant label, which is worse than none."""
    rng = np.random.default_rng(3)
    n = 200
    x = rng.normal(size=n)
    y = [1 if i % 20 == 0 else 0 for i in range(n)]
    ds = bt.from_pydict({"x": x.tolist(), "y": y})
    model = CalibratedClassifierCV(LogisticRegression, ["x"], "y", cv=5).fit(ds)
    values = model.predict_proba(ds).to_pydict()["calibrated"]
    assert all(v == v for v in values), "a degenerate fold produced NaN"


# --------------------------------------------------------------------------------------
# Rejections
# --------------------------------------------------------------------------------------


def test_no_features_is_rejected() -> None:
    with pytest.raises(PlanError, match="at least one feature"):
        CalibratedClassifierCV(LogisticRegression, [], "y")


def test_an_unknown_method_is_rejected() -> None:
    with pytest.raises(PlanError, match="method must be one of"):
        CalibratedClassifierCV(LogisticRegression, ["x"], "y", method="spline")


def test_one_fold_is_rejected_because_nothing_is_held_out() -> None:
    with pytest.raises(PlanError, match="at least two folds"):
        CalibratedClassifierCV(LogisticRegression, ["x"], "y", cv=1)


def test_an_estimator_without_predict_proba_is_rejected() -> None:
    from batcher.ml import NearestCentroid

    with pytest.raises(PlanError, match="predict_proba"):
        CalibratedClassifierCV(NearestCentroid, ["x"], "y")


def test_a_missing_column_is_named() -> None:
    ds = bt.from_pydict({"x": [0.0, 1.0, 6.0, 7.0], "y": [0, 0, 1, 1]})
    with pytest.raises(ColumnNotFoundError):
        CalibratedClassifierCV(LogisticRegression, ["nope"], "y", cv=2).fit(ds)


def test_predicting_before_fitting_is_rejected() -> None:
    ds = bt.from_pydict({"x": [0.0, 1.0, 6.0, 7.0], "y": [0, 0, 1, 1]})
    with pytest.raises(PlanError, match="must be fitted"):
        CalibratedClassifierCV(LogisticRegression, ["x"], "y", cv=2).predict_proba(ds)
