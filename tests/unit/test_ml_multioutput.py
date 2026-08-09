"""One model per target, kept as one object.

Every estimator here takes exactly one target column. Predicting several things from the same
features - demand per region, a reading per axis, a set of independent tags - therefore leaves
the caller holding a list of models and the job of remembering which column each was fitted
on. That is the train/serve skew `Pipeline` exists to prevent, one level up.

These also pin the distinction that gets confused: multi-*class* picks one of several classes
and is `OneVsRestClassifier`'s job; multi-*label* lets a row carry several tags at once and is
this one's. A test drives a row that is genuinely two labels at the same time.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher._internal.errors import ColumnNotFoundError, PlanError
from batcher.ml import (
    LinearRegression,
    LogisticRegression,
    MultiOutputClassifier,
    MultiOutputRegressor,
)

pytestmark = pytest.mark.unit


def _targets() -> bt.Dataset:
    """Two targets that are exact linear functions of one feature, with different slopes."""
    return bt.from_pydict(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "north": [2.0, 4.0, 6.0, 8.0, 10.0],
            "south": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )


def _tags() -> bt.Dataset:
    """Documents carrying two independent tags, so a row can be both or neither."""
    return bt.from_pydict(
        {
            "length": [10.0, 12.0, 90.0, 95.0, 11.0, 92.0],
            "words": [3.0, 4.0, 40.0, 42.0, 3.0, 41.0],
            "urgent": [1, 1, 0, 0, 1, 0],
            "long": [0, 0, 1, 1, 0, 1],
        }
    )


def test_each_target_gets_its_own_model() -> None:
    model = MultiOutputRegressor(LinearRegression, ["x"], ["north", "south"]).fit(_targets())
    assert len(model.estimators_) == 2
    assert [m.target for m in model.estimators_] == ["north", "south"]


def test_the_models_learn_the_different_slopes() -> None:
    """A single shared model would have to compromise between 2x and 10x."""
    model = MultiOutputRegressor(LinearRegression, ["x"], ["north", "south"]).fit(_targets())
    assert model.estimators_[0].coef_[0] == pytest.approx(2.0, abs=1e-6)
    assert model.estimators_[1].coef_[0] == pytest.approx(10.0, abs=1e-6)


def test_predict_appends_one_column_per_target() -> None:
    ds = _targets()
    model = MultiOutputRegressor(LinearRegression, ["x"], ["north", "south"]).fit(ds)
    scored = model.predict(ds)
    assert "prediction_north" in scored.columns
    assert "prediction_south" in scored.columns
    table = scored.to_pydict()
    assert table["prediction_north"] == pytest.approx(table["north"], abs=1e-6)
    assert table["prediction_south"] == pytest.approx(table["south"], abs=1e-6)


def test_each_sub_model_agrees_with_fitting_it_alone() -> None:
    """The wrapper must change the bookkeeping and nothing about the fit."""
    ds = _targets()
    wrapped = MultiOutputRegressor(LinearRegression, ["x"], ["north", "south"]).fit(ds)
    alone = LinearRegression(["x"], "south").fit(ds)
    assert wrapped.estimators_[1].coef_ == pytest.approx(alone.coef_, abs=1e-9)
    assert wrapped.estimators_[1].intercept_ == pytest.approx(alone.intercept_, abs=1e-9)


def test_scoring_every_target_is_one_pass() -> None:
    """Each sub-model contributes an expression to the same frame, not another scan."""
    ds = _targets()
    model = MultiOutputRegressor(LinearRegression, ["x"], ["north", "south"]).fit(ds)
    frame = type(ds)
    calls = {"n": 0}
    originals = {name: getattr(frame, name) for name in ("collect", "to_pydict", "count")}

    def wrap(original):
        def counted(self, *args, **kwargs):
            calls["n"] += 1
            return original(self, *args, **kwargs)

        return counted

    for name, original in originals.items():
        setattr(frame, name, wrap(original))
    try:
        model.predict(ds)
    finally:
        for name, original in originals.items():
            setattr(frame, name, original)
    assert calls["n"] == 0, "predict must build expressions, not execute"


def test_the_output_prefix_is_configurable() -> None:
    ds = _targets()
    model = MultiOutputRegressor(LinearRegression, ["x"], ["north"], output_prefix="yhat").fit(ds)
    assert "yhat_north" in model.predict(ds).columns


def test_params_reach_every_sub_model() -> None:
    from batcher.ml import Ridge

    model = MultiOutputRegressor(Ridge, ["x"], ["north", "south"], params={"alpha": 2.0}).fit(
        _targets()
    )
    assert [m._alpha for m in model.estimators_] == [2.0, 2.0]


# --------------------------------------------------------------------------------------
# Multi-label classification
# --------------------------------------------------------------------------------------


def test_a_row_can_carry_two_labels_at_once() -> None:
    """The property that distinguishes multi-label from multi-class.

    `OneVsRestClassifier` would pick one of the two; here both are answered independently,
    so a row is allowed to be both or neither.
    """
    ds = bt.from_pydict(
        {
            "length": [10.0, 95.0, 50.0, 12.0],
            "words": [3.0, 42.0, 20.0, 4.0],
            "urgent": [1, 0, 1, 1],
            "long": [0, 1, 1, 0],
        }
    )
    model = MultiOutputClassifier(LogisticRegression, ["length", "words"], ["urgent", "long"]).fit(
        ds
    )
    scored = model.predict(ds).to_pydict()
    both = [
        (urgent > 0.5, long > 0.5)
        for urgent, long in zip(scored["prediction_urgent"], scored["prediction_long"], strict=True)
    ]
    assert (True, True) in both, "a row must be able to hold both labels"


def test_the_classifier_separates_independent_tags() -> None:
    ds = _tags()
    model = MultiOutputClassifier(LogisticRegression, ["length"], ["urgent", "long"]).fit(ds)
    scored = model.predict(ds).to_pydict()
    assert scored["prediction_urgent"][0] > 0.5
    assert scored["prediction_long"][0] < 0.5
    assert scored["prediction_urgent"][3] < 0.5
    assert scored["prediction_long"][3] > 0.5


def test_the_classifier_is_the_regressor_with_a_different_name() -> None:
    """Shared implementation, so a fix to one cannot miss the other."""
    assert issubclass(MultiOutputClassifier, MultiOutputRegressor)


# --------------------------------------------------------------------------------------
# Round trip
# --------------------------------------------------------------------------------------


def test_it_survives_a_save_and_load() -> None:
    import tempfile

    from batcher.ml import load_model, save_model

    ds = _targets()
    model = MultiOutputRegressor(LinearRegression, ["x"], ["north", "south"]).fit(ds)
    with tempfile.TemporaryDirectory() as directory:
        path = f"{directory}/multi.json"
        save_model(model, path)
        restored = load_model(path)
    assert restored.targets == ["north", "south"]
    assert restored.estimator is LinearRegression
    assert len(restored.estimators_) == 2
    assert restored.predict(ds).to_pydict() == model.predict(ds).to_pydict()


# --------------------------------------------------------------------------------------
# Rejections
# --------------------------------------------------------------------------------------


def test_no_features_is_rejected() -> None:
    with pytest.raises(PlanError, match="at least one feature"):
        MultiOutputRegressor(LinearRegression, [], ["a"])


def test_no_targets_is_rejected() -> None:
    with pytest.raises(PlanError, match="at least one target"):
        MultiOutputRegressor(LinearRegression, ["x"], [])


def test_a_repeated_target_is_rejected() -> None:
    """Two models writing the same output column: the second would silently win."""
    with pytest.raises(PlanError, match="repeats"):
        MultiOutputRegressor(LinearRegression, ["x"], ["a", "a"])


def test_a_column_that_is_both_feature_and_target_is_rejected() -> None:
    """It would predict itself and score perfectly, which reads as a great model."""
    with pytest.raises(PlanError, match="both a feature and a target"):
        MultiOutputRegressor(LinearRegression, ["x", "y"], ["y"])


def test_too_many_targets_is_rejected() -> None:
    with pytest.raises(PlanError, match="ceiling"):
        MultiOutputRegressor(LinearRegression, ["x"], [f"t{i}" for i in range(6)], max_targets=5)


def test_a_missing_column_is_named() -> None:
    with pytest.raises(ColumnNotFoundError):
        MultiOutputRegressor(LinearRegression, ["nope"], ["north"]).fit(_targets())


def test_predicting_before_fitting_is_rejected() -> None:
    model = MultiOutputRegressor(LinearRegression, ["x"], ["north"])
    with pytest.raises(PlanError, match="must be fitted"):
        model.predict(_targets())
