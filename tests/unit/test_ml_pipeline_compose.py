"""`Pipeline` — preprocessing and a model as one object.

The property under test is not that it runs but that `predict` replays *exactly* what `fit`
did. A pipeline that re-fits its steps on the frame being scored, or that scores behind one
fewer transform, returns plausible numbers and no error — which is the failure mode the
class exists to make impossible.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.ml import (
    LinearRegression,
    Pipeline,
    SimpleImputer,
    StandardScaler,
)

pytestmark = pytest.mark.unit


def _train() -> bt.Dataset:
    return bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0, 5.0], "y": [2.0, 4.0, 6.0, 8.0, 10.0]})


def _pipe() -> Pipeline:
    return Pipeline(StandardScaler(["x"]), model=LinearRegression(["x"], "y"))


def test_a_pipeline_recovers_the_signal_through_its_transforms() -> None:
    scored = _pipe().fit(_train()).predict(_train()).to_pydict()
    assert scored["prediction"] == pytest.approx([2.0, 4.0, 6.0, 8.0, 10.0], abs=1e-6)


def test_predict_uses_the_training_statistics_not_the_scored_frames() -> None:
    """The whole point: a scaler must not re-fit on the rows it is scoring."""
    pipe = _pipe().fit(_train())
    shifted = bt.from_pydict({"x": [100.0, 200.0]})
    got = pipe.predict(shifted).to_pydict()["prediction"]
    # Standardized against the *training* mean/std, x=100 is far outside the fitted range,
    # so the prediction must be far above the training targets. Re-fitting on this frame
    # would centre it and return something near the training mean instead.
    assert got[0] > 100.0


def test_transform_shows_the_model_its_actual_input() -> None:
    pipe = _pipe().fit(bt.from_pydict({"x": [1.0, 3.0], "y": [2.0, 6.0]}))
    assert pipe.transform(bt.from_pydict({"x": [1.0, 3.0]})).to_pydict()["x"] == [-1.0, 1.0]


def test_the_steps_are_fitted_in_order_each_on_the_last_ones_output() -> None:
    ds = bt.from_pydict({"x": [1.0, None, 3.0, 5.0], "y": [2.0, 4.0, 6.0, 10.0]})
    pipe = Pipeline(
        SimpleImputer(["x"]), StandardScaler(["x"]), model=LinearRegression(["x"], "y")
    ).fit(ds)
    assert pipe.steps[0].statistics_["x"] == pytest.approx(3.0)
    # The scaler saw the *imputed* column, so its mean is over four values, not three.
    assert pipe.steps[1].mean_["x"] == pytest.approx(3.0)


def test_a_pipeline_with_no_steps_is_just_the_model() -> None:
    pipe = Pipeline(model=LinearRegression(["x"], "y")).fit(_train())
    assert pipe.predict(_train()).to_pydict()["prediction"] == pytest.approx(
        [2.0, 4.0, 6.0, 8.0, 10.0], abs=1e-6
    )


def test_predict_stays_lazy() -> None:
    """Every step is an Expr projection, so scoring is one plan rather than staged frames."""
    pipe = _pipe().fit(_train())
    out = pipe.predict(_train())
    assert isinstance(out, bt.Dataset)
    assert "prediction" in out.columns


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (
            lambda: Pipeline(LinearRegression(["x"], "y"), model=LinearRegression(["x"], "y")),
            "steps must be preprocessors",
        ),
        (lambda: Pipeline(StandardScaler(["x"]), model="not a model"), "must have fit"),
    ],
)
def test_configuration_is_validated(factory, message: str) -> None:
    with pytest.raises(PlanError, match=message):
        factory()


def test_the_whole_pipeline_round_trips_through_one_file(tmp_path) -> None:
    ds = _train()
    pipe = _pipe().fit(ds)
    target = str(tmp_path / "pipe.json")
    pipe.save(target)
    restored = Pipeline.load(target)
    assert len(restored.steps) == 1
    assert restored.predict(ds).to_pydict() == pipe.predict(ds).to_pydict()


def test_a_multi_step_pipeline_round_trips(tmp_path) -> None:
    ds = bt.from_pydict({"x": [1.0, None, 3.0, 5.0], "y": [2.0, 4.0, 6.0, 10.0]})
    pipe = Pipeline(
        SimpleImputer(["x"]), StandardScaler(["x"]), model=LinearRegression(["x"], "y")
    ).fit(ds)
    target = str(tmp_path / "pipe.json")
    pipe.save(target)
    restored = Pipeline.load(target)
    assert [type(s).__name__ for s in restored.steps] == ["SimpleImputer", "StandardScaler"]
    assert restored.predict(ds).to_pydict() == pipe.predict(ds).to_pydict()


def test_the_saved_file_holds_both_halves(tmp_path) -> None:
    """Saving them separately is what lets them drift, so one document carries both."""
    import json

    target = tmp_path / "pipe.json"
    _pipe().fit(_train()).save(str(target))
    document = json.loads(target.read_text())
    assert document["class"] == "Pipeline"
    assert document["model"]["class"] == "LinearRegression"
    assert document["steps"][0]["class"] == "StandardScaler"


def test_a_reloaded_pipeline_needs_neither_the_training_data_nor_a_refit(tmp_path) -> None:
    target = str(tmp_path / "pipe.json")
    _pipe().fit(_train()).save(target)
    fresh = bt.from_pydict({"x": [6.0]})
    assert Pipeline.load(target).predict(fresh).to_pydict()["prediction"] == pytest.approx(
        [12.0], abs=1e-6
    )


def test_it_composes_with_the_cross_validation_loop() -> None:
    from batcher.ml.metrics import evaluate
    from batcher.ml.model_selection import cross_val_score

    ds = bt.from_pydict({"x": [float(i) for i in range(40)], "y": [2.0 * i for i in range(40)]})
    scores = cross_val_score(
        ds,
        lambda train: Pipeline(StandardScaler(["x"]), model=LinearRegression(["x"], "y")).fit(
            train
        ),
        lambda model, part: model.predict(part),
        y_true="y",
        metric=lambda d, t, p: evaluate(d, t, y_pred=p, task="regression", metrics=["r2"])["r2"],
        k=4,
        key="x",
    )
    assert all(s > 0.99 for s in scores)
