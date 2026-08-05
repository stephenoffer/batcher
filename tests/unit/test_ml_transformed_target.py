"""`TransformedTargetRegressor`.

The failure this prevents is silent: fitting on `log1p(y)` and forgetting to invert returns
predictions wrong by a factor of *e*, with the right shape and no error. So the tests are
about the round trip — the prediction must come back on the *original* scale — and about the
wrapper genuinely helping on the skewed target it exists for.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import batcher as bt
from batcher._internal.errors import ColumnNotFoundError, PlanError
from batcher.ml import LinearRegression, TransformedTargetRegressor
from batcher.ml.metrics import evaluate

pytestmark = pytest.mark.unit


def _exponential() -> bt.Dataset:
    """``y = exp(x) - 1``, so ``log1p(y)`` is exactly linear in x."""
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    return bt.from_pydict({"x": xs, "y": [math.expm1(v) for v in xs]})


def test_the_prediction_comes_back_on_the_original_scale() -> None:
    ds = _exponential()
    model = TransformedTargetRegressor(LinearRegression(["x"], "y"), target="y").fit(ds)
    got = model.predict(ds).to_pydict()["prediction"]
    assert got == pytest.approx(ds.to_pydict()["y"], rel=1e-6)


def test_forgetting_the_inverse_would_be_visible() -> None:
    """The wrapped model's own output is on the log scale; the wrapper's is not."""
    ds = _exponential()
    wrapped = TransformedTargetRegressor(LinearRegression(["x"], "y"), target="y").fit(ds)
    raw = wrapped.model.predict(ds).to_pydict()["prediction"]
    inverted = wrapped.predict(ds).to_pydict()["prediction"]
    assert raw[-1] < 6.0  # log1p(148) is about 5
    assert inverted[-1] > 100.0


def test_it_beats_an_untransformed_fit_on_a_skewed_target() -> None:
    """The reason the technique exists, measured rather than asserted."""
    ds = _exponential()
    plain = LinearRegression(["x"], "y").fit(ds)
    wrapped = TransformedTargetRegressor(LinearRegression(["x"], "y"), target="y").fit(ds)

    def error(model) -> float:
        scored = model.predict(ds)
        return evaluate(scored, "y", y_pred="prediction", task="regression", metrics=["rmse"])[
            "rmse"
        ]

    assert error(wrapped) < error(plain)


@pytest.mark.parametrize(
    ("transform", "targets"),
    [
        # Each case is exactly linear in the *transformed* space, so a perfect fit there
        # must invert back to the original values with no error left over.
        ("log1p", [math.expm1(1.0), math.expm1(2.0), math.expm1(3.0), math.expm1(4.0)]),
        ("sqrt", [4.0, 9.0, 16.0, 25.0]),
    ],
)
def test_each_transform_inverts_exactly(transform: str, targets: list[float]) -> None:
    ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0], "y": targets})
    model = TransformedTargetRegressor(
        LinearRegression(["x"], "y"), target="y", transform=transform
    ).fit(ds)
    assert model.predict(ds).to_pydict()["prediction"] == pytest.approx(targets, rel=1e-6)


def test_log_transform_is_available_for_a_strictly_positive_target() -> None:
    xs = [1.0, 2.0, 3.0, 4.0]
    ds = bt.from_pydict({"x": xs, "y": [math.exp(v) for v in xs]})
    model = TransformedTargetRegressor(
        LinearRegression(["x"], "y"), target="y", transform="log"
    ).fit(ds)
    assert model.predict(ds).to_pydict()["prediction"] == pytest.approx(
        ds.to_pydict()["y"], rel=1e-6
    )


def test_scoring_does_not_require_the_target_column() -> None:
    """Only `fit` transforms the target, so a serving frame need not carry it."""
    model = TransformedTargetRegressor(LinearRegression(["x"], "y"), target="y").fit(_exponential())
    scored = model.predict(bt.from_pydict({"x": [2.0]}))
    assert "y" not in scored.columns
    assert scored.to_pydict()["prediction"][0] == pytest.approx(math.expm1(2.0), rel=1e-6)


def test_the_training_frame_is_not_mutated() -> None:
    """The forward transform is a lazy projection, not an edit of the caller's dataset."""
    ds = _exponential()
    before = ds.to_pydict()["y"]
    TransformedTargetRegressor(LinearRegression(["x"], "y"), target="y").fit(ds)
    assert ds.to_pydict()["y"] == before


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (
            lambda: TransformedTargetRegressor("not a model", target="y"),
            "needs a model with fit",
        ),
        (
            lambda: TransformedTargetRegressor(
                LinearRegression(["x"], "y"), target="y", transform="cube"
            ),
            "transform must be one of",
        ),
    ],
)
def test_configuration_is_validated(factory, message: str) -> None:
    with pytest.raises(PlanError, match=message):
        factory()


def test_a_missing_target_is_named() -> None:
    model = TransformedTargetRegressor(LinearRegression(["x"], "nope"), target="nope")
    with pytest.raises(ColumnNotFoundError):
        model.fit(_exponential())


def test_it_composes_with_cross_validation() -> None:
    from batcher.ml.model_selection import cross_val_score

    rng = np.random.default_rng(0)
    xs = rng.uniform(0.5, 4.0, size=60)
    ds = bt.from_pydict({"x": xs.tolist(), "y": np.expm1(xs).tolist()})
    scores = cross_val_score(
        ds,
        lambda train: TransformedTargetRegressor(LinearRegression(["x"], "y"), target="y").fit(
            train
        ),
        lambda model, part: model.predict(part),
        y_true="y",
        metric=lambda d, t, p: evaluate(d, t, y_pred=p, task="regression", metrics=["r2"])["r2"],
        k=4,
        key="x",
    )
    assert all(s > 0.99 for s in scores)
