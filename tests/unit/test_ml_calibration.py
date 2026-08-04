"""`PlattCalibrator` and `IsotonicCalibrator`.

The property that matters for a calibrator is not that it moves the numbers but that it
moves them the right way: the calibrated column must be *better calibrated* than the input,
measured by the metric the package already has. Several tests below assert exactly that
rather than pinning particular values, because a calibration that lowered the expected
calibration error only on a hand-picked frame would be worth nothing.
"""

from __future__ import annotations

import math

import pytest

import batcher as bt
from batcher._internal.errors import ColumnNotFoundError, PlanError
from batcher.ml.metrics import expected_calibration_error
from batcher.ml.preprocessors import (
    IsotonicCalibrator,
    PlattCalibrator,
    Preprocessor,
)
from batcher.ml.preprocessors.calibration.isotonic import pool_adjacent_violators

pytestmark = pytest.mark.unit


def _overconfident(n: int = 400) -> bt.Dataset:
    """Scores pushed towards the extremes, the way a boosted tree's are.

    The true rate is a clean logistic in the underlying signal, but the reported score is
    that probability cubed-and-renormalized, so it is systematically too confident. That is
    the distortion a calibrator is supposed to undo.
    """
    scores: list[float] = []
    labels: list[int] = []
    for i in range(n):
        true_rate = (i + 0.5) / n
        reported = true_rate**3 / (true_rate**3 + (1 - true_rate) ** 3)
        scores.append(reported)
        # Deterministic labels at the true rate: every other row flips once the running
        # share passes the rate, which reproduces the rate without a random seed.
        labels.append(1 if (i * 7919) % n < true_rate * n else 0)
    return bt.from_pydict({"score": scores, "label": labels})


def _ece(ds: bt.Dataset, column: str) -> float:
    return expected_calibration_error(ds, "label", column, bins=10)


@pytest.mark.parametrize("calibrator", [PlattCalibrator, IsotonicCalibrator])
def test_calibration_reduces_the_expected_calibration_error(calibrator) -> None:
    ds = _overconfident()
    out = calibrator("score", "label").fit_transform(ds)
    assert _ece(out, "calibrated") < _ece(out, "score")


@pytest.mark.parametrize("calibrator", [PlattCalibrator, IsotonicCalibrator])
def test_the_calibrated_column_is_a_probability(calibrator) -> None:
    out = calibrator("score", "label").fit_transform(_overconfident())
    values = out.to_pydict()["calibrated"]
    assert all(0.0 <= v <= 1.0 for v in values)


@pytest.mark.parametrize("calibrator", [PlattCalibrator, IsotonicCalibrator])
def test_calibration_is_monotone_in_the_score(calibrator) -> None:
    """A calibrator may not reorder the model's ranking, so AUC must be untouched."""
    fitted = calibrator("score", "label").fit(_overconfident())
    probe = bt.from_pydict({"score": [i / 20 for i in range(21)]})
    values = fitted.transform(probe).to_pydict()["calibrated"]
    assert values == sorted(values)


@pytest.mark.parametrize("calibrator", [PlattCalibrator, IsotonicCalibrator])
def test_the_fitted_calibration_applies_unchanged_to_another_split(calibrator) -> None:
    ds = _overconfident()
    train, test = ds.ml.train_test_split(0.5, seed=0, key="score")
    fitted = calibrator("score", "label").fit(train)
    out = fitted.transform(test)
    assert out.count() == test.count()
    assert "calibrated" in out.columns


@pytest.mark.parametrize("calibrator", [PlattCalibrator, IsotonicCalibrator])
def test_the_output_column_is_configurable(calibrator) -> None:
    out = calibrator("score", "label", output_column="p").fit_transform(_overconfident())
    assert "p" in out.columns and "calibrated" not in out.columns


@pytest.mark.parametrize("calibrator", [PlattCalibrator, IsotonicCalibrator])
def test_a_non_default_positive_class_is_honoured(calibrator) -> None:
    ds = bt.from_pydict(
        {"score": [0.1, 0.2, 0.8, 0.9] * 5, "label": ["no", "no", "yes", "yes"] * 5}
    )
    fitted = calibrator("score", "label", positive="yes").fit(ds)
    values = fitted.transform(ds).to_pydict()["calibrated"]
    assert values[0] < values[2]


@pytest.mark.parametrize("calibrator", [PlattCalibrator, IsotonicCalibrator])
def test_transform_before_fit_names_the_class(calibrator) -> None:
    with pytest.raises(PlanError, match="must be fitted"):
        calibrator("score", "label").transform(_overconfident())


@pytest.mark.parametrize("calibrator", [PlattCalibrator, IsotonicCalibrator])
def test_a_missing_column_is_named(calibrator) -> None:
    ds = bt.from_pydict({"score": [0.1, 0.9], "label": [0, 1]})
    with pytest.raises(ColumnNotFoundError):
        calibrator("nope", "label").fit(ds)
    with pytest.raises(ColumnNotFoundError):
        calibrator("score", "nope").fit(ds)


@pytest.mark.parametrize("calibrator", [PlattCalibrator, IsotonicCalibrator])
def test_a_fitted_calibrator_round_trips_through_save(calibrator, tmp_path) -> None:
    ds = _overconfident()
    fitted = calibrator("score", "label").fit(ds)
    target = str(tmp_path / "calibrator.json")
    fitted.save(target)
    restored = Preprocessor.load(target)
    assert (
        restored.transform(ds).to_pydict()["calibrated"]
        == fitted.transform(ds).to_pydict()["calibrated"]
    )


def test_platt_rejects_a_single_class_split() -> None:
    ds = bt.from_pydict({"score": [0.1, 0.5, 0.9], "label": [1, 1, 1]})
    with pytest.raises(PlanError, match="only one class"):
        PlattCalibrator("score", "label").fit(ds)


def test_platt_learns_a_positive_slope_when_the_score_ranks_correctly() -> None:
    fitted = PlattCalibrator("score", "label").fit(_overconfident())
    assert fitted.coef_ > 0


def test_isotonic_rejects_too_few_bins() -> None:
    with pytest.raises(PlanError, match="n_bins must be at least 2"):
        IsotonicCalibrator("score", "label", n_bins=1)


def test_isotonic_fitted_values_are_non_decreasing() -> None:
    fitted = IsotonicCalibrator("score", "label", n_bins=20).fit(_overconfident())
    assert fitted.values_ == sorted(fitted.values_)
    assert len(fitted.values_) == len(fitted.thresholds_) + 1


def test_isotonic_clamps_outside_the_fitted_range() -> None:
    """Below the first boundary and above the last, the step function is flat."""
    fitted = IsotonicCalibrator("score", "label", n_bins=10).fit(_overconfident())
    edges = fitted.transform(bt.from_pydict({"score": [-5.0, 5.0]})).to_pydict()["calibrated"]
    assert edges[0] == pytest.approx(fitted.values_[0])
    assert edges[1] == pytest.approx(fitted.values_[-1])


def test_isotonic_fit_is_independent_of_partitioning() -> None:
    one = IsotonicCalibrator("score", "label", n_bins=10).fit(_overconfident())
    many = IsotonicCalibrator("score", "label", n_bins=10).fit(_overconfident().repartition(4))
    assert one.values_ == many.values_


@pytest.mark.parametrize(
    ("values", "weights", "expected"),
    [
        ([1.0, 2.0, 3.0], [1.0, 1.0, 1.0], [1.0, 2.0, 3.0]),
        ([3.0, 2.0, 1.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]),
        ([0.1, 0.9, 0.5], [1.0, 1.0, 1.0], [0.1, 0.7, 0.7]),
        ([1.0], [5.0], [1.0]),
    ],
)
def test_pool_adjacent_violators_cases(values, weights, expected) -> None:
    got = pool_adjacent_violators(values, weights)
    assert all(math.isclose(a, b) for a, b in zip(got, expected, strict=True))


def test_pool_adjacent_violators_matches_sklearn() -> None:
    sklearn_isotonic = pytest.importorskip("sklearn.isotonic")
    values = [0.2, 0.1, 0.5, 0.4, 0.9, 0.3, 0.8]
    weights = [1.0, 2.0, 1.0, 3.0, 1.0, 1.0, 2.0]
    got = pool_adjacent_violators(values, weights)
    want = sklearn_isotonic.IsotonicRegression().fit_transform(
        list(range(len(values))), values, sample_weight=weights
    )
    assert all(
        math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12) for a, b in zip(got, want, strict=True)
    )


def test_pool_adjacent_violators_respects_weights() -> None:
    """A heavy bucket should pull the pooled mean towards itself."""
    light = pool_adjacent_violators([1.0, 0.0], [1.0, 1.0])
    heavy = pool_adjacent_violators([1.0, 0.0], [1.0, 9.0])
    assert light[0] == pytest.approx(0.5)
    assert heavy[0] == pytest.approx(0.1)


def test_calibrators_compose_in_a_chain() -> None:
    from batcher.ml.preprocessors import Chain, FunctionTransformer

    ds = _overconfident()
    out = Chain(
        FunctionTransformer("score", lambda c: c.clip(bt.lit(0.01), bt.lit(0.99))),
        PlattCalibrator("score", "label"),
    ).fit_transform(ds)
    assert "calibrated" in out.columns
