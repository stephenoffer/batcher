"""Fairness metrics, count-model deviances, and imbalanced-learning resampling.

The fairness metrics are pinned to hand-computed group rates — the whole point is that they
are transparent, so the test computes the per-group rate the plain way and checks the gap.
The deviances are checked against scikit-learn exactly. The resampling is pinned to *exact*
class counts, because a resampler that is only approximately balanced is a resampler with a
latent bug, and to reproducibility, because a training set that changes between runs is not
one you can debug.
"""

from __future__ import annotations

import numpy as np
import pytest

import batcher as bt
from batcher._internal.errors import ColumnNotFoundError, PlanError
from batcher.ml.metrics import (
    d2_tweedie_score,
    demographic_parity_difference,
    disparate_impact_ratio,
    equal_opportunity_difference,
    equalized_odds_difference,
    group_fairness_report,
    predictive_parity_difference,
)
from batcher.ml.sampling import (
    balanced_sample,
    class_counts,
    class_weights,
    oversample,
    sample_weights,
    undersample,
)

pytestmark = pytest.mark.unit


# --- fairness metrics ------------------------------------------------------------------


def test_demographic_parity_is_the_selection_rate_gap() -> None:
    # Group a: 2/3 selected; group b: 1/3 selected; gap = 1/3.
    ds = bt.from_pydict({"g": ["a", "a", "a", "b", "b", "b"], "p": [1, 1, 0, 1, 0, 0]})
    assert demographic_parity_difference(ds, "g", "p") == pytest.approx(1.0 / 3.0)


def test_demographic_parity_is_zero_when_rates_match() -> None:
    ds = bt.from_pydict({"g": ["a", "a", "b", "b"], "p": [1, 0, 1, 0]})
    assert demographic_parity_difference(ds, "g", "p") == pytest.approx(0.0)


def test_disparate_impact_is_the_rate_ratio() -> None:
    # Group a: 2/2 selected; group b: 1/2 selected; ratio = 0.5.
    ds = bt.from_pydict({"g": ["a", "a", "b", "b"], "p": [1, 1, 1, 0]})
    assert disparate_impact_ratio(ds, "g", "p") == pytest.approx(0.5)


def test_equal_opportunity_is_the_true_positive_rate_gap() -> None:
    # Both groups all-positive labels; group a recalls 1.0, group b recalls 0.5.
    ds = bt.from_pydict({"g": ["a", "a", "b", "b"], "y": [1, 1, 1, 1], "p": [1, 1, 1, 0]})
    assert equal_opportunity_difference(ds, "g", "y", "p") == pytest.approx(0.5)


def test_equalized_odds_takes_the_worse_of_tpr_and_fpr_gaps() -> None:
    ds = bt.from_pydict({"g": ["a", "a", "b", "b"], "y": [1, 0, 1, 0], "p": [1, 0, 1, 1]})
    # Group a: tpr 1, fpr 0. Group b: tpr 1, fpr 1. FPR gap (1.0) dominates.
    assert equalized_odds_difference(ds, "g", "y", "p") == pytest.approx(1.0)


def test_predictive_parity_is_the_precision_gap() -> None:
    # Group a precision 1.0; group b precision 0.5; gap 0.5.
    ds = bt.from_pydict({"g": ["a", "a", "b", "b"], "y": [1, 1, 1, 0], "p": [1, 1, 1, 1]})
    assert predictive_parity_difference(ds, "g", "y", "p") == pytest.approx(0.5)


def test_group_fairness_report_matches_the_per_group_rates() -> None:
    rng = np.random.default_rng(0)
    g = rng.choice(["a", "b"], 400)
    y = rng.integers(0, 2, 400)
    p = rng.integers(0, 2, 400)
    ds = bt.from_pydict({"g": g.tolist(), "y": y.tolist(), "p": p.tolist()})
    report = group_fairness_report(ds, "g", "y", "p").sort("g").to_pydict()
    for i, group in enumerate(report["g"]):
        mask = g == group
        assert report["selection_rate"][i] == pytest.approx(p[mask].mean())
        assert report["support"][i] == int(mask.sum())


def test_a_perfectly_fair_model_has_zero_disparity() -> None:
    ds = bt.from_pydict({"g": ["a", "a", "b", "b"], "y": [1, 0, 1, 0], "p": [1, 0, 1, 0]})
    assert demographic_parity_difference(ds, "g", "p") == pytest.approx(0.0)
    assert equal_opportunity_difference(ds, "g", "y", "p") == pytest.approx(0.0)


def test_fairness_names_a_missing_column() -> None:
    ds = bt.from_pydict({"g": ["a"], "p": [1]})
    with pytest.raises(ColumnNotFoundError):
        demographic_parity_difference(ds, "g", "nope")


# --- count-model deviances -------------------------------------------------------------

skm = pytest.importorskip("sklearn.metrics", reason="scikit-learn is the deviance oracle")


@pytest.fixture(scope="module")
def counts() -> tuple[np.ndarray, np.ndarray, bt.Dataset]:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 12, 400).astype(float)
    p = np.abs(rng.normal(5.0, 2.0, 400)) + 0.5
    return y, p, bt.from_pydict({"y": y.tolist(), "p": p.tolist()})


@pytest.fixture(scope="module")
def positives() -> tuple[np.ndarray, np.ndarray, bt.Dataset]:
    rng = np.random.default_rng(1)
    y = np.abs(rng.normal(5.0, 2.0, 400)) + 0.5
    p = np.abs(rng.normal(5.0, 2.0, 400)) + 0.5
    return y, p, bt.from_pydict({"y": y.tolist(), "p": p.tolist()})


def test_poisson_deviance_matches_sklearn(counts) -> None:
    y, p, ds = counts
    got = ds.agg(m=bt.poisson_deviance("y", "p")).collect().column("m")[0].as_py()
    assert got == pytest.approx(skm.mean_poisson_deviance(y, p))


def test_gamma_deviance_matches_sklearn(positives) -> None:
    y, p, ds = positives
    got = ds.agg(m=bt.gamma_deviance("y", "p")).collect().column("m")[0].as_py()
    assert got == pytest.approx(skm.mean_gamma_deviance(y, p))


@pytest.mark.parametrize("power", [0.0, 1.0, 1.5, 2.0])
def test_tweedie_deviance_matches_sklearn(positives, power) -> None:
    y, p, ds = positives
    got = ds.agg(m=bt.tweedie_deviance("y", "p", power=power)).collect().column("m")[0].as_py()
    assert got == pytest.approx(skm.mean_tweedie_deviance(y, p, power=power))


def test_tweedie_rejects_the_undefined_power_range() -> None:
    ds = bt.from_pydict({"y": [1.0], "p": [1.0]})
    with pytest.raises(PlanError, match="undefined for power"):
        ds.agg(m=bt.tweedie_deviance("y", "p", power=0.5)).collect()


def test_d2_tweedie_matches_sklearn(positives) -> None:
    y, p, ds = positives
    got = d2_tweedie_score(ds, "y", "p", power=1.5)
    assert got == pytest.approx(skm.d2_tweedie_score(y, p, power=1.5))


def test_d2_is_one_for_a_perfect_fit() -> None:
    ds = bt.from_pydict({"y": [1.0, 2.0, 3.0], "p": [1.0, 2.0, 3.0]})
    assert d2_tweedie_score(ds, "y", "p", power=1.0) == pytest.approx(1.0)


# --- resampling ------------------------------------------------------------------------


def test_class_counts_are_exact() -> None:
    ds = bt.from_pydict({"y": [0] * 7 + [1] * 3})
    assert class_counts(ds, "y") == {0: 7, 1: 3}


def test_undersample_makes_the_classes_exactly_equal() -> None:
    ds = bt.from_pydict({"y": [0] * 100 + [1] * 17, "x": list(range(117))})
    got = class_counts(undersample(ds, "y", seed=1), "y")
    assert got[0] == got[1] == 17


def test_oversample_makes_the_classes_exactly_equal() -> None:
    ds = bt.from_pydict({"y": [0] * 40 + [1] * 7, "x": list(range(47))})
    got = class_counts(oversample(ds, "y", seed=1), "y")
    assert got[0] == got[1] == 40


def test_oversample_keeps_every_original_row() -> None:
    ds = bt.from_pydict({"y": [0] * 10 + [1] * 3, "x": list(range(13))})
    out = oversample(ds, "y", seed=1)
    # Every original row id survives (duplicates only add, never remove).
    assert set(range(13)) <= set(out.to_pydict()["x"])


def test_balanced_sample_moves_every_class_to_the_median() -> None:
    ds = bt.from_pydict({"y": [0] * 100 + [1] * 20 + [2] * 4, "x": list(range(124))})
    got = class_counts(balanced_sample(ds, "y", seed=1), "y")
    assert got[0] == got[1] == got[2] == 20


def test_resampling_is_reproducible() -> None:
    ds = bt.from_pydict({"y": [0] * 50 + [1] * 8, "x": list(range(58))})
    first = sorted(undersample(ds, "y", seed=3).to_pydict()["x"])
    second = sorted(undersample(ds, "y", seed=3).to_pydict()["x"])
    assert first == second


def test_undersample_to_an_explicit_target() -> None:
    ds = bt.from_pydict({"y": [0] * 50 + [1] * 30, "x": list(range(80))})
    got = class_counts(undersample(ds, "y", target=10, seed=1), "y")
    assert got[0] == got[1] == 10


def test_class_weights_balance_the_classes() -> None:
    ds = bt.from_pydict({"y": [0] * 3 + [1]})
    weights = class_weights(ds, "y")
    # n / (k * count): 4 / (2 * 3) and 4 / (2 * 1).
    assert weights[0] == pytest.approx(2.0 / 3.0)
    assert weights[1] == pytest.approx(2.0)


def test_sample_weights_appends_the_per_row_weight() -> None:
    ds = bt.from_pydict({"y": [0, 0, 0, 1]})
    out = sample_weights(ds, "y").to_pydict()
    assert out["sample_weight"] == [
        pytest.approx(2.0 / 3.0),
        pytest.approx(2.0 / 3.0),
        pytest.approx(2.0 / 3.0),
        pytest.approx(2.0),
    ]


def test_class_weights_average_to_one() -> None:
    rng = np.random.default_rng(4)
    y = rng.integers(0, 3, 300)
    ds = bt.from_pydict({"y": y.tolist()})
    weights = class_weights(ds, "y")
    counts = class_counts(ds, "y")
    weighted = sum(weights[c] * counts[c] for c in counts) / sum(counts.values())
    assert weighted == pytest.approx(1.0)
