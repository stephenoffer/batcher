"""Calibration metrics and weight-of-evidence encoding.

The calibration metrics are checked against a reference implementation written the plain way
over NumPy — bin, compare mean prediction to observed rate, average — because they measure
the property AUC cannot, and "trust the aggregate" is not a check. The WOE encoder is pinned
to the log-odds definition a credit scorecard depends on: a category that leans positive gets
a positive weight, an unseen category gets neutral 0, and a single-class target is refused.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.ml.metrics import (
    brier_skill_score,
    expected_calibration_error,
    maximum_calibration_error,
)
from batcher.ml.preprocessors import WOEEncoder

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def scored() -> tuple[np.ndarray, np.ndarray, bt.Dataset]:
    """A scored binary problem with a genuinely informative, roughly calibrated score."""
    rng = np.random.default_rng(0)
    score = rng.random(3000)
    labels = (rng.random(3000) < score).astype(int)
    return labels, score, bt.from_pydict({"y": labels.tolist(), "s": score.tolist()})


def _reference_ece(labels: np.ndarray, score: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for i in range(bins):
        low, high = edges[i], edges[i + 1]
        mask = (score >= low) & (score < high) if i < bins - 1 else (score >= low) & (score <= high)
        if mask.sum() == 0:
            continue
        total += mask.sum() * abs(score[mask].mean() - labels[mask].mean())
    return total / len(labels)


# --- calibration metrics ---------------------------------------------------------------


def test_expected_calibration_error_matches_the_reference(scored) -> None:
    labels, score, ds = scored
    assert expected_calibration_error(ds, "y", "s") == pytest.approx(_reference_ece(labels, score))


def test_a_perfectly_calibrated_model_has_zero_ece() -> None:
    ds = bt.from_pydict({"y": [0, 0, 1, 1], "s": [0.0, 0.0, 1.0, 1.0]})
    assert expected_calibration_error(ds, "y", "s", bins=2) == pytest.approx(0.0)


def test_maximum_calibration_error_is_the_worst_bin() -> None:
    # Bin 0 (scores < 0.5) holds a 0 and a 1 predicted ~0.15: gap ≈ 0.35.
    # Bin 1 (scores >= 0.5) holds two 1s predicted ~0.85: gap ≈ 0.15.
    ds = bt.from_pydict({"y": [0, 1, 1, 1], "s": [0.1, 0.2, 0.8, 0.9]})
    assert maximum_calibration_error(ds, "y", "s", bins=2) >= expected_calibration_error(
        ds, "y", "s", bins=2
    )


def test_maximum_calibration_error_ignores_empty_bins() -> None:
    # All scores land in one bin; the empty bins must not count as perfect zeros and pull the
    # maximum down.
    ds = bt.from_pydict({"y": [0, 1], "s": [0.95, 0.96]})
    value = maximum_calibration_error(ds, "y", "s", bins=10)
    assert value == pytest.approx(abs(0.955 - 0.5), abs=0.02)


def test_brier_skill_score_matches_the_reference(scored) -> None:
    labels, score, ds = scored
    from sklearn.metrics import brier_score_loss

    base = labels.mean()
    expected = 1.0 - brier_score_loss(labels, score) / (base * (1 - base))
    assert brier_skill_score(ds, "y", "s") == pytest.approx(expected)


def test_brier_skill_score_is_one_for_perfect_probabilities() -> None:
    ds = bt.from_pydict({"y": [0, 0, 1, 1], "s": [0.0, 0.0, 1.0, 1.0]})
    assert brier_skill_score(ds, "y", "s") == pytest.approx(1.0)


def test_brier_skill_score_is_negative_for_worse_than_base_rate() -> None:
    # Confidently wrong on every row: worse than predicting the base rate.
    ds = bt.from_pydict({"y": [0, 0, 1, 1], "s": [1.0, 1.0, 0.0, 0.0]})
    assert brier_skill_score(ds, "y", "s") < 0.0


def test_calibration_rejects_a_degenerate_bin_count(scored) -> None:
    _, _, ds = scored
    with pytest.raises(PlanError, match="at least 2 bins"):
        expected_calibration_error(ds, "y", "s", bins=1)


# --- WOE encoding ----------------------------------------------------------------------


def test_woe_leans_positive_for_a_positive_leaning_category() -> None:
    ds = bt.from_pydict({"grade": ["a", "a", "b", "b"], "default": [0, 0, 1, 1]})
    out = WOEEncoder(["grade"], "default").fit_transform(ds).to_pydict()["grade"]
    # Grade "a" is all-negative (WOE < 0); grade "b" is all-positive (WOE > 0).
    assert out[0] < 0 < out[2]


def test_woe_matches_the_log_odds_definition() -> None:
    # 2 positives of 4 total positives in cat "b"; 0 negatives of 4 total negatives.
    ds = bt.from_pydict(
        {
            "g": ["a", "a", "a", "a", "b", "b", "b", "b"],
            "y": [0, 0, 0, 0, 1, 1, 0, 0],
        }
    )
    pre = WOEEncoder(["g"], "y").fit(ds)
    total_pos, total_neg = 2.0, 6.0
    # Category "a": 0 pos, 4 neg.
    pos_share = (0 + 0.5) / (total_pos + 0.5)
    neg_share = (4 + 0.5) / (total_neg + 0.5)
    assert pre.woe_["g"]["a"] == pytest.approx(math.log(pos_share / neg_share))


def test_woe_encodes_an_unseen_category_as_neutral() -> None:
    train = bt.from_pydict({"g": ["a", "b", "b"], "y": [0, 1, 1]})
    pre = WOEEncoder(["g"], "y").fit(train)
    assert pre.transform(bt.from_pydict({"g": ["never_seen"]})).to_pydict()["g"] == [0.0]


def test_woe_applies_training_weights_to_serving() -> None:
    train = bt.from_pydict({"g": ["a", "a", "b", "b"], "y": [0, 0, 1, 1]})
    pre = WOEEncoder(["g"], "y").fit(train)
    learned = pre.woe_["g"]["b"]
    assert pre.transform(bt.from_pydict({"g": ["b"]})).to_pydict()["g"] == [pytest.approx(learned)]


def test_woe_is_finite_for_a_single_class_category() -> None:
    # Category "b" has only positives; the smoothing keeps its WOE large but finite.
    ds = bt.from_pydict({"g": ["a", "a", "b", "b"], "y": [0, 1, 1, 1]})
    pre = WOEEncoder(["g"], "y").fit(ds)
    assert math.isfinite(pre.woe_["g"]["b"])


def test_woe_refuses_a_single_class_target() -> None:
    ds = bt.from_pydict({"g": ["a", "b"], "y": [1, 1]})
    with pytest.raises(PlanError, match="only one class"):
        WOEEncoder(["g"], "y").fit(ds)


def test_woe_produces_an_additive_scorecard_feature() -> None:
    # The property that justifies WOE over target encoding: a strongly predictive category
    # gets a strongly signed weight, so a linear model reads it as a straight coefficient.
    rng = np.random.default_rng(1)
    grade = rng.choice(["a", "b", "c"], size=600)
    prob = {"a": 0.1, "b": 0.5, "c": 0.9}
    y = np.array([rng.random() < prob[g] for g in grade]).astype(int)
    ds = bt.from_pydict({"grade": grade.tolist(), "y": y.tolist()})
    weights = WOEEncoder(["grade"], "y").fit(ds).woe_["grade"]
    assert weights["a"] < weights["b"] < weights["c"]
