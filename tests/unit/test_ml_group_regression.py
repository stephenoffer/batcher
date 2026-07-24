"""Group-aggregate features, group imputation, and the regression diagnostics.

The group features are pinned to the property that makes them safe: a fitted encoder or
imputer must apply the *training* group statistics to a serving row, not recompute them from
the serving batch — otherwise a single-row serving call would encode that row against itself.
The regression diagnostics are checked against hand-computed or scikit-learn values.
"""

from __future__ import annotations

import numpy as np
import pytest

import batcher as bt
from batcher._internal.errors import ColumnNotFoundError, PlanError
from batcher.ml.metrics import (
    prediction_interval_coverage,
    residual_summary,
    top_k_accuracy,
)
from batcher.ml.preprocessors import GroupImputer, GroupStatEncoder

pytestmark = pytest.mark.unit


# --- GroupStatEncoder ------------------------------------------------------------------


def test_group_stat_encoder_attaches_the_group_mean() -> None:
    ds = bt.from_pydict({"cust": ["a", "a", "b"], "amount": [10.0, 20.0, 100.0]})
    out = GroupStatEncoder("amount", by="cust", statistics=["mean"]).fit_transform(ds)
    got = out.sort("amount").to_pydict()["amount_mean_by_cust"]
    assert got == [15.0, 15.0, 100.0]


def test_group_stat_encoder_can_attach_several_statistics() -> None:
    ds = bt.from_pydict({"g": ["a", "a", "a"], "v": [1.0, 2.0, 3.0]})
    out = GroupStatEncoder("v", by="g", statistics=["mean", "std", "count"]).fit_transform(ds)
    assert {"v_mean_by_g", "v_std_by_g", "v_count_by_g"} <= set(out.columns)


def test_group_stat_encoder_applies_training_stats_to_serving() -> None:
    # The safety property: a serving row is described by its group's *training* behaviour,
    # not by itself. A one-row serving call must not recompute the mean from that row.
    train = bt.from_pydict({"g": ["a", "a"], "v": [2.0, 4.0]})
    pre = GroupStatEncoder("v", by="g", statistics=["mean"]).fit(train)
    serve = bt.from_pydict({"g": ["a"], "v": [99.0]})
    assert pre.transform(serve).to_pydict()["v_mean_by_g"] == [3.0]


def test_group_stat_encoder_gives_an_unseen_group_null() -> None:
    train = bt.from_pydict({"g": ["a"], "v": [5.0]})
    pre = GroupStatEncoder("v", by="g", statistics=["mean"]).fit(train)
    got = pre.transform(bt.from_pydict({"g": ["never_seen"], "v": [1.0]})).to_pydict()
    assert got["v_mean_by_g"] == [None]


def test_group_stat_encoder_supports_a_composite_key() -> None:
    ds = bt.from_pydict({"a": ["x", "x"], "b": ["p", "q"], "v": [1.0, 3.0]})
    out = GroupStatEncoder("v", by=["a", "b"], statistics=["mean"]).fit_transform(ds)
    assert "v_mean_by_a_b" in out.columns


def test_group_stat_encoder_rejects_an_unknown_statistic() -> None:
    with pytest.raises(PlanError, match="unknown group statistic"):
        GroupStatEncoder("v", by="g", statistics=["variance"])


# --- GroupImputer ----------------------------------------------------------------------


def test_group_imputer_fills_with_the_group_mean() -> None:
    ds = bt.from_pydict({"seg": ["a", "a", "b"], "income": [10.0, None, 50.0]})
    out = GroupImputer("income", by="seg").fit_transform(ds)
    assert out.sort("seg").to_pydict()["income"] == [10.0, 10.0, 50.0]


def test_group_imputer_uses_the_training_group_mean_at_serve() -> None:
    train = bt.from_pydict({"g": ["a", "a"], "v": [2.0, 4.0]})
    pre = GroupImputer("v", by="g").fit(train)
    assert pre.transform(bt.from_pydict({"g": ["a"], "v": [None]})).to_pydict()["v"] == [3.0]


def test_group_imputer_falls_back_to_the_global_mean_for_an_unseen_group() -> None:
    train = bt.from_pydict({"g": ["a", "a", "b"], "v": [2.0, 4.0, 12.0]})
    pre = GroupImputer("v", by="g").fit(train)
    # Global mean is (2 + 4 + 12) / 3 = 6.0; the group "z" was never seen.
    got = pre.transform(bt.from_pydict({"g": ["z"], "v": [None]})).to_pydict()["v"]
    assert got == [6.0]


def test_group_imputer_leaves_present_values_untouched() -> None:
    ds = bt.from_pydict({"g": ["a", "a"], "v": [7.0, None]})
    out = GroupImputer("v", by="g").fit_transform(ds).sort("v").to_pydict()
    assert 7.0 in out["v"]


# --- residual_summary ------------------------------------------------------------------


def test_residual_summary_reports_the_mean_residual() -> None:
    ds = bt.from_pydict({"y": [1.0, 2.0, 3.0], "p": [1.0, 2.0, 4.0]})
    assert residual_summary(ds, "y", "p")["mean_residual"] == pytest.approx(1.0 / 3.0)


def test_residual_summary_surfaces_a_biased_segment() -> None:
    # Zero bias overall, but one segment is systematically high — the thing a global RMSE
    # hides and grouping the residuals reveals.
    ds = bt.from_pydict(
        {
            "seg": ["a", "a", "b", "b"],
            "y": [10.0, 10.0, 10.0, 10.0],
            "p": [12.0, 12.0, 8.0, 8.0],
        }
    )
    report = residual_summary(ds, "y", "p", by="seg").sort("seg").to_pydict()
    assert report["mean_residual"] == [2.0, -2.0]


def test_residual_summary_names_a_missing_column() -> None:
    ds = bt.from_pydict({"y": [1.0]})
    with pytest.raises(ColumnNotFoundError):
        residual_summary(ds, "y", "nope")


# --- prediction_interval_coverage ------------------------------------------------------


def test_interval_coverage_is_the_fraction_inside() -> None:
    ds = bt.from_pydict({"y": [1.0, 5.0, 9.0], "lo": [0.0, 4.0, 20.0], "hi": [2.0, 6.0, 30.0]})
    assert prediction_interval_coverage(ds, "y", "lo", "hi") == pytest.approx(2.0 / 3.0)


def test_a_calibrated_interval_covers_its_nominal_rate() -> None:
    # A 90% interval built as the empirical 5th and 95th percentiles must cover ≈0.9.
    rng = np.random.default_rng(3)
    actuals = rng.normal(size=5000)
    lo, hi = np.quantile(actuals, 0.05), np.quantile(actuals, 0.95)
    ds = bt.from_pydict({"y": actuals.tolist(), "lo": [lo] * 5000, "hi": [hi] * 5000})
    assert prediction_interval_coverage(ds, "y", "lo", "hi") == pytest.approx(0.9, abs=0.02)


def test_interval_coverage_can_be_grouped() -> None:
    ds = bt.from_pydict({"g": ["a", "b"], "y": [1.0, 9.0], "lo": [0.0, 0.0], "hi": [2.0, 2.0]})
    got = prediction_interval_coverage(ds, "y", "lo", "hi", by="g").sort("g").to_pydict()
    assert got["coverage"] == [1.0, 0.0]


# --- top_k_accuracy --------------------------------------------------------------------

skm = pytest.importorskip("sklearn.metrics", reason="scikit-learn is the metric oracle")


@pytest.mark.parametrize("k", [1, 2, 3])
def test_top_k_accuracy_matches_sklearn(k) -> None:
    rng = np.random.default_rng(0)
    n, classes = 500, 5
    probs = rng.random((n, classes))
    probs /= probs.sum(axis=1, keepdims=True)
    labels = rng.integers(0, classes, n)
    ds = bt.from_pydict(
        {"y": labels.tolist(), **{f"c{i}": probs[:, i].tolist() for i in range(classes)}}
    )
    got = top_k_accuracy(ds, "y", [f"c{i}" for i in range(classes)], k=k)
    expected = skm.top_k_accuracy_score(labels, probs, k=k, labels=list(range(classes)))
    assert got == pytest.approx(expected)


def test_top_1_accuracy_is_ordinary_accuracy() -> None:
    ds = bt.from_pydict({"y": [0, 1], "c0": [0.9, 0.4], "c1": [0.1, 0.6]})
    assert top_k_accuracy(ds, "y", ["c0", "c1"], k=1) == pytest.approx(1.0)


def test_top_k_accuracy_honors_explicit_labels() -> None:
    ds = bt.from_pydict({"y": ["cat", "dog"], "cat_score": [0.9, 0.3], "dog_score": [0.1, 0.7]})
    got = top_k_accuracy(ds, "y", ["cat_score", "dog_score"], k=1, labels=["cat", "dog"])
    assert got == pytest.approx(1.0)


def test_top_k_accuracy_rejects_a_k_wider_than_the_class_set() -> None:
    ds = bt.from_pydict({"y": [0], "c0": [0.5], "c1": [0.5]})
    with pytest.raises(PlanError, match="exceeds"):
        top_k_accuracy(ds, "y", ["c0", "c1"], k=3)
