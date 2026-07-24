"""Outlier detection and agreement/efficiency metrics.

The outlier rules are pinned to their textbook definitions — Tukey's fence, the z-score rule,
the MAD rule — and to the behaviour that matters: a fitted clipper applies the *training*
bounds to serving data. The agreement metrics are checked against reference implementations,
because their whole reason to exist is measuring something a correlation does not, so "trust
the correlation" is precisely the wrong check.
"""

from __future__ import annotations

import numpy as np
import pytest

import batcher as bt
from batcher._internal.errors import ColumnNotFoundError, PlanError
from batcher.ml.outliers import (
    OutlierClipper,
    count_outliers,
    flag_outliers,
    outlier_bounds,
)

pytestmark = pytest.mark.unit


# --- outlier bounds --------------------------------------------------------------------


def test_iqr_bounds_are_tukeys_fence() -> None:
    ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    # q1 = 2, q3 = 4, IQR = 2; fence = [2 - 3, 4 + 3] = [-1, 7].
    assert outlier_bounds(ds, "x", method="iqr", threshold=1.5) == (-1.0, 7.0)


def test_zscore_bounds_are_mean_plus_minus_k_std() -> None:
    ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    lo, hi = outlier_bounds(ds, "x", method="zscore", threshold=2.0)
    mean = 3.0
    std = float(np.std([1, 2, 3, 4, 5], ddof=1))
    assert lo == pytest.approx(mean - 2 * std)
    assert hi == pytest.approx(mean + 2 * std)


def test_mad_bounds_are_robust_to_the_tail() -> None:
    # The MAD bounds are set by the median and MAD, so a single huge value barely moves them.
    clean = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    tainted = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 1e9]})
    clean_hi = outlier_bounds(clean, "x", method="mad", threshold=3.0)[1]
    tainted_hi = outlier_bounds(tainted, "x", method="mad", threshold=3.0)[1]
    assert abs(tainted_hi - clean_hi) < 5.0


def test_outlier_bounds_rejects_an_unknown_method() -> None:
    ds = bt.from_pydict({"x": [1.0]})
    with pytest.raises(PlanError, match="method must be"):
        outlier_bounds(ds, "x", method="isolation")


# --- flag / count ----------------------------------------------------------------------


def test_flag_outliers_marks_the_extreme_row() -> None:
    ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 100.0]})
    assert flag_outliers(ds, "x").to_pydict()["x_outlier"] == [False, False, False, True]


def test_flag_outliers_leaves_a_clean_column_unflagged() -> None:
    ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    assert not any(flag_outliers(ds, "x").to_pydict()["x_outlier"])


def test_count_outliers_tallies_per_column() -> None:
    ds = bt.from_pydict({"a": [1.0, 2.0, 3.0, 100.0], "b": [1.0, 1.0, 1.0, 1.0]})
    assert count_outliers(ds, ["a", "b"]) == {"a": 1, "b": 0}


def test_count_outliers_names_a_missing_column() -> None:
    ds = bt.from_pydict({"x": [1.0]})
    with pytest.raises(ColumnNotFoundError):
        count_outliers(ds, "nope")


def test_the_three_rules_disagree_on_a_skewed_column() -> None:
    # A right-skewed column: the z-score rule (dragged by the tail) flags fewer than the
    # robust IQR/MAD rules. The exact counts vary, but the robust rules should flag at least
    # as many as the mean-based one.
    rng = np.random.default_rng(0)
    values = np.concatenate([rng.exponential(1.0, 990), rng.uniform(20, 40, 10)])
    ds = bt.from_pydict({"x": values.tolist()})
    z = count_outliers(ds, "x", method="zscore", threshold=3.0)["x"]
    iqr = count_outliers(ds, "x", method="iqr", threshold=1.5)["x"]
    assert iqr >= z


# --- OutlierClipper --------------------------------------------------------------------


def test_outlier_clipper_clamps_the_tail() -> None:
    ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 100.0]})
    clipped = OutlierClipper("x").fit_transform(ds).to_pydict()["x"]
    assert max(clipped) < 100.0


def test_outlier_clipper_keeps_every_row() -> None:
    ds = bt.from_pydict({"x": [1.0, 2.0, 100.0]})
    assert OutlierClipper("x").fit_transform(ds).count() == 3


def test_outlier_clipper_applies_training_bounds_to_serving() -> None:
    train = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    pre = OutlierClipper("x").fit(train)
    # Training IQR fence is [-1, 7]; serving values clamp to it.
    assert pre.transform(bt.from_pydict({"x": [-50.0, 50.0]})).to_pydict()["x"] == [-1.0, 7.0]


def test_outlier_clipper_rejects_an_unknown_method() -> None:
    with pytest.raises(PlanError, match="method must be"):
        OutlierClipper("x", method="dbscan")


# --- agreement metrics -----------------------------------------------------------------


@pytest.fixture(scope="module")
def paired() -> tuple[np.ndarray, np.ndarray, bt.Dataset]:
    rng = np.random.default_rng(0)
    y = rng.normal(size=300)
    p = 0.8 * y + 0.5 + rng.normal(0, 0.3, 300)
    return y, p, bt.from_pydict({"y": y.tolist(), "p": p.tolist()})


def _agg(ds: bt.Dataset, expr) -> float:
    return ds.agg(m=expr).collect().column("m")[0].as_py()


def test_concordance_correlation_matches_the_reference(paired) -> None:
    y, p, ds = paired
    cov = np.mean((y - y.mean()) * (p - p.mean()))
    expected = 2 * cov / (y.var() + p.var() + (y.mean() - p.mean()) ** 2)
    assert _agg(ds, bt.concordance_correlation("y", "p")) == pytest.approx(expected)


def test_concordance_penalises_a_shifted_prediction() -> None:
    # A prediction shifted by a constant has a perfect Pearson correlation but a CCC below 1.
    ds = bt.from_pydict({"y": [1.0, 2.0, 3.0, 4.0], "p": [3.0, 4.0, 5.0, 6.0]})
    assert _agg(ds, bt.concordance_correlation("y", "p")) < 1.0
    assert _agg(ds, bt.corr("y", "p")) == pytest.approx(1.0)


def test_nash_sutcliffe_matches_the_reference(paired) -> None:
    y, p, ds = paired
    expected = 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
    assert _agg(ds, bt.nash_sutcliffe_efficiency("y", "p")) == pytest.approx(expected)


def test_nash_sutcliffe_is_one_for_a_perfect_fit() -> None:
    ds = bt.from_pydict({"y": [1.0, 2.0, 3.0], "p": [1.0, 2.0, 3.0]})
    assert _agg(ds, bt.nash_sutcliffe_efficiency("y", "p")) == pytest.approx(1.0)


def test_kling_gupta_matches_the_reference(paired) -> None:
    y, p, ds = paired
    r = np.corrcoef(y, p)[0, 1]
    expected = 1 - np.sqrt(
        (r - 1) ** 2 + (p.std() / y.std() - 1) ** 2 + (p.mean() / y.mean() - 1) ** 2
    )
    assert _agg(ds, bt.kling_gupta_efficiency("y", "p")) == pytest.approx(expected)


def test_all_agreement_metrics_are_one_for_identical_series() -> None:
    ds = bt.from_pydict({"y": [1.0, 2.0, 3.0, 4.0, 5.0], "p": [1.0, 2.0, 3.0, 4.0, 5.0]})
    assert _agg(ds, bt.concordance_correlation("y", "p")) == pytest.approx(1.0)
    assert _agg(ds, bt.nash_sutcliffe_efficiency("y", "p")) == pytest.approx(1.0)
    assert _agg(ds, bt.kling_gupta_efficiency("y", "p")) == pytest.approx(1.0)


def test_agreement_metrics_compose_with_group_by(paired) -> None:
    _, _, ds = paired
    grouped = (
        ds.with_columns(g=bt.lit("x")).group_by("g").agg(ccc=bt.concordance_correlation("y", "p"))
    )
    assert grouped.count() == 1
