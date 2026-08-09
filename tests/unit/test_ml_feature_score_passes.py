"""The univariate scorers must cost one pass, whatever the table's width.

`f_classif_scores` and `f_regression_scores` used to loop over the features and run one
aggregate each, so screening a hundred-column table scanned it a hundred times. That is
invisible on a toy frame and is the whole cost on a real one — and on the distributed path
each of those scans is a full pass across the cluster.

The pass count is therefore pinned here rather than left to a benchmark nobody runs. The
equivalence tests beside it are what make the optimisation safe: the batched statistic has
to equal the one-feature-at-a-time `stats.anova_f` it replaced, including on the columns
where the two could plausibly diverge — a constant column, and one with nulls.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import batcher as bt
from batcher.api.dataset.frame import Dataset
from batcher.ml.feature_scores import (
    chi2_scores,
    f_classif_scores,
    f_regression_scores,
    mutual_info_scores,
)
from batcher.ml.stats import anova_f

pytestmark = pytest.mark.unit


@pytest.fixture
def counted(monkeypatch) -> list[int]:
    """Count how many times a query is actually executed."""
    tally = [0]
    original = Dataset.collect

    def counting(self, *args, **kwargs):
        tally[0] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Dataset, "collect", counting)
    return tally


def _frame(width: int, rows: int = 200) -> tuple[bt.Dataset, list[str]]:
    rng = np.random.default_rng(0)
    groups = rng.integers(0, 3, size=rows)
    data = {f"f{i}": rng.normal(size=rows).tolist() for i in range(width)}
    data["y_cls"] = groups.tolist()
    data["y_num"] = rng.normal(size=rows).tolist()
    return bt.from_pydict(data), [f"f{i}" for i in range(width)]


@pytest.mark.parametrize("width", [1, 5, 25])
def test_f_classif_costs_one_pass_whatever_the_width(width: int, counted) -> None:
    ds, features = _frame(width)
    counted[0] = 0
    f_classif_scores(ds, "y_cls", features)
    assert counted[0] == 1


@pytest.mark.parametrize("width", [1, 5, 25])
def test_f_regression_costs_one_pass_whatever_the_width(width: int, counted) -> None:
    ds, features = _frame(width)
    counted[0] = 0
    f_regression_scores(ds, "y_num", features)
    assert counted[0] == 1


def test_the_batched_anova_equals_the_one_at_a_time_statistic() -> None:
    """Including the two columns where a shortcut formula could plausibly diverge."""
    rng = np.random.default_rng(7)
    rows = 400
    groups = rng.integers(0, 4, size=rows)
    ds = bt.from_pydict(
        {
            "y": groups.tolist(),
            "signal": (groups * 5.0 + rng.normal(size=rows)).tolist(),
            "noise": rng.normal(size=rows).tolist(),
            "constant": [3.0] * rows,
            "holey": [None if i % 7 == 0 else float(rng.normal()) for i in range(rows)],
        }
    )
    features = ["signal", "noise", "constant", "holey"]
    batched = f_classif_scores(ds, "y", features)
    for name in features:
        alone = anova_f(ds, name, "y")
        if math.isnan(alone):
            assert math.isnan(batched[name])
        else:
            assert batched[name] == pytest.approx(alone, rel=1e-9)


def test_f_regression_matches_the_definition() -> None:
    rng = np.random.default_rng(3)
    rows = 300
    x = rng.normal(size=rows)
    y = 2.0 * x + rng.normal(size=rows)
    ds = bt.from_pydict({"x": x.tolist(), "noise": rng.normal(size=rows).tolist(), "y": y.tolist()})
    got = f_regression_scores(ds, "y", ["x", "noise"])
    for name in ("x", "noise"):
        r = float(np.corrcoef(np.asarray(ds.to_pydict()[name]), y)[0, 1])
        assert got[name] == pytest.approx(r**2 / (1 - r**2) * (rows - 2), rel=1e-7)


def test_a_constant_feature_scores_nan_rather_than_infinity() -> None:
    """Zero within-group variance is undefined, not infinitely significant."""
    ds = bt.from_pydict({"y": [0, 0, 1, 1], "flat": [2.0, 2.0, 2.0, 2.0]})
    assert math.isnan(f_classif_scores(ds, "y", ["flat"])["flat"])


def test_a_single_class_scores_nan() -> None:
    ds = bt.from_pydict({"y": [1, 1, 1], "x": [1.0, 2.0, 3.0]})
    assert math.isnan(f_classif_scores(ds, "y", ["x"])["x"])


def test_a_group_the_feature_is_entirely_null_in_is_not_counted() -> None:
    """Such a group would have contributed no rows to the one-at-a-time filter either."""
    ds = bt.from_pydict(
        {
            "y": ["a", "a", "b", "b", "c", "c"],
            "x": [1.0, 2.0, 9.0, 10.0, None, None],
        }
    )
    assert f_classif_scores(ds, "y", ["x"])["x"] == pytest.approx(anova_f(ds, "x", "y"), rel=1e-9)


def test_an_empty_feature_list_costs_no_pass(counted) -> None:
    ds, _ = _frame(3)
    counted[0] = 0
    assert f_classif_scores(ds, "y_cls", []) == {}
    assert f_regression_scores(ds, "y_num", []) == {}
    assert counted[0] == 0


def test_the_scores_still_rank_signal_above_noise() -> None:
    rng = np.random.default_rng(11)
    rows = 300
    groups = rng.integers(0, 2, size=rows)
    ds = bt.from_pydict(
        {
            "y": groups.tolist(),
            "signal": (groups * 8.0 + rng.normal(size=rows)).tolist(),
            "noise": rng.normal(size=rows).tolist(),
        }
    )
    scores = f_classif_scores(ds, "y", ["signal", "noise"])
    assert scores["signal"] > scores["noise"]


def test_the_categorical_scorers_are_unchanged() -> None:
    """chi2 and mutual information still build a contingency grid per feature."""
    ds = bt.from_pydict(
        {"y": ["p", "p", "q", "q"], "linked": ["a", "a", "b", "b"], "free": ["a", "b", "a", "b"]}
    )
    assert chi2_scores(ds, "y")["linked"] > chi2_scores(ds, "y")["free"]
    assert mutual_info_scores(ds, "y")["linked"] > mutual_info_scores(ds, "y")["free"]
