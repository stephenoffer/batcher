"""`evaluate(weight=..., support=True)` against scikit-learn and DuckDB (AP-413, AP-419).

scikit-learn's ``sample_weight`` is the oracle for the weighted metrics, DuckDB's weighted
``SUM`` the oracle for the same sums written as SQL, and DuckDB's ``GROUP BY count(*)`` the
oracle for the per-slice support counts. Nulls, an empty slice and a one-row slice are in
the fixture because they are where a weighted or counted metric goes wrong.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.ml.metrics import evaluate

sk = pytest.importorskip("sklearn.metrics", reason="scikit-learn is the metrics oracle")
duckdb = pytest.importorskip("duckdb")

pytestmark = pytest.mark.differential

_RNG = np.random.default_rng(7)
_N = 400
_Y = _RNG.integers(0, 2, _N)
_P = np.where(_RNG.random(_N) < 0.7, _Y, 1 - _Y)
_W = _RNG.uniform(0.1, 5.0, _N)
_T = _RNG.normal(10.0, 3.0, _N)
_R = _T + _RNG.normal(0.0, 1.0, _N)


def _classification() -> bt.Dataset:
    return bt.from_pydict({"y": _Y.tolist(), "p": _P.tolist(), "w": _W.tolist()})


def _regression() -> bt.Dataset:
    return bt.from_pydict({"y": _T.tolist(), "p": _R.tolist(), "w": _W.tolist()})


@pytest.mark.parametrize(
    ("name", "oracle"),
    [
        ("accuracy", sk.accuracy_score),
        ("precision", sk.precision_score),
        ("recall", sk.recall_score),
        ("f1", sk.f1_score),
        ("balanced_accuracy", sk.balanced_accuracy_score),
    ],
)
def test_weighted_classification_metrics_match_sklearn_sample_weight(name, oracle):
    got = evaluate(_classification(), "y", y_pred="p", task="binary", metrics=[name], weight="w")
    assert got[name] == pytest.approx(oracle(_Y, _P, sample_weight=_W), rel=1e-9)


@pytest.mark.parametrize(
    ("name", "oracle"),
    [
        ("mse", sk.mean_squared_error),
        ("mae", sk.mean_absolute_error),
        ("r2", sk.r2_score),
    ],
)
def test_weighted_regression_metrics_match_sklearn_sample_weight(name, oracle):
    got = evaluate(_regression(), "y", y_pred="p", task="regression", metrics=[name], weight="w")
    assert got[name] == pytest.approx(oracle(_T, _R, sample_weight=_W), rel=1e-9)


def test_weighted_rmse_is_the_root_of_weighted_mse():
    got = evaluate(_regression(), "y", y_pred="p", task="regression", metrics=["rmse"], weight="w")
    want = math.sqrt(sk.mean_squared_error(_T, _R, sample_weight=_W))
    assert got["rmse"] == pytest.approx(want, rel=1e-9)


def test_weighted_mae_matches_a_duckdb_weighted_sum_with_nulls_left_out():
    table = {
        "y": [1.0, 2.0, None, 4.0, 5.0],
        "p": [1.5, None, 3.0, 3.0, 5.0],
        "w": [2.0, 1.0, 1.0, None, 3.0],
    }
    got = evaluate(
        bt.from_pydict(table), "y", y_pred="p", task="regression", metrics=["mae"], weight="w"
    )
    import pyarrow as pa

    arrow = pa.table(table)  # noqa: F841 - read by DuckDB's replacement scan
    want = duckdb.sql(
        "SELECT sum(w * abs(y - p)) / sum(w) FROM arrow "
        "WHERE y IS NOT NULL AND p IS NOT NULL AND w IS NOT NULL"
    ).fetchone()[0]
    assert got["mae"] == pytest.approx(want)


def test_a_metric_with_no_weighted_form_is_refused_rather_than_unweighted():
    with pytest.raises(PlanError, match="no weighted form"):
        evaluate(_classification(), "y", y_pred="p", task="binary", metrics=["mcc"], weight="w")


def test_the_default_set_under_weight_is_the_weighted_subset():
    got = evaluate(_classification(), "y", y_pred="p", task="binary", weight="w")
    assert list(got) == ["accuracy", "precision", "recall", "f1", "balanced_accuracy"]


# --- support ----------------------------------------------------------------------------


def _sliced() -> dict:
    return {
        "g": ["a", "a", "a", "b", "c", "c"],
        "y": [1, 0, 1, 0, 1, None],
        "p": [1, 0, 0, 0, None, 1],
    }


def test_support_counts_match_a_duckdb_group_by_count():
    table = _sliced()
    got = evaluate(
        bt.from_pydict(table),
        "y",
        y_pred="p",
        task="binary",
        metrics=["precision"],
        by="g",
        support=True,
    ).sort("g")
    import pyarrow as pa

    arrow = pa.table(table)  # noqa: F841 - read by DuckDB's replacement scan
    want = duckdb.sql(
        "SELECT g, count(*) FILTER (WHERE y IS NOT NULL AND p IS NOT NULL) AS n, "
        "count(*) FILTER (WHERE y IS NOT NULL AND p IS NOT NULL AND y = 1) AS n_positive "
        "FROM arrow GROUP BY g ORDER BY g"
    ).fetchall()
    rows = got.to_pydict()
    assert list(zip(rows["g"], rows["n"], rows["n_positive"], strict=True)) == want


def test_an_empty_slice_is_distinguishable_from_a_measured_zero():
    rows = (
        evaluate(
            bt.from_pydict(_sliced()),
            "y",
            y_pred="p",
            task="binary",
            metrics=["precision"],
            by="g",
            support=True,
        )
        .sort("g")
        .to_pydict()
    )
    # Slice c has no row with both a label and a prediction, so it measured nothing; slice
    # b has one, and its 0.0 is the zero_division convention, not a measured precision.
    assert rows["n"][2] == 0
    assert rows["precision"][1] == 0.0 and rows["n"][1] == 1 and rows["n_positive"][1] == 0


def test_support_off_by_default_leaves_the_report_unchanged():
    got = evaluate(_classification(), "y", y_pred="p", task="binary", metrics=["recall"])
    assert list(got) == ["recall"]


def test_support_for_regression_has_no_positive_count():
    got = evaluate(_regression(), "y", y_pred="p", task="regression", metrics=["mae"], support=True)
    assert set(got) == {"mae", "n"}
    assert got["n"] == _N
