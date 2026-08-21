"""Four metric functions with no test, against scikit-learn and DuckDB.

``rmsle``, ``negative_predictive_value``, ``nunique_ratio`` and ``non_null_rate`` are the
kind of one-line aggregate that looks too small to test and is exactly where a
denominator goes wrong. Each has an independent oracle here:

* ``rmsle`` and ``negative_predictive_value`` against scikit-learn, which defines both.
* ``nunique_ratio`` and ``non_null_rate`` against the SQL they are shorthand for, so the
  null handling is checked rather than assumed -- ``count(DISTINCT x)`` excludes nulls
  while ``count(*)`` does not, and getting that pairing wrong is the whole bug surface.

Every test also covers the degenerate input each metric has: no negative predictions for
``negative_predictive_value``, an all-null column for ``non_null_rate``, and an empty
frame for both ratios.
"""

from __future__ import annotations

import math

import pytest

import batcher as bt

pytestmark = pytest.mark.differential

duckdb = pytest.importorskip("duckdb")


def _scalar(ds, expr) -> float | None:
    return ds.agg(v=expr).to_pydict()["v"][0]


def test_rmsle_matches_scikit_learn():
    """Root mean squared logarithmic error, on values spanning three orders of magnitude."""
    metrics = pytest.importorskip("sklearn.metrics")
    truth = [1.0, 2.0, 3.0, 100.0, 0.0]
    predicted = [1.1, 2.2, 2.7, 90.0, 0.5]
    got = _scalar(
        bt.from_pydict({"y": truth, "p": predicted}),
        bt.rmsle(bt.col("y"), bt.col("p")),
    )
    want = math.sqrt(metrics.mean_squared_log_error(truth, predicted))
    assert got == pytest.approx(want, rel=1e-12)


def test_rmsle_is_zero_for_a_perfect_prediction_and_grows_with_error():
    """The two properties that separate it from a metric computed on the wrong scale."""
    truth = [1.0, 10.0, 100.0]
    perfect = _scalar(bt.from_pydict({"y": truth, "p": truth}), bt.rmsle(bt.col("y"), bt.col("p")))
    assert perfect == pytest.approx(0.0, abs=1e-15)

    close = _scalar(
        bt.from_pydict({"y": truth, "p": [1.1, 11.0, 110.0]}),
        bt.rmsle(bt.col("y"), bt.col("p")),
    )
    far = _scalar(
        bt.from_pydict({"y": truth, "p": [5.0, 50.0, 500.0]}),
        bt.rmsle(bt.col("y"), bt.col("p")),
    )
    assert 0 < close < far, "a larger relative error must give a larger RMSLE"


def test_rmsle_is_the_log1p_definition_and_not_a_scaled_rmse():
    """Computed on ``log1p``, which is what makes a proportional error scale-free.

    The contrast with a plain RMSE is the point: doubling a prediction costs RMSLE about
    ``log(2)`` at any magnitude -- exactly ``log(2)`` in the limit, and the ``+1`` in
    ``log1p`` is why it is only "about" at small values -- while RMSE's cost grows without
    bound. A metric that had quietly become an RMSE passes every value comparison above on
    a single fixture and fails this one by orders of magnitude.
    """
    doubled = [
        _scalar(
            bt.from_pydict({"y": [scale], "p": [2 * scale]}),
            bt.rmsle(bt.col("y"), bt.col("p")),
        )
        for scale in (1e3, 1e6, 1e9)
    ]
    for value in doubled:
        assert value == pytest.approx(math.log(2.0), rel=1e-3), (
            "doubling costs log(2) at every scale a log metric is used at"
        )
    assert max(doubled) - min(doubled) < 1e-3, "the cost must not grow with the magnitude"

    exact = _scalar(
        bt.from_pydict({"y": [1.0, 10.0], "p": [2.0, 12.0]}),
        bt.rmsle(bt.col("y"), bt.col("p")),
    )
    want = math.sqrt(
        ((math.log1p(2.0) - math.log1p(1.0)) ** 2 + (math.log1p(12.0) - math.log1p(10.0)) ** 2) / 2
    )
    assert exact == pytest.approx(want, rel=1e-12), "the log1p definition, term by term"


def test_negative_predictive_value_matches_scikit_learn():
    """``tn / (tn + fn)``, which is precision computed on the negative class."""
    metrics = pytest.importorskip("sklearn.metrics")
    truth = [1, 0, 1, 0, 0, 1, 0]
    predicted = [1, 0, 0, 0, 1, 1, 0]
    got = _scalar(
        bt.from_pydict({"y": truth, "p": predicted}),
        bt.negative_predictive_value(bt.col("y"), bt.col("p")),
    )
    # Precision of the negative class *is* the negative predictive value: scikit-learn
    # computes it by flipping the labels, which is an independent route to the number.
    want = metrics.precision_score(truth, predicted, pos_label=0)
    assert got == pytest.approx(want, rel=1e-12)


def test_negative_predictive_value_honours_the_positive_label():
    """The ``positive`` argument must move the answer, or it is decoration."""
    truth = ["yes", "no", "yes", "no"]
    predicted = ["yes", "no", "no", "no"]
    ds = bt.from_pydict({"y": truth, "p": predicted})
    as_yes = _scalar(ds, bt.negative_predictive_value(bt.col("y"), bt.col("p"), positive="yes"))
    as_no = _scalar(ds, bt.negative_predictive_value(bt.col("y"), bt.col("p"), positive="no"))
    assert as_yes == pytest.approx(2 / 3), "two of the three predicted 'no' are 'no'"
    assert as_no == pytest.approx(1.0), "with the labels flipped, the one predicted 'yes' is"
    assert as_yes != as_no


def test_negative_predictive_value_with_no_negative_prediction_is_zero_not_nan():
    """The project's zero-division convention, which this metric must share.

    ``tn / (tn + fn)`` is 0/0 on a batch where nothing was predicted negative. NaN is
    defensible in isolation and wrong in place: it spreads, so one fold or one streaming
    batch makes the whole score undefined, and on imbalanced data such a batch is
    ordinary. scikit-learn answers 0.0 here and so does every other rate in this engine
    (see ``tests/unit/test_ml_metric_zero_division.py``), so this one has to as well.
    """
    got = _scalar(
        bt.from_pydict({"y": [1, 1, 1], "p": [1, 1, 1]}),
        bt.negative_predictive_value(bt.col("y"), bt.col("p")),
    )
    assert got == 0.0, f"expected the zero-division convention, got {got!r}"
    assert not (isinstance(got, float) and math.isnan(got))


@pytest.fixture(scope="module")
def duck():
    """A DuckDB table with nulls in every column, for the ratio comparisons."""
    con = duckdb.connect()
    con.execute("CREATE TABLE t (a BIGINT, s VARCHAR, z BIGINT)")
    con.executemany(
        "INSERT INTO t VALUES (?, ?, ?)",
        [(1, "x", None), (2, "y", None), (2, "y", None), (None, None, None)],
    )
    return con


ROWS = {"a": [1, 2, 2, None], "s": ["x", "y", "y", None], "z": [None, None, None, None]}


def test_nunique_ratio_matches_the_sql_it_abbreviates(duck):
    """``count(DISTINCT x) / count(*)``, with nulls excluded from the numerator only."""
    ds = bt.from_pydict(ROWS)
    for column in ("a", "s", "z"):
        got = _scalar(ds, bt.nunique_ratio(column))
        want = duck.execute(
            f"SELECT count(DISTINCT {column})::DOUBLE / count(*) FROM t"
        ).fetchone()[0]
        assert got == pytest.approx(want, abs=1e-12), f"nunique_ratio({column})"
    assert _scalar(ds, bt.nunique_ratio("a")) == pytest.approx(0.5), (
        "two distinct values over four rows; the null is not a distinct value"
    )
    assert _scalar(ds, bt.nunique_ratio("z")) == pytest.approx(0.0), "an all-null column"


def test_non_null_rate_matches_the_sql_it_abbreviates(duck):
    """``count(x) / count(*)``, and the complement of the null rate."""
    ds = bt.from_pydict(ROWS)
    for column in ("a", "s", "z"):
        got = _scalar(ds, bt.non_null_rate(column))
        want = duck.execute(f"SELECT count({column})::DOUBLE / count(*) FROM t").fetchone()[0]
        assert got == pytest.approx(want, abs=1e-12), f"non_null_rate({column})"
    assert _scalar(ds, bt.non_null_rate("a")) == pytest.approx(0.75)
    assert _scalar(ds, bt.non_null_rate("z")) == pytest.approx(0.0)


def test_non_null_rate_is_one_minus_the_null_rate(duck):
    """The identity the docstring states, checked against the other function."""
    ds = bt.from_pydict(ROWS)
    for column in ("a", "s", "z"):
        present = _scalar(ds, bt.non_null_rate(column))
        missing = _scalar(ds, bt.null_rate(column))
        assert present + missing == pytest.approx(1.0, abs=1e-12), column


def test_both_ratios_accept_an_expression_as_well_as_a_column_name():
    """The ``str | Expr`` overload, which a name-only implementation would fail."""
    ds = bt.from_pydict(ROWS)
    assert _scalar(ds, bt.non_null_rate("a")) == _scalar(ds, bt.non_null_rate(bt.col("a")))
    assert _scalar(ds, bt.nunique_ratio("a")) == _scalar(ds, bt.nunique_ratio(bt.col("a")))
    derived = _scalar(ds, bt.nunique_ratio(bt.col("a") * 0))
    assert derived == pytest.approx(0.25), "a * 0 collapses to one distinct value over four rows"


def test_the_ratios_over_an_empty_frame_have_no_value_rather_than_zero():
    """Zero rows means an empty denominator, which is undefined rather than zero."""
    empty = bt.from_pydict({"a": []})
    for expr in (bt.non_null_rate("a"), bt.nunique_ratio("a")):
        got = _scalar(empty, expr)
        assert got is None or (isinstance(got, float) and math.isnan(got)), (
            f"an empty frame gave {got!r} rather than an undefined ratio"
        )


def test_the_ratios_work_per_group(duck):
    """Grouped, since a whole-column shortcut that ignores the grouping is easy to write."""
    ds = bt.from_pydict({"g": ["a", "a", "b", "b"], "v": [1, None, 3, 3]})
    got = (
        ds.group_by("g").agg(present=bt.non_null_rate("v"), ratio=bt.nunique_ratio("v")).to_pydict()
    )
    by_group = dict(zip(got["g"], got["present"], strict=True))
    assert by_group["a"] == pytest.approx(0.5)
    assert by_group["b"] == pytest.approx(1.0)
    ratios = dict(zip(got["g"], got["ratio"], strict=True))
    assert ratios["a"] == pytest.approx(0.5)
    assert ratios["b"] == pytest.approx(0.5), "one distinct value over two rows"
