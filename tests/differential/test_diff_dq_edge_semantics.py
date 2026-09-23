"""`ds.dq` on the edges where a quiet wrong answer used to look like a pass.

Each test here pins a behaviour an audit found broken, on the inputs that exposed it: NaN,
infinity, an empty relation, an all-null column, and a single row.

- A relation-level bound over a NaN measurement must fail. ``avg`` over a column holding a
  NaN is NaN, and every comparison against NaN is false, so a bounds check that tested
  ``value < low`` and ``value > high`` passed it. DuckDB is the oracle: ``avg(f) BETWEEN lo
  AND hi`` is the question the constraint asks.
- ``suggest()`` promises every proposal holds on the data it was read from, so it may only
  propose ``is_finite`` for a column that has no NaN or infinity. DuckDB's ``isfinite``
  counts the offenders.
- ``suggest("id")`` is a one-column typo, not a request for the columns ``'i'`` and ``'d'``.
- ``rows`` is the relation's row count on every path, including the one where metadata
  proved the contract without executing it. DuckDB's ``count(*)`` is the oracle.
- ``to_dict()`` is JSON-safe, so ``json.dumps(..., allow_nan=False)`` accepts it.
- A relation-level result that failed has a pass rate of 0.0, not 1.0.
"""

from __future__ import annotations

import json
import math

import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import PlanError

pytestmark = pytest.mark.differential

NAN = float("nan")
INF = float("inf")

# (label, float column values), covering the edges the audit named.
_EDGES = [
    ("nan", [1.0, NAN, 3.0]),
    ("inf", [1.0, INF, 3.0]),
    ("neg_inf", [1.0, -INF]),
    ("finite", [1.0, 2.0, 3.0]),
    ("single_row", [4.0]),
    ("single_nan", [NAN]),
    ("all_null", [None, None]),
    ("empty", []),
]


def _table(values: list[float | None]) -> pa.Table:
    return pa.table({"f": pa.array(values, type=pa.float64())})


@pytest.mark.parametrize(("label", "values"), _EDGES, ids=[e[0] for e in _EDGES])
def test_mean_between_agrees_with_duckdb_between(duck, label, values):
    """``mean_between(f, 0, 10)`` passes exactly when DuckDB's ``avg(f) BETWEEN 0 AND 10``
    is TRUE. A NULL (empty or all-null) or NaN measurement is not TRUE, so it fails."""
    t = _table(values)
    duck.register("t", t)
    expected = duck.execute("SELECT avg(f) BETWEEN 0 AND 10 FROM t").fetchone()[0]
    report = bt.from_arrow(t).dq.mean_between("f", 0, 10).validate()
    assert report.ok is (expected is True), (label, expected, report.results[0].value)


def test_a_nan_mean_fails_even_a_one_sided_bound():
    """A NaN measurement cannot be evaluated, so it fails like a NULL one does.

    DuckDB orders NaN above every number, so ``NaN >= 0`` is TRUE there. The constraint
    deliberately does not follow that for a one-sided bound: a mean that is NaN says the
    column holds a NaN, and "the mean is at least 0" has not been shown.
    """
    ds = bt.from_pydict({"f": [1.0, NAN]})
    assert not ds.dq.mean_between("f", 0).validate().ok
    assert not ds.dq.mean_between("f", None, 10).validate().ok


@pytest.mark.parametrize(("label", "values"), _EDGES, ids=[e[0] for e in _EDGES])
def test_suggest_proposes_is_finite_only_for_a_finite_column(duck, label, values):
    t = _table(values)
    duck.register("t", t)
    offenders = duck.execute("SELECT count(*) FROM t WHERE NOT isfinite(f)").fetchone()[0]
    proposed = bt.from_arrow(t).dq.suggest()
    report = proposed.validate()
    assert ("is_finite(f)" in report.violations) is (offenders == 0), label
    assert report.ok, (label, report.violations)


def test_suggest_rejects_a_bare_column_name():
    ds = bt.from_pydict({"id": [1, 2, 3]})
    with pytest.raises(PlanError, match="string 'id'"):
        ds.dq.suggest("id")
    # the list form still works, and is what the error tells you to type
    assert "unique(id)" in ds.dq.suggest(["id"]).validate().violations


@pytest.mark.parametrize(
    "values", [[1, 2], [7], [1, None, 3], []], ids=["two", "single", "with_null", "empty"]
)
def test_rows_is_the_relation_row_count_on_every_path(duck, values):
    """The metadata-proved path used to report ``rows == 0`` for a clean relation."""
    t = pa.table({"d": pa.array(values, type=pa.int64())})
    duck.register("t", t)
    expected = duck.execute("SELECT count(*) FROM t").fetchone()[0]
    ds = bt.from_arrow(t)
    for chain in (ds.dq.not_null("d"), ds.dq.in_range("d", 0, 10), ds.dq.unique("d")):
        report = chain.validate()
        assert report.rows == expected, (values, report)
        assert all(r.rows == expected for r in report.results)


def test_the_proved_clean_path_reports_rows():
    """Pin the exact case the audit found, where `validate` executes nothing per constraint."""
    report = bt.from_pydict({"d": [1, 2]}).dq.not_null("d").validate()
    assert report.ok
    assert report.rows == 2
    assert report.results[0].rows == 2


@pytest.mark.parametrize(
    "values", [[1.0, NAN], [1.0, INF], [-INF], [NAN]], ids=["nan", "inf", "neg_inf", "only_nan"]
)
def test_to_dict_is_strict_json(values):
    """A non-finite measurement is emitted as None, so a strict JSON sink accepts it."""
    report = bt.from_pydict({"f": values}).dq.mean_between("f", 0, 10).validate()
    assert not math.isfinite(report.results[0].value)  # the attribute keeps the real value
    payload = json.loads(json.dumps(report.to_dict(), allow_nan=False))
    assert payload["constraints"][0]["value"] is None
    assert payload["ok"] is False


def test_to_dict_keeps_a_finite_measurement():
    report = bt.from_pydict({"f": [1.0, 3.0]}).dq.mean_between("f", 0, 10).validate()
    assert report.to_dict()["constraints"][0]["value"] == 2.0


def test_a_failed_aggregate_has_zero_pass_rate_and_the_relation_row_count(duck):
    t = pa.table({"f": pa.array([100.0, 200.0, 300.0])})
    duck.register("t", t)
    expected = duck.execute("SELECT count(*) FROM t").fetchone()[0]
    ds = bt.from_arrow(t)
    failed = ds.dq.mean_between("f", 0, 10).validate().results[0]
    assert (failed.ok, failed.pass_rate, failed.rows) == (False, 0.0, expected)
    passed = ds.dq.mean_between("f", 0, 1000).validate().results[0]
    assert (passed.ok, passed.pass_rate, passed.rows) == (True, 1.0, expected)


def test_an_aggregate_over_an_empty_relation_fails_with_zero_pass_rate():
    """An empty relation's mean is NULL: the contract could not be evaluated, so it failed."""
    empty = bt.from_arrow(_table([]))
    result = empty.dq.mean_between("f", 0, 10).validate().results[0]
    assert (result.ok, result.pass_rate, result.rows, result.value) == (False, 0.0, 0, None)
    assert json.dumps(result.to_dict(), allow_nan=False)


def test_a_share_of_rows_over_an_empty_relation_agrees_with_its_row_level_check():
    """``null_rate`` and the distinct ratio are 0/0 over no rows, which the engine makes NaN.

    Under the NaN rule above that would fail, while the row-level `not_null` and `unique`
    pass the same empty relation with zero violations. The share takes the value that
    agrees with them, so the two spellings of one contract give one answer.
    """
    empty = bt.from_arrow(pa.table({"id": pa.array([], type=pa.int64())}))
    rows = empty.dq.not_null("id").unique("id").validate()
    shares = empty.dq.null_rate_below("id", 0.0).unique_ratio_above("id", 1.0).validate()
    assert rows.ok and shares.ok, (rows.violations, shares.violations)
    assert [r.value for r in shares.results] == [0.0, 1.0]
    # a non-empty all-null column is still measured, not excused
    nulls = bt.from_pydict({"id": pa.array([None, None], type=pa.int64())})
    assert not nulls.dq.null_rate_below("id", 0.5).validate().ok
