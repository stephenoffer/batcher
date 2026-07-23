"""Every `ds.meta` shortcut must equal executing the query. That is the whole contract.

A metadata shortcut is an optimisation, and an optimisation that changes an answer is not an
optimisation — it is a wrong-answer generator, and the faster it runs the worse it is. So this
file does not assert *what* each shortcut returns (the per-operator differential tests own
that). It asserts the shortcut and the long way round **agree**, for every shortcut, over a
cross-product of the shapes that break things: nulls, an all-null column, NaN, ``-0.0``,
duplicates, a single row, an empty relation, strings, booleans — from memory and from Parquet.

The forcing mechanism is the load-bearing part. An `always-true filter` does **not** work: the
optimizer folds it away and the statistics come back EXACT, so the "forced" path is answered
from metadata too and the comparison is vacuous — a test that passes by comparing a value to
itself. `map_batches` does work: the IR cannot describe a Python callback, so the whole
metadata layer declines (`_metadata_answerable` is False) and every shortcut falls through to
the engine. An identity callback changes no row, so the two datasets are the same relation
computed two ways — which is exactly the pair the contract is about.
"""

from __future__ import annotations

import math
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

pytestmark = pytest.mark.differential

bt = pytest.importorskip("batcher")

#: The shapes that break metadata reasoning, in one table: nulls, an all-null column, NaN
#: (greater than everything in SQL's order, and dropped by every footer), `-0.0` (equal to
#: `0.0` but differently encoded), duplicates, a key, a constant, a boolean, and a string.
TABLE = {
    "i": pa.array([5, 1, 9, 3, None], pa.int64()),
    "key": pa.array([1, 2, 3, 4, 5], pa.int64()),
    "dup": pa.array([2, 2, 2, 7, 7], pa.int64()),
    "const": pa.array([4, 4, 4, 4, 4], pa.int64()),
    "neg": pa.array([-5, -1, -9, -3, -2], pa.int64()),
    "zero": pa.array([0, 0, 0, 0, 0], pa.int64()),
    "f": pa.array([2.5, float("nan"), 0.5, -0.0, None], pa.float64()),
    "f_clean": pa.array([2.5, 0.5, 8.5, 4.0, None], pa.float64()),
    "allnull": pa.array([None, None, None, None, None], pa.int64()),
    "s": pa.array(["pear", "apple", None, "fig", "kiwi"]),
    "b": pa.array([True, False, True, None, True]),
}

NUMERIC = ["i", "key", "dup", "const", "neg", "zero", "f", "f_clean", "allnull"]
EVERY_COLUMN = sorted(TABLE)


def _force(ds):
    """The same relation, computed with the metadata layer switched off.

    `map_batches` is opaque to the IR, so Kyber refuses to reason about the plan at all and
    every shortcut falls back to the engine. The callback is the identity, so the rows are
    untouched — this is the *same* relation, obtained the long way round.
    """
    return ds.map_batches(lambda batch: batch)


def _same(shortcut: Any, executed: Any) -> bool:
    """Whether the two answers agree, treating NaN as equal to itself and ints to floats.

    `nan == nan` is False in IEEE, but "the metadata path and the engine both said NaN" is
    agreement, not disagreement — the comparison has to say so or the NaN cases can never pass.
    """
    if (
        isinstance(shortcut, float)
        and isinstance(executed, float)
        and math.isnan(shortcut)
        and math.isnan(executed)
    ):
        return True
    if isinstance(shortcut, (list, tuple)) and isinstance(executed, (list, tuple)):
        return len(shortcut) == len(executed) and all(
            _same(a, b) for a, b in zip(shortcut, executed, strict=True)
        )
    if isinstance(shortcut, dict) and isinstance(executed, dict):
        return shortcut.keys() == executed.keys() and all(
            _same(shortcut[k], executed[k]) for k in shortcut
        )
    if isinstance(shortcut, (int, float)) and isinstance(executed, (int, float)):
        return math.isclose(float(shortcut), float(executed), rel_tol=1e-9, abs_tol=1e-12)
    return bool(shortcut == executed)


@pytest.fixture(scope="module")
def parquet_path(tmp_path_factory) -> str:
    """The same table on disk, so the footer-bound path is exercised too."""
    path = str(tmp_path_factory.mktemp("meta_shortcuts") / "t.parquet")
    pq.write_table(pa.table(TABLE), path)
    return path


@pytest.fixture(params=["memory", "parquet"])
def ds(request, parquet_path):
    """The same relation, from an in-memory source and from a Parquet file."""
    if request.param == "memory":
        return bt.from_arrow(pa.table(TABLE))
    return bt.read.parquet(parquet_path)


def _assert_agrees(ds, call) -> None:
    """`call` must give the same answer with the metadata layer on and with it off."""
    shortcut = call(ds)
    executed = call(_force(ds))
    assert _same(shortcut, executed), f"metadata said {shortcut!r}, executing said {executed!r}"


# --- relation-level ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda d: d.meta.shape(), id="shape"),
        pytest.param(lambda d: d.meta.count_where(bt.col("i").is_null()), id="count_where-null"),
        pytest.param(lambda d: d.meta.count_where(bt.col("i") > 100), id="count_where-empty"),
        pytest.param(lambda d: d.meta.count_where(bt.col("i") > 2), id="count_where-partial"),
        pytest.param(lambda d: d.meta.is_empty_where(bt.col("i") > 100), id="is_empty_where"),
        pytest.param(lambda d: d.meta.any_match(bt.col("i") > 2), id="any_match"),
        pytest.param(lambda d: d.meta.none_match(bt.col("i") > 100), id="none_match"),
        pytest.param(lambda d: d.meta.all_match(bt.col("key") > 0), id="all_match"),
        pytest.param(lambda d: d.meta.is_key("key"), id="is_key-true"),
        pytest.param(lambda d: d.meta.is_key("dup"), id="is_key-false"),
        pytest.param(lambda d: d.meta.is_key("i"), id="is_key-nullable"),
        pytest.param(lambda d: d.meta.is_key(["dup", "key"]), id="is_key-composite"),
        pytest.param(lambda d: d.meta.nulls.counts(), id="nulls-counts"),
        pytest.param(lambda d: d.meta.nulls.fractions(), id="nulls-fractions"),
        pytest.param(lambda d: d.meta.nulls.total(), id="nulls-total"),
        pytest.param(lambda d: d.meta.nulls.any(), id="nulls-any"),
        pytest.param(lambda d: d.meta.nulls.is_complete(), id="nulls-is_complete"),
        pytest.param(lambda d: d.meta.nulls.columns_with_nulls(), id="nulls-with"),
        pytest.param(lambda d: d.meta.nulls.complete_columns(), id="nulls-complete"),
    ],
)
def test_relation_shortcut_equals_execution(ds, call):
    """A relation-level shortcut and the executed query must agree."""
    _assert_agrees(ds, call)


# --- per column -------------------------------------------------------------------------

COLUMN_SHORTCUTS = {
    "bounds": lambda c: c.bounds(),
    "n_unique": lambda c: c.n_unique(),
    "is_unique": lambda c: c.is_unique(),
    "has_duplicates": lambda c: c.has_duplicates(),
    "duplicate_count": lambda c: c.duplicate_count(),
    "is_key": lambda c: c.is_key(),
    "is_constant": lambda c: c.is_constant(),
    "constant_value": lambda c: c.constant_value(),
    "is_low_cardinality": lambda c: c.is_low_cardinality(3),
    "is_binary_valued": lambda c: c.is_binary_valued(),
    "null_fraction": lambda c: c.null_fraction(),
    "no_nulls": lambda c: c.no_nulls(),
    # `summary` minus its `dtype` entry. `dtype` is a pure *schema* fact — never scanned — and
    # the forcing mechanism (`map_batches`, an opaque IR node) deliberately erases the output
    # schema to `null`, so it is the one field this cross-check cannot compare. The dtype is
    # covered on its own by `test_diff_metadata_shortcuts`'s schema cases; here we compare the
    # value facets, which are what the metadata path actually computes.
    "summary": lambda c: {k: v for k, v in c.summary().items() if k != "dtype"},
}

NUMERIC_SHORTCUTS = {
    "range": lambda c: c.range(),
    "midpoint": lambda c: c.midpoint(),
    "abs_max": lambda c: c.abs_max(),
    "sum": lambda c: c.sum(),
    "mean": lambda c: c.mean(),
}


@pytest.mark.parametrize("column", EVERY_COLUMN)
@pytest.mark.parametrize("shortcut", sorted(COLUMN_SHORTCUTS))
def test_column_shortcut_equals_execution(ds, column, shortcut):
    """Every column shortcut, over every column, must equal executing it."""
    _assert_agrees(ds, lambda d: COLUMN_SHORTCUTS[shortcut](d.meta.col(column)))


@pytest.mark.parametrize("column", NUMERIC)
@pytest.mark.parametrize("shortcut", sorted(NUMERIC_SHORTCUTS))
def test_numeric_column_shortcut_equals_execution(ds, column, shortcut):
    """The numeric-only column shortcuts (range, midpoint, sum, …) likewise."""
    _assert_agrees(ds, lambda d: NUMERIC_SHORTCUTS[shortcut](d.meta.col(column)))


# --- the predicate checks ----------------------------------------------------------------

CHECKS = {
    "all_greater_than": lambda c: c.all_greater_than(0),
    "all_greater_equal": lambda c: c.all_greater_equal(0),
    "all_less_than": lambda c: c.all_less_than(3),
    "all_less_equal": lambda c: c.all_less_equal(3),
    "all_between": lambda c: c.all_between(0, 5),
    "all_positive": lambda c: c.all_positive(),
    "all_non_negative": lambda c: c.all_non_negative(),
    "all_negative": lambda c: c.all_negative(),
    "all_non_positive": lambda c: c.all_non_positive(),
    "all_zero": lambda c: c.all_zero(),
    "any_greater_than": lambda c: c.any_greater_than(3),
    "any_greater_equal": lambda c: c.any_greater_equal(3),
    "any_less_than": lambda c: c.any_less_than(1),
    "any_less_equal": lambda c: c.any_less_equal(1),
    "contains": lambda c: c.contains(3),
    "contains-absent": lambda c: c.contains(9999),
    "never_equals": lambda c: c.never_equals(9999),
    "any_in": lambda c: c.any_in([3, 4]),
    "any_in-absent": lambda c: c.any_in([9998, 9999]),
    "none_in": lambda c: c.none_in([9998, 9999]),
}


@pytest.mark.parametrize("column", NUMERIC)
@pytest.mark.parametrize("check", sorted(CHECKS))
def test_check_equals_execution(ds, column, check):
    """Every bound-derived predicate check must equal running the filter that decides it."""
    _assert_agrees(ds, lambda d: CHECKS[check](d.meta.col(column).check))


def test_may_contain_never_refutes_a_present_value(ds):
    """`may_contain` may say "maybe" for an absent value, but never "no" for a present one.

    The one-sided guarantee the whole pruning story rests on: a `False` authorises skipping a
    file, so a `False` for a value that *is* there would silently drop rows.
    """
    for value in [5, 1, 9, 3]:
        assert ds.meta.col("i").check.may_contain(value) is True
    assert ds.meta.col("i").check.may_contain(9999) is False


# --- joins -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("right", "expected_empty"),
    [([900, 901], True), ([3, 4], False), ([-100, -99], True)],
)
def test_join_emptiness_equals_execution(ds, right, expected_empty):
    """A disjoint key range must prove the join empty, and agree with running the join."""
    other = bt.from_pydict({"key": right})
    shortcut = ds.meta.against(other).join_is_empty("key")
    executed = _force(ds).join(other, on="key", how="inner").is_empty()
    assert shortcut == executed == expected_empty


def test_overlaps_is_the_complement_of_join_is_empty(ds):
    """`overlaps` and `join_is_empty` must never both be true (or both be false)."""
    other = bt.from_pydict({"key": [3, 4, 5]})
    pair = ds.meta.against(other)
    assert pair.overlaps("key") is not pair.join_is_empty("key")


def test_overlapping_key_ranges_that_share_no_value_still_join_empty():
    """An overlap proves nothing, and claiming otherwise is a *wrong* answer, not a weak one.

    Left holds `{1, 5}` and right holds `{3}`: both live inside `[1, 5]`, so the bounds overlap
    — and the join is still empty, because no value is shared. A `join_is_empty` that reported
    `False` from the overlap (as it briefly did) would have told the caller the join matches.
    Only disjointness is a proof; everything else runs the join.
    """
    left = bt.from_pydict({"key": [1, 5]})
    right = bt.from_pydict({"key": [3]})
    assert left.meta.against(right).join_is_empty("key") is True
    assert left.join(right, on="key", how="inner").is_empty() is True
    assert left.meta.against(right).overlaps("key") is False


# --- edge relations ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "table",
    [
        pytest.param({"x": pa.array([], pa.int64())}, id="empty"),
        pytest.param({"x": pa.array([7], pa.int64())}, id="single-row"),
        pytest.param({"x": pa.array([None, None], pa.int64())}, id="all-null"),
        pytest.param({"x": pa.array([0, -0.0], pa.float64())}, id="signed-zero"),
    ],
)
@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda d: d.meta.col("x").bounds(), id="bounds"),
        pytest.param(lambda d: d.meta.col("x").n_unique(), id="n_unique"),
        pytest.param(lambda d: d.meta.col("x").is_unique(), id="is_unique"),
        pytest.param(lambda d: d.meta.col("x").is_constant(), id="is_constant"),
        pytest.param(lambda d: d.meta.col("x").null_fraction(), id="null_fraction"),
        pytest.param(lambda d: d.meta.col("x").check.all_positive(), id="all_positive"),
        pytest.param(lambda d: d.meta.col("x").check.all_non_negative(), id="all_non_negative"),
        pytest.param(lambda d: d.meta.col("x").check.any_greater_than(0), id="any_greater_than"),
        pytest.param(lambda d: d.meta.col("x").check.contains(7), id="contains"),
        pytest.param(lambda d: d.meta.nulls.counts(), id="null-counts"),
        pytest.param(lambda d: d.meta.shape(), id="shape"),
    ],
)
def test_edge_relation_shortcut_equals_execution(table, call):
    """The empty / single-row / all-null / signed-zero relations, where the rules bite."""
    _assert_agrees(bt.from_arrow(pa.table(table)), call)


# --- and both agree with the oracle -------------------------------------------------------


@pytest.mark.parametrize("column", ["i", "key", "dup", "const", "f_clean", "s"])
def test_shortcuts_match_duckdb(duck, column):
    """The shortcuts must agree with DuckDB, not merely with our own executor."""
    table = pa.table(TABLE)
    duck.register("t", table)
    ds = bt.from_arrow(table)
    meta = ds.meta.col(column)

    want = duck.sql(
        f"SELECT min({column}) AS lo, max({column}) AS hi, "
        f"count({column}) AS n, count(DISTINCT {column}) AS d, "
        f"count(*) - count({column}) AS nulls FROM t"
    ).fetchone()
    lo, hi, non_null, ndv, nulls = want

    assert _same(meta.bounds(), (lo, hi))
    assert meta.n_unique() == ndv
    assert meta.is_unique() == (ndv == non_null)
    assert meta.has_duplicates() == (ndv != non_null)
    assert meta.duplicate_count() == non_null - ndv
    assert meta.is_key() == (ndv == non_null and nulls == 0)
    assert ds.meta.nulls.counts()[column] == nulls


@pytest.mark.parametrize("column", ["i", "key", "dup", "const", "neg", "zero", "f_clean"])
def test_numeric_shortcuts_match_duckdb(duck, column):
    """The additive and bound-derived numeric shortcuts, against the oracle."""
    table = pa.table(TABLE)
    duck.register("t", table)
    meta = bt.from_arrow(table).meta.col(column)

    total, avg, lo, hi = duck.sql(
        f"SELECT sum({column}), avg({column}), min({column}), max({column}) FROM t"
    ).fetchone()

    assert _same(meta.sum(), total)
    assert _same(meta.mean(), avg)
    assert _same(meta.range(), hi - lo)
    assert _same(meta.midpoint(), (lo + hi) / 2)
    assert meta.check.all_positive() == (lo > 0)
    assert meta.check.all_negative() == (hi < 0)
    assert meta.check.any_greater_than(0) == (hi > 0)
