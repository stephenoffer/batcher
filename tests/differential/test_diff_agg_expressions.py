"""Differential tests for expressions over aggregates and the ``regr_*`` family.

`group_by().agg()` now accepts a whole expression over aggregates (``sum(x)/sum(y)``),
which the control plane lowers to one mergeable aggregate pass plus a projection. The
oracle for that rewrite is DuckDB: the same query in SQL must give the same answer, over
groups, the whole table, and NULLs. A separate check proves the rewrite is
partition-independent (single-node == multi-partition), since the feature's whole promise
is that it stays distributed-safe.

The regression aggregates (`regr_slope`/`regr_intercept`/`regr_r2`/...) are built on that
feature; they are checked against DuckDB's ``regr_*`` over groups with at least two paired
rows (the fit is undefined below that, where the engines legitimately differ: NaN vs NULL).
"""

from __future__ import annotations

import math

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same, assert_tables_equal
from batcher import col
from batcher._internal.errors import PlanError

pytestmark = pytest.mark.differential


@pytest.fixture
def sales(duck):
    tbl = pa.table(
        {
            "g": ["a", "a", "a", "b", "b", "c"],
            "x": [10.0, 20.0, 30.0, 5.0, None, 8.0],
            "y": [1.0, 2.0, 3.0, None, 4.0, 2.0],
        }
    )
    duck.register("t", tbl)
    return tbl


def test_aggregate_expressions_match_duckdb(duck, sales):
    out = (
        bt.from_arrow(sales)
        .group_by("g")
        .agg(
            ratio=col("x").sum() / col("y").sum(),
            spread=col("x").max() - col("x").min(),
            avg=col("x").sum() / col("x").count(),
            shifted=col("x").mean() + 100,
        )
    )
    expected = duck.sql(
        "SELECT g, "
        "sum(x)::double / sum(y) AS ratio, "
        "max(x) - min(x) AS spread, "
        "sum(x)::double / count(x) AS avg, "
        "avg(x) + 100 AS shifted "
        "FROM t GROUP BY g"
    )
    assert_same(out.to_arrow(), expected)


def test_whole_table_aggregate_expression_matches_duckdb(duck, sales):
    out = bt.from_arrow(sales).group_by().agg(r=col("x").sum() / col("y").sum())
    expected = duck.sql("SELECT sum(x)::double / sum(y) AS r FROM t")
    assert_same(out.to_arrow(), expected)


@pytest.fixture
def fit(duck):
    # Every group has >= 2 non-null (x, y) pairs, so the least-squares fit is defined
    # in both engines (below that DuckDB returns NULL and Batcher NaN by design).
    tbl = pa.table(
        {
            "g": ["a", "a", "a", "a", "b", "b", "b"],
            "x": [1.0, 2.0, 3.0, None, 2.0, 4.0, 6.0],
            "y": [2.0, 4.0, 5.0, 9.0, 1.0, 2.0, 4.0],
        }
    )
    duck.register("t", tbl)
    return tbl


def test_regression_family_matches_duckdb(duck, fit):
    out = (
        bt.from_arrow(fit)
        .group_by("g")
        .agg(
            slope=bt.regr_slope(col("y"), col("x")),
            intercept=bt.regr_intercept(col("y"), col("x")),
            r2=bt.regr_r2(col("y"), col("x")),
            n=bt.regr_count(col("y"), col("x")),
            avgx=bt.regr_avgx(col("y"), col("x")),
            avgy=bt.regr_avgy(col("y"), col("x")),
            sxx=bt.regr_sxx(col("y"), col("x")),
            syy=bt.regr_syy(col("y"), col("x")),
            sxy=bt.regr_sxy(col("y"), col("x")),
        )
    )
    expected = duck.sql(
        "SELECT g, "
        "regr_slope(y, x) AS slope, regr_intercept(y, x) AS intercept, "
        "regr_r2(y, x) AS r2, regr_count(y, x) AS n, "
        "regr_avgx(y, x) AS avgx, regr_avgy(y, x) AS avgy, "
        "regr_sxx(y, x) AS sxx, regr_syy(y, x) AS syy, regr_sxy(y, x) AS sxy "
        "FROM t GROUP BY g"
    )
    assert_same(out.to_arrow(), expected)


def test_aggregate_expression_is_partition_independent():
    import random

    rng = random.Random(7)
    n = 1500
    tbl = pa.table(
        {
            "g": [rng.choice(["a", "b", "c", "d"]) for _ in range(n)],
            "x": [float(rng.randint(1, 50)) for _ in range(n)],
            "y": [float(rng.randint(1, 50)) for _ in range(n)],
        }
    )

    def query(ds):
        return (
            ds.group_by("g")
            .agg(
                ratio=col("x").sum() / col("y").sum(),
                slope=bt.regr_slope(col("y"), col("x")),
            )
            .sort("g")
        )

    single = query(bt.from_arrow(tbl)).to_pydict()
    distributed = query(bt.from_arrow(tbl)).collect(distributed=True, num_workers=4).to_pydict()
    assert single["g"] == distributed["g"]
    for key in ("ratio", "slope"):
        for a, b in zip(single[key], distributed[key], strict=True):
            assert math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)


def test_a_scalar_only_aggregate_is_still_rejected():
    ds = bt.from_pydict({"g": ["a"], "x": [1.0]})
    # A scalar expression with no aggregate is not an aggregation.
    with pytest.raises(PlanError, match="aggregate"):
        ds.group_by("g").agg(bad=col("x") + 1)


def test_an_aggregate_expression_in_a_select_is_the_whole_frame_aggregation():
    """This used to be the other half of the rejection test above.

    An expression over aggregates in a `select` no longer fails to lower: every item is
    an aggregate, so the projection *is* a whole-frame aggregation and returns one row —
    what `SELECT sum(x) / sum(x) FROM t` means, and what Polars and pandas answer.
    """
    ds = bt.from_pydict({"g": ["a", "b"], "x": [1.0, 3.0]})
    assert ds.select(r=col("x").sum() / col("x").sum()).to_pydict() == {"r": [1.0]}
    assert ds.select(r=col("x").sum()).to_pydict() == {"r": [4.0]}


#: A group's ``(x, y)`` pairs, and what makes the least-squares fit degenerate there. Each
#: case is a shape the family used to answer differently from DuckDB — and, worse, from
#: itself: `regr_r2` said NULL for a flat ``y`` where the standard says the fit is perfect,
#: and `regr_intercept` said NaN for a flat ``x`` where `regr_r2` on the very same input
#: said NULL.
_DEGENERATE_FITS = [
    pytest.param([1.0, 1.0, 1.0], [2.0, 4.0, 7.0], id="x-flat"),
    pytest.param([1.0, 2.0, 3.0], [5.0, 5.0, 5.0], id="y-flat"),
    pytest.param([1.0, 1.0], [5.0, 5.0], id="both-flat"),
    pytest.param([1.0], [2.0], id="one-pair"),
    pytest.param([], [], id="no-rows"),
    pytest.param([None, 2.0], [4.0, 8.0], id="one-pair-after-null-drop"),
]


@pytest.mark.parametrize(("xs", "ys"), _DEGENERATE_FITS)
@pytest.mark.parametrize("fn", ["regr_slope", "regr_intercept", "regr_r2"])
def test_a_degenerate_fit_answers_what_duckdb_answers(duck, fn, xs, ys):
    """NaN and NULL are different answers, and which one is right is per-function.

    DuckDB gives `regr_slope` NaN and `regr_intercept` NULL on the same flat ``x`` — they
    are not two spellings of one rule, so neither can be derived from the other. Asserting
    the *pair* is the point: a change that unified them would pass a test on either alone.
    """
    tbl = pa.table({"x": pa.array(xs, pa.float64()), "y": pa.array(ys, pa.float64())})
    duck.register("t", tbl)
    sql = f"SELECT {fn}(y, x) AS v FROM t"
    expected = duck.execute(sql).fetch_arrow_table().to_pydict()["v"][0]
    actual = bt.sql(sql, t=bt.from_arrow(tbl)).to_pydict()["v"][0]
    if expected is None or actual is None:
        assert expected is actual, f"{fn}: NULL on one side only"
    elif math.isnan(expected) or math.isnan(actual):
        assert math.isnan(expected) and math.isnan(actual), f"{fn}: NaN on one side only"
    else:
        assert math.isclose(expected, actual, rel_tol=1e-12, abs_tol=1e-12)


def test_a_degenerate_fit_is_partition_independent():
    """The degenerate branches are `CASE`s over aggregates, so they must still merge.

    A conditional in the projection is evaluated *after* combine, on the merged state; a
    version that folded it into the partial would give a different answer per partition
    count, which is the failure mode invariant #7 exists to stop.
    """
    tbl = pa.table(
        {
            "g": ["flat-x", "flat-x", "flat-y", "flat-y", "fit", "fit", "fit"],
            "x": [1.0, 1.0, 1.0, 2.0, 1.0, 2.0, 3.0],
            "y": [2.0, 5.0, 4.0, 4.0, 2.0, 4.0, 6.0],
        }
    )

    def query(ds):
        return ds.group_by("g").agg(
            r2=bt.regr_r2(col("y"), col("x")),
            b=bt.regr_intercept(col("y"), col("x")),
        )

    single = bt.from_arrow(tbl)
    assert_tables_equal(
        query(single).collect(distributed=True, num_workers=3), query(single).collect()
    )
