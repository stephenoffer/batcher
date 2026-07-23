"""Differential tests for the derived statistical aggregates and epoch-unit extractors.

The statistics (`var_pop`, `stddev_pop`, `geometric_mean`, `harmonic_mean`, `rms`, `cv`,
`sem`, `midrange`) are expressions over Batcher's mergeable aggregates; DuckDB computes
each from the same closed-form, so the SQL is the oracle. The `.dt.epoch_ms/us/ns`
extractors are checked against DuckDB's `epoch_*`.
"""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same
from batcher import col

pytestmark = pytest.mark.differential


@pytest.fixture
def nums(duck):
    tbl = pa.table(
        {
            "g": ["a", "a", "a", "a", "b", "b", "b"],
            "x": [1.0, 2.0, 4.0, 8.0, 3.0, 5.0, 7.0],
        }
    )
    duck.register("t", tbl)
    return tbl


def test_derived_statistics_match_duckdb(duck, nums):
    out = (
        bt.from_arrow(nums)
        .group_by("g")
        .agg(
            vp=bt.var_pop("x"),
            sp=bt.stddev_pop("x"),
            gm=bt.geometric_mean("x"),
            hm=bt.harmonic_mean("x"),
            rms=bt.rms("x"),
            cv=bt.cv("x"),
            sem=bt.sem("x"),
            mid=bt.midrange("x"),
        )
    )
    expected = duck.sql(
        "SELECT g, "
        "var_pop(x) AS vp, "
        "stddev_pop(x) AS sp, "
        "exp(avg(ln(x))) AS gm, "
        "count(x)::double / sum(1.0 / x) AS hm, "
        "sqrt(avg(x * x)) AS rms, "
        "stddev_samp(x) / avg(x) AS cv, "
        "stddev_samp(x) / sqrt(count(x)) AS sem, "
        "(max(x) + min(x)) / 2 AS mid "
        "FROM t GROUP BY g"
    )
    assert_same(out.to_arrow(), expected)


@pytest.fixture
def singleton(duck):
    """A group of one, a group of two, and a group whose only value is null.

    `var_pop`/`stddev_pop` are *population* statistics, so a one-row group is defined
    and equal to 0 — where the *sample* `var`/`std` are null. The 4-and-3-row fixture
    above never exercises that, which is how `var_pop` shipped as
    ``var_samp * (n - 1) / n``: algebraically right, but null at ``n == 1`` because
    `var_samp` is, so ``NULL * 0`` swallowed the answer.
    """
    tbl = pa.table(
        {
            "g": ["one", "two", "two", "null"],
            "x": [5.0, 2.0, 6.0, None],
        }
    )
    duck.register("t", tbl)
    return tbl


def test_rms_of_large_integers_does_not_overflow(duck):
    """`rms` must widen to Float64 before squaring, not square in the input type.

    Squaring an Int64 column overflows silently: `4e9 * 4e9` wraps to a negative, the
    mean of those is negative, and the square root of a negative is `NaN`. The true RMS
    of a constant column is that constant. DuckDB raises `OutOfRangeException` on the
    un-widened square rather than returning a wrong number, so the oracle here is the
    explicitly widened `x::double * x::double`.
    """
    tbl = pa.table({"x": pa.array([4_000_000_000, 4_000_000_000, 3], type=pa.int64())})
    duck.register("t", tbl)
    out = bt.from_arrow(tbl).agg(r=bt.rms("x"))
    expected = duck.sql("SELECT sqrt(avg(x::double * x::double)) AS r FROM t")
    assert_same(out.to_arrow(), expected)


def test_population_stats_of_a_singleton_group_are_zero(duck, singleton):
    out = (
        bt.from_arrow(singleton)
        .group_by("g")
        .agg(vp=bt.var_pop("x"), sp=bt.stddev_pop("x"), n=col("x").count())
    )
    expected = duck.sql(
        "SELECT g, var_pop(x) AS vp, stddev_pop(x) AS sp, count(x) AS n FROM t GROUP BY g"
    )
    assert_same(out.to_arrow(), expected)


@pytest.fixture
def stamps(duck):
    tbl = pa.table(
        {
            "d": [
                dt.datetime(2021, 1, 1, 0, 0, 0),
                dt.datetime(1970, 1, 1, 0, 0, 1),
                dt.datetime(2024, 6, 15, 12, 30, 45),
                None,
            ]
        }
    )
    duck.register("t", tbl)
    return tbl


def test_epoch_units_match_duckdb(duck, stamps):
    out = bt.from_arrow(stamps).select(
        ms=col("d").dt.epoch_ms(),
        us=col("d").dt.epoch_us(),
        ns=col("d").dt.epoch_ns(),
    )
    expected = duck.sql("SELECT epoch_ms(d) AS ms, epoch_us(d) AS us, epoch_ns(d) AS ns FROM t")
    assert_same(out.to_arrow(), expected)


@pytest.fixture
def weighted(duck):
    tbl = pa.table(
        {
            "g": ["a", "a", "a", "b", "b"],
            "v": [1.0, 2.0, 3.0, 10.0, None],
            "w": [1.0, 2.0, 3.0, None, 5.0],
        }
    )
    duck.register("t", tbl)
    return tbl


def test_weighted_mean_matches_duckdb(duck, weighted):
    out = (
        bt.from_arrow(weighted)
        .group_by("g")
        .agg(
            wm=bt.weighted_mean(col("v"), col("w")),
        )
    )
    # DuckDB has no weighted_mean; the pairing (both non-null) is expressed explicitly.
    expected = duck.sql(
        "SELECT g, "
        "sum(CASE WHEN v IS NOT NULL AND w IS NOT NULL THEN v * w END) "
        "  / sum(CASE WHEN v IS NOT NULL AND w IS NOT NULL THEN w END) AS wm "
        "FROM t GROUP BY g"
    )
    assert_same(out.to_arrow(), expected)
