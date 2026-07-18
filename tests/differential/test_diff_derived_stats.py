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
from batcher import col
from conftest import assert_same

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
