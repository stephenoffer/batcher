"""Differential coverage for the bool_and / bool_or aggregates vs DuckDB."""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col

pytestmark = pytest.mark.differential


def _t() -> pa.Table:
    return pa.table(
        {
            "g": ["a", "a", "b", "b", "c", "a", "b"],
            # group a: T,T,T → and=T; b: F,T,(null) → and=F,or=T; c: F → and/or=F
            "flag": [True, True, False, True, False, True, None],
        }
    )


def test_bool_and_or_grouped_matches_duckdb(duck):
    from conftest import assert_same

    t = _t()
    duck.register("t", t)
    out = (
        bt.from_arrow(t)
        .group_by("g")
        .agg(ba=col("flag").bool_and(), bo=col("flag").bool_or())
        .collect()
    )
    assert_same(out, duck.sql("SELECT g, bool_and(flag) ba, bool_or(flag) bo FROM t GROUP BY g"))


def test_bool_and_or_global_matches_duckdb(duck):
    from conftest import assert_same

    t = _t()
    duck.register("t", t)
    out = bt.from_arrow(t).agg(ba=col("flag").bool_and(), bo=col("flag").bool_or()).collect()
    assert_same(out, duck.sql("SELECT bool_and(flag) ba, bool_or(flag) bo FROM t"))


def test_mode_grouped_matches_duckdb(duck):
    from conftest import assert_same

    # Unique mode per group (no frequency ties) so the tiebreak rule is irrelevant
    # and the result matches DuckDB's mode() exactly.
    t = pa.table(
        {
            "g": ["a", "a", "a", "b", "b", "b", "b", "c"],
            "v": [5, 5, 7, 9, 9, 9, 1, 4],  # a→5, b→9, c→4
        }
    )
    duck.register("t", t)
    out = bt.from_arrow(t).group_by("g").agg(m=col("v").mode()).collect()
    assert_same(out, duck.sql("SELECT g, mode(v) m FROM t GROUP BY g"))


def test_arg_min_max_grouped_matches_duckdb(duck):
    from conftest import assert_same

    # Unique keys per group → arg_min/arg_max are unambiguous and match DuckDB's
    # arg_min/arg_max(value, key).
    t = pa.table(
        {
            "g": ["a", "a", "a", "b", "b", "c"],
            "val": [10, 20, 30, 40, 50, 60],
            "key": [1, 3, 2, 5, 4, 7],
        }
    )
    duck.register("t", t)
    out = (
        bt.from_arrow(t)
        .group_by("g")
        .agg(hi=col("val").arg_max(by=col("key")), lo=col("val").arg_min(by=col("key")))
        .collect()
    )
    assert_same(
        out,
        duck.sql("SELECT g, arg_max(val, key) hi, arg_min(val, key) lo FROM t GROUP BY g"),
    )
