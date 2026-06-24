"""Coverage for the Phase 2 math additions (gcd/lcm/factorial/bit_count/hypot/
width_bucket).

`gcd`/`lcm`/`factorial`/`bit_count` are checked against DuckDB; `hypot` against
Python's `math.hypot`; `width_bucket` (absent from this DuckDB build) against the
SQL-standard bucket convention.
"""

from __future__ import annotations

import math

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col, gcd, hypot, lcm, lit, width_bucket

pytestmark = pytest.mark.differential


def _ints():
    return pa.table({"a": pa.array([48, 17, 6, 1], type=pa.int64()), "b": [36, 5, 3, 7]})


def test_gcd_lcm_match_duckdb(duck):
    from conftest import assert_same

    duck.register("t", _ints())
    out = (
        bt.from_arrow(_ints())
        .select(g=gcd(col("a"), col("b")), l=lcm(col("a"), col("b")))
        .collect()
    )
    assert_same(out, duck.sql("SELECT gcd(a, b) AS g, lcm(a, b) AS l FROM t"))


def test_factorial_bit_count_match_duckdb(duck):
    from conftest import assert_same

    # Small values so factorial stays in DuckDB's HUGEINT range.
    data = pa.table({"a": pa.array([6, 5, 3, 0, 10], type=pa.int64())})
    duck.register("t", data)
    out = bt.from_arrow(data).select(f=col("a").factorial(), bc=col("a").bit_count()).collect()
    # DuckDB factorial takes INTEGER (a is BIGINT); cast for the oracle.
    assert_same(out, duck.sql("SELECT factorial(a::INTEGER) AS f, bit_count(a) AS bc FROM t"))


def test_hypot_matches_python():
    out = bt.from_arrow(_ints()).select(h=hypot(col("a"), col("b"))).collect().to_pydict()["h"]
    expected = [math.hypot(a, b) for a, b in zip([48, 17, 6, 1], [36, 5, 3, 7], strict=True)]
    assert [round(x, 9) for x in out] == [round(x, 9) for x in expected]


def test_width_bucket_standard_convention():
    # SQL width_bucket: count equal-width buckets over [low, high]; below → 0,
    # at/above high → count+1.
    data = pa.table({"a": pa.array([6, 15, 49, 55, -5], type=pa.int64())})
    out = (
        bt.from_arrow(data)
        .select(b=width_bucket(col("a"), lit(0), lit(50), 5))
        .collect()
        .to_pydict()["b"]
    )
    assert [int(x) for x in out] == [1, 2, 5, 6, 0]
