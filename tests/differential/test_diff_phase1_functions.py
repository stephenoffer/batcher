"""Differential coverage for the Phase 1 Python ergonomics & function library.

The new operators (`//`, `^`, `<<`, `>>`, `[]`, unary `-`/`abs`/`round`/`floor`/
`ceil`/`trunc`) and free functions (`concat`/`concat_ws`/`format_string`/`iff`/
`nanvl`/`ifnull`/`log`) all desugar to existing IR; here we prove they match DuckDB
on real data, including the null/negative edges where the desugaring is subtle.
"""

from __future__ import annotations

import math

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col, concat, concat_ws, format_string, iff, ifnull, lit, log, nanvl

pytestmark = pytest.mark.differential

_NAN = float("nan")


def _nums():
    return pa.table(
        {
            "a": pa.array([7, -7, 4, -8, None], type=pa.int64()),
            "b": pa.array([2, 2, 3, 3, 2], type=pa.int64()),
        }
    )


def test_floordiv_matches_duckdb(duck):
    from conftest import assert_same

    duck.register("t", _nums())
    out = bt.from_arrow(_nums()).select(fd=col("a") // col("b")).collect()
    # floor(a/b) with true division — floors toward -inf (Python/Polars `//`).
    assert_same(out, duck.sql("SELECT floor(a::DOUBLE / b) AS fd FROM t"))


def test_bitwise_operators_match_duckdb(duck):
    from conftest import assert_same

    duck.register("t", _nums())
    out = (
        bt.from_arrow(_nums())
        .select(
            x=col("a") ^ col("b"),
            ls=col("b") << lit(2),
            rs=col("a") >> lit(1),
        )
        .collect()
    )
    assert_same(out, duck.sql("SELECT xor(a, b) AS x, b << 2 AS ls, a >> 1 AS rs FROM t"))


def test_unary_and_round_builtins_match_duckdb(duck):
    from conftest import assert_same

    data = pa.table({"x": pa.array([3.4, -3.6, 2.5, None], type=pa.float64())})
    duck.register("t", data)
    out = (
        bt.from_arrow(data)
        .select(
            neg=-col("x"),
            ab=abs(col("x")),
            fl=math.floor(col("x")),
            ce=math.ceil(col("x")),
            tr=math.trunc(col("x")),
            rn=round(col("x")),
        )
        .collect()
    )
    assert_same(
        out,
        duck.sql(
            "SELECT -x AS neg, abs(x) AS ab, floor(x) AS fl, ceil(x) AS ce, "
            "trunc(x) AS tr, round(x) AS rn FROM t"
        ),
    )


def test_list_struct_indexing_matches_duckdb(duck):
    from conftest import assert_same

    # list element (Python 0-based → DuckDB 1-based) and slice.
    out = (
        bt.from_arrow(pa.table({"id": [1, 2]}))
        .select(
            first=bt.array(lit(10), lit(20), lit(30))[0],
            last=bt.array(lit(10), lit(20), lit(30))[-1],
            mid=bt.array(lit(10), lit(20), lit(30))[1:3],
        )
        .collect()
    )
    expected = duck.sql(
        "SELECT [10,20,30][1] AS first, [10,20,30][3] AS last, [10,20,30][2:3] AS mid "
        "FROM (VALUES (1),(2)) t(id)"
    )
    assert_same(out, expected)


def test_concat_treats_null_as_empty(duck):
    from conftest import assert_same

    data = pa.table({"a": ["x", None, "z"], "b": ["1", "2", None]})
    duck.register("t", data)
    out = bt.from_arrow(data).select(c=concat(col("a"), col("b"))).collect()
    assert_same(out, duck.sql("SELECT concat(a, b) AS c FROM t"))


def test_concat_ws_skips_nulls(duck):
    from conftest import assert_same

    data = pa.table({"a": ["x", None, "z"], "b": ["1", "2", None]})
    duck.register("t", data)
    out = bt.from_arrow(data).select(c=concat_ws("-", col("a"), col("b"))).collect()
    assert_same(out, duck.sql("SELECT concat_ws('-', a, b) AS c FROM t"))


def test_format_string_matches_concat(duck):
    from conftest import assert_same

    data = pa.table({"k": ["a", "b"], "v": [1, 2]})
    duck.register("t", data)
    out = bt.from_arrow(data).select(f=format_string("{}={}", col("k"), col("v"))).collect()
    assert_same(out, duck.sql("SELECT concat(k, '=', v::VARCHAR) AS f FROM t"))


def test_iff_matches_duckdb(duck):
    from conftest import assert_same

    duck.register("t", _nums())
    out = bt.from_arrow(_nums()).select(s=iff(col("a") > 0, lit("pos"), lit("neg"))).collect()
    assert_same(out, duck.sql("SELECT CASE WHEN a > 0 THEN 'pos' ELSE 'neg' END AS s FROM t"))


def test_nanvl_matches_duckdb(duck):
    from conftest import assert_same

    data = pa.table({"x": pa.array([1.0, _NAN, None, -2.5], type=pa.float64())})
    duck.register("t", data)
    out = bt.from_arrow(data).select(r=nanvl(col("x"), lit(0.0))).collect()
    assert_same(out, duck.sql("SELECT CASE WHEN isnan(x) THEN 0.0 ELSE x END AS r FROM t"))


def test_ifnull_matches_duckdb(duck):
    from conftest import assert_same

    duck.register("t", _nums())
    out = bt.from_arrow(_nums()).select(r=ifnull(col("a"), lit(0))).collect()
    assert_same(out, duck.sql("SELECT ifnull(a, 0) AS r FROM t"))


def test_log_base_matches_duckdb(duck):
    from conftest import assert_same

    data = pa.table({"x": pa.array([2.0, 8.0, 100.0], type=pa.float64())})
    duck.register("t", data)
    out = bt.from_arrow(data).select(l2=log(2, col("x")), l10=log(10, col("x"))).collect()
    assert_same(out, duck.sql("SELECT ln(x)/ln(2) AS l2, ln(x)/ln(10) AS l10 FROM t"))
