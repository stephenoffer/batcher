"""Float-identity follow-up sweep vs DuckDB — the wave-22 paths that still ranked raw bits.

Closing ledger B26 aligned scalar `=`/`<` and `ORDER BY` to the engine's float identity
(`-0.0`==`0.0`; every NaN is one value, ranked greatest — DuckDB's own semantics). But a
handful of paths still fed a **raw-bit** total order into a `RowConverter` or `total_cmp`,
so on the same column they disagreed with `GROUP BY`/`=`/`MAX`: a *negative* NaN (what
`0.0/0.0` yields on x86) ranked below `-inf`, and `-0.0` split from `0.0`. Each is fixed to
route through the one canonical identity (`bc_arrow::float_ident` / `bc_runtime::keys`).

These use `duck_materialize` (not `duck.register`): a registered Arrow table is scanned with
predicates pushed into the Arrow scan under IEEE semantics, which contradicts DuckDB's own
executor on NaN. See `conftest.duck_materialize`.
"""

from __future__ import annotations

import struct

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same, duck_materialize

pytestmark = pytest.mark.differential


def _f64(bits: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", bits))[0]


NEG_NAN = _f64(0xFFF8000000000000)  # sign bit set — raw total order ranks it below -inf
POS_NAN = _f64(0x7FF8000000000000)


def test_greatest_least_rank_nan_like_duckdb(duck):
    """`greatest`/`least` over columns holding a negative NaN and signed zeros. DuckDB:
    NaN is the greatest value, so `greatest` returns it and `least` never does."""
    a = [1.0, NEG_NAN, -0.0, 5.0]
    b = [NEG_NAN, 2.0, 0.0, 3.0]
    table = pa.table({"a": pa.array(a, pa.float64()), "b": pa.array(b, pa.float64())})
    duck_materialize(duck, "t", table)
    out = (
        bt.from_arrow(table)
        .select(
            bt.greatest(bt.col("a"), bt.col("b")).alias("g"),
            bt.least(bt.col("a"), bt.col("b")).alias("l"),
        )
        .collect()
    )
    assert_same(out, duck.sql("SELECT greatest(a, b) AS g, least(a, b) AS l FROM t"))


def test_median_over_negative_nan_matches_duckdb(duck):
    """`median` must rank a negative NaN greatest (not below -inf), so the selected rank is
    unshifted: median([1,2,3,-NaN]) = 2.5 in DuckDB, not 1.5."""
    v = [1.0, 2.0, 3.0, NEG_NAN]
    table = pa.table({"g": [0, 0, 0, 0], "v": pa.array(v, pa.float64())})
    duck_materialize(duck, "m", table)
    out = bt.from_arrow(table).group_by("g").agg(md=bt.col("v").median()).collect()
    assert_same(out, duck.sql("SELECT g, median(v) AS md FROM m GROUP BY g"))


def test_arg_max_with_negative_nan_key_matches_duckdb(duck):
    """`arg_max(v, key)` on a negative-NaN key: NaN ranks greatest, so that row wins."""
    table = pa.table(
        {
            "g": [0, 0, 0],
            "v": pa.array([10.0, 20.0, 30.0], pa.float64()),
            "k": pa.array([1.0, NEG_NAN, 2.0], pa.float64()),
        }
    )
    duck_materialize(duck, "k", table)
    out = bt.from_arrow(table).group_by("g").agg(am=bt.col("v").arg_max(bt.col("k"))).collect()
    assert_same(out, duck.sql("SELECT g, arg_max(v, k) AS am FROM k GROUP BY g"))


def test_signed_zero_is_one_group_in_mode_and_distinct(duck):
    """`-0.0` and `0.0` are one value: `mode` folds them and `count(distinct)` counts one."""
    table = pa.table({"g": [0, 0, 0], "v": pa.array([-0.0, -0.0, 0.0], pa.float64())})
    duck_materialize(duck, "z", table)
    out = bt.from_arrow(table).group_by("g").agg(nu=bt.col("v").n_unique()).collect()
    assert_same(out, duck.sql("SELECT g, count(DISTINCT v) AS nu FROM z GROUP BY g"))
