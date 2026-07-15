"""`.list` / `.map` edge-case parity with DuckDB.

Covers the corners a naive per-row reducer gets wrong: type promotion in
`contains`/`position`, NULL/NaN ordering in `sort`/`median`/`min`/`max`, `-0.0`
folding in `unique`/`n_unique`/`intersect`, the empty-list `join`, an out-of-range
`get`, and a narrow-int map key. Each was a real defect; the oracle is DuckDB.
"""

from __future__ import annotations

import math

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col

pytestmark = pytest.mark.differential


def _same_list(got: list, exp: list) -> None:
    """Element-wise compare two lists-of-lists, treating NaN == NaN and None == None."""
    assert len(got) == len(exp), f"{got} vs {exp}"
    for g, e in zip(got, exp, strict=True):
        if g is None or e is None:
            assert g is e, f"{got} vs {exp}"
            continue
        assert len(g) == len(e), f"{got} vs {exp}"
        for x, y in zip(g, e, strict=True):
            if isinstance(x, float) and math.isnan(x):
                assert isinstance(y, float) and math.isnan(y), f"{got} vs {exp}"
            else:
                assert x == y, f"{got} vs {exp}"


def test_contains_does_not_narrow_the_child(duck):
    # [2.5].contains(2) must be False — casting the float child down to the int literal
    # truncated 2.5→2 and wrongly reported True.
    t = pa.table({"a": [[2.5], [2.0], []]})
    duck.register("t", t)
    got = bt.from_arrow(t).select(r=col("a").list.contains(2)).to_pydict()["r"]
    exp = duck.sql("SELECT list_contains(a, 2) r FROM t").to_arrow_table().to_pydict()["r"]
    assert got == exp


def test_position_does_not_narrow_the_child(duck):
    t = pa.table({"a": [[2.5], [2.0]]})
    duck.register("t", t)
    got = bt.from_arrow(t).select(r=col("a").list.position(2)).to_pydict()["r"]
    exp = duck.sql("SELECT list_position(a, 2) r FROM t").to_arrow_table().to_pydict()["r"]
    assert got == exp


def test_sort_nulls_last_and_nan_greatest(duck):
    t = pa.table({"a": [[3.0, float("nan"), 1.0, None]]})
    duck.register("t", t)
    got = bt.from_arrow(t).select(r=col("a").list.sort()).to_pydict()["r"]
    exp = duck.sql("SELECT list_sort(a) r FROM t").to_arrow_table().to_pydict()["r"]
    _same_list(got, exp)


def test_median_and_max_order_nan_greatest(duck):
    t = pa.table({"a": [[1.0, float("nan"), 2.0]]})
    duck.register("t", t)
    got_m = bt.from_arrow(t).select(r=col("a").list.median()).to_pydict()["r"]
    exp_m = duck.sql("SELECT list_median(a) r FROM t").to_arrow_table().to_pydict()["r"]
    assert got_m == exp_m  # 2.0
    got_mx = bt.from_arrow(t).select(r=col("a").list.max()).to_pydict()["r"]
    assert math.isnan(got_mx[0])
    exp_mx = duck.sql("SELECT list_max(a) r FROM t").to_arrow_table().to_pydict()["r"]
    assert math.isnan(exp_mx[0])


def test_unique_is_type_general_over_strings(duck):
    # A string list used to be cast to Float64 → every element null → empty result.
    t = pa.table({"a": [["a", "b", "a"]]})
    duck.register("t", t)
    got = bt.from_arrow(t).select(r=col("a").list.unique()).to_pydict()["r"]
    n = bt.from_arrow(t).select(r=col("a").list.n_unique()).to_pydict()["r"]
    assert sorted(got[0]) == ["a", "b"]
    assert n == duck.sql("SELECT list_unique(a) r FROM t").to_arrow_table().to_pydict()["r"]


def test_unique_folds_signed_zero(duck):
    t = pa.table({"a": [[0.0, -0.0]]})
    duck.register("t", t)
    n = bt.from_arrow(t).select(r=col("a").list.n_unique()).to_pydict()["r"]
    exp = duck.sql("SELECT list_unique(a) r FROM t").to_arrow_table().to_pydict()["r"]
    assert n == exp  # 1 — -0.0 and 0.0 are one value


def test_join_empty_list_is_empty_string(duck):
    # DuckDB array_to_string([]) == '' but an all-null list stays null.
    t = pa.table({"a": [[], ["a", None], None]})
    duck.register("t", t)
    got = bt.from_arrow(t).select(r=col("a").list.join("-")).to_pydict()["r"]
    exp = duck.sql("SELECT array_to_string(a, '-') r FROM t").to_arrow_table().to_pydict()["r"]
    assert got == exp


def test_get_out_of_range_is_null():
    # A huge index must not panic; it lands out of range → null.
    t = pa.table({"a": [[10, 20, 30]]})
    got = bt.from_arrow(t).select(r=col("a").list.get(2**40)).to_pydict()["r"]
    assert got == [None]


def test_intersect_folds_signed_zero(duck):
    t = pa.table({"a": [[0.0]], "b": [[-0.0]]})
    duck.register("t", t)
    got = bt.from_arrow(t).select(r=col("a").list.intersect(col("b"))).to_pydict()["r"]
    exp = duck.sql("SELECT list_intersect(a, b) r FROM t").to_arrow_table().to_pydict()["r"]
    _same_list(got, exp)  # [[0.0]], not [[]]


def test_map_get_matches_narrow_int_key():
    # Int32 map keys are not normalized to Int64 inside a nested Map; get(2) must still hit.
    col_m = pa.array([[(1, 10), (2, 20)]], type=pa.map_(pa.int32(), pa.int64()))
    ds = bt.from_arrow(pa.table({"m": col_m}))
    assert ds.select(r=col("m").map.get(2)).to_pydict()["r"] == [20]
    assert ds.select(r=col("m").map.get(9)).to_pydict()["r"] == [None]
