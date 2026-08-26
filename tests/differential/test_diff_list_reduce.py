"""Per-row list reductions (product/std/var) and `unique`, against DuckDB.

This file was marked "structural" and asserted only hand-written values. DuckDB has
`list_product`, `list_var_samp`, `list_stddev_samp` and `list_distinct`, so the oracle the
contract asks for was available and absent.

Checked: **product, var and std agree exactly**, on the sample-variance convention
(`n-1`) and on every degenerate row — a single-element list is null rather than 0, an empty
list is null, a null list is null.

**`unique` agrees as a set and differs in element order, deliberately on both sides.**
Batcher's `list.unique()` keeps first-occurrence order (`[3,1,2]` stays `[3,1,2]`), which is
what `test_list_unique_dedups_first_occurrence` pins; DuckDB's `list_distinct` makes no such
promise and returns `[2,1,3]` here. So that one case is compared as a multiset, with
Batcher's ordering asserted separately — comparing it positionally would fail on a
difference neither engine considers a defect, and comparing it only as a set would drop the
ordering guarantee this API actually makes.

Measured against duckdb 1.5.5.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same
from batcher import col

pytestmark = pytest.mark.differential

pytest.importorskip("duckdb")

#: An unsorted list, a singleton (too short for a sample variance), an empty list, and a
#: null. The last three are where a reduction most often conflates "no value" with zero.
_INTS = pa.table({"a": pa.array([[3, 1, 2], [4], [], None], type=pa.list_(pa.int64()))})


def _ints():
    return bt.from_arrow(_INTS)


def _duck(duck, sql: str):
    duck.register("t", _INTS)
    return duck.sql(sql)


def test_list_product(duck):
    got = _ints().select(p=col("a").list.product()).collect()
    assert got.to_pydict()["p"] == [6.0, 4.0, None, None]
    assert_same(got, _duck(duck, "SELECT list_product(a) AS p FROM t"))


def test_list_var(duck):
    # [3,1,2]: mean=2 → var=((3-2)²+(1-2)²+(2-2)²)/(3-1)=(1+1+0)/2=1.0
    # [4]: n<2 → null; []: empty → null; None → null.
    got = _ints().select(v=col("a").list.var()).collect()
    assert got.to_pydict()["v"] == [1.0, None, None, None]
    # `list_var_samp`, not `list_var_pop`: naming the wrong one would compare a different
    # statistic and still produce a plausible-looking pass on some inputs.
    assert_same(got, _duck(duck, "SELECT list_var_samp(a) AS v FROM t"))


def test_list_std(duck):
    # std = sqrt(var); for [3,1,2] var=1.0 → std=1.0.
    got = _ints().select(s=col("a").list.std()).collect()
    assert got.to_pydict()["s"] == [1.0, None, None, None]
    assert_same(got, _duck(duck, "SELECT list_stddev_samp(a) AS s FROM t"))


def test_list_unique(duck):
    out = _ints().select(u=col("a").list.unique()).collect().to_pydict()
    assert out["u"] == [[3, 1, 2], [4], [], None]
    # Compared as a multiset per row, because `list_distinct` promises no element order and
    # answers `[2, 1, 3]` for the first row. What both engines must agree on is *which*
    # elements survive; Batcher's ordering is a stronger guarantee, asserted above and in
    # `test_list_unique_dedups_first_occurrence`.
    expected = _duck(duck, "SELECT list_distinct(a) AS u FROM t").to_arrow_table().to_pydict()
    for mine, theirs in zip(out["u"], expected["u"], strict=True):
        assert (mine is None) == (theirs is None)
        if mine is not None:
            assert sorted(mine) == sorted(theirs)


def test_list_unique_dedups_first_occurrence():
    ds = bt.from_arrow(pa.table({"a": pa.array([[1, 1, 2, 2, 3]], type=pa.list_(pa.int64()))}))
    out = ds.select(u=col("a").list.unique()).collect().to_pydict()
    assert out["u"] == [[1, 2, 3]]


def test_list_var_known_spread():
    # [2,4,4,4,5,5,7,9]: mean=5, Σ(x-mean)²=32, sample var=32/7.
    ds = bt.from_arrow(
        pa.table({"a": pa.array([[2, 4, 4, 4, 5, 5, 7, 9]], type=pa.list_(pa.int64()))})
    )
    out = ds.select(v=col("a").list.var(), s=col("a").list.std()).collect().to_pydict()
    assert out["v"][0] == 32.0 / 7.0
    assert out["s"][0] == (32.0 / 7.0) ** 0.5
