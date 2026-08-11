"""Semi/anti joins large enough to reach the swapped build side, checked against DuckDB.

`bc_runtime::join` builds a semi/anti join's hash table on the *left* (returned) side
instead of the right whenever the right is much the larger relation — the TPC-H q4 shape,
where the standard build-right order builds a table over 3.79M `lineitem` rows to answer a
membership question for 57k orders. The relation is meant to be identical either way.

**Every other semi/anti differential test in this suite is too small to reach it.** The swap
needs ~65k rows on the discarded side, and the existing join tests run on tens; they would all
pass with the swapped path completely broken. So these fix the sizes deliberately, and each
case states which internal path it is aimed at — an assertion here is only worth what the
input size buys.

The cases sweep what the swapped path handles differently from the oracle it replaces: NULL
keys on either side (a NULL is refused at the probe and never inserted at the build, so a
NULL-keyed left row is unmatched — `Semi` drops it, `Anti` keeps it), duplicate keys on either
side (a chain longer than one, versus the `unique`-build short-circuit), the empty and
no-overlap extremes, and each of the three key encodings the dispatch can pick.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

# Comfortably past the runtime's `SEMI_SWAP_MIN_PROBE_ROWS` (65,536) *and* past its 4x
# ratio against the returned side below, so the swapped path is what these exercise. Small
# enough that the file runs in a couple of seconds.
_PROBE_ROWS = 120_000
_RETURN_ROWS = 2_000


def _register(duck, left: pa.Table, right: pa.Table):
    duck.register("lft", left)
    duck.register("rgt", right)
    return bt.from_arrow(left), bt.from_arrow(right)


def _int_tables(*, left_nulls=False, right_nulls=False, overlap=True):
    """A small returned side against a large discarded one, on a single Int64 key."""
    lk: list[int | None] = list(range(_RETURN_ROWS))
    if left_nulls:
        lk[3] = None
        lk[17] = None
    # Half the returned side's keys occur on the discarded side and half do not, so `Semi`
    # and `Anti` each keep about half. A *full* overlap would make `Semi` return everything
    # and `Anti` nothing, and both of those agree with an implementation that ignores its
    # input — the row-count assertion below is what makes that a real check.
    base = list(range(_RETURN_ROWS // 2) if overlap else range(10**6, 10**6 + _RETURN_ROWS))
    # Cycles that key space, so each key repeats many times: the chain walk in the swapped
    # direction is over the *returned* side, and this makes the discarded side fan in.
    rk: list[int | None] = [base[i % len(base)] for i in range(_PROBE_ROWS)]
    if right_nulls:
        for i in range(0, _PROBE_ROWS, 1_000):
            rk[i] = None
    return (
        pa.table({"k": pa.array(lk, pa.int64()), "v": pa.array(range(_RETURN_ROWS), pa.int64())}),
        pa.table({"k": pa.array(rk, pa.int64())}),
    )


@pytest.mark.differential
@pytest.mark.parametrize("how", ["semi", "anti"])
@pytest.mark.parametrize(
    ("left_nulls", "right_nulls", "overlap", "label"),
    [
        (False, False, True, "plain"),
        (True, False, True, "null returned key"),
        (False, True, True, "null discarded key"),
        (True, True, True, "nulls on both sides"),
        (False, False, False, "no key in common"),
    ],
)
def test_int_key_semi_anti_matches_duckdb(duck, how, left_nulls, right_nulls, overlap, label):
    """The single-Int64 key path — the dispatch's `I64Keys` fast path."""
    left, right = _int_tables(left_nulls=left_nulls, right_nulls=right_nulls, overlap=overlap)
    lds, rds = _register(duck, left, right)
    got = lds.join(rds, left_on=["k"], right_on=["k"], how=how).collect()
    # `NOT IN` over a NULL-bearing set is SQL's three-valued trap and is *not* an anti-join,
    # so the anti case is expressed as `NOT EXISTS`, which is.
    exists = "EXISTS" if how == "semi" else "NOT EXISTS"
    expected = duck.sql(
        f"SELECT lft.k, lft.v FROM lft WHERE {exists} (SELECT 1 FROM rgt WHERE rgt.k = lft.k)"
    )
    assert_same(got, expected)
    # `overlap=False` shares no key at all, which is the one case where a whole-relation
    # answer is the *correct* one; everywhere else both outcomes must be non-empty, or the
    # case would pass against an implementation that returned a constant.
    if overlap:
        assert 0 < got.num_rows < _RETURN_ROWS, label


@pytest.mark.differential
@pytest.mark.parametrize("how", ["semi", "anti"])
def test_duplicate_returned_keys_are_judged_independently(duck, how):
    """Duplicates on the *returned* side: the swapped build holds a chain, not a unique key.

    This is the case the build-right order never has to think about, because there the
    returned side is the probe and each of its rows is judged on its own. Here they share a
    hash chain, and every copy must still be kept or dropped together with its key.
    """
    lk = [i % 50 for i in range(_RETURN_ROWS)]  # 50 distinct keys, 40 copies each
    rk = [i % 25 for i in range(_PROBE_ROWS)]  # only keys 0..24 occur
    left = pa.table({"k": pa.array(lk, pa.int64()), "v": pa.array(range(_RETURN_ROWS), pa.int64())})
    right = pa.table({"k": pa.array(rk, pa.int64())})
    lds, rds = _register(duck, left, right)
    got = lds.join(rds, left_on=["k"], right_on=["k"], how=how).collect()
    exists = "EXISTS" if how == "semi" else "NOT EXISTS"
    expected = duck.sql(
        f"SELECT lft.k, lft.v FROM lft WHERE {exists} (SELECT 1 FROM rgt WHERE rgt.k = lft.k)"
    )
    assert_same(got, expected)
    # Half the keys match, so neither side of the trichotomy is vacuous — a test that
    # returned everything (or nothing) would agree with a broken implementation too.
    assert 0 < got.num_rows < _RETURN_ROWS


@pytest.mark.differential
@pytest.mark.parametrize("how", ["semi", "anti"])
def test_string_key_semi_anti_matches_duckdb(duck, how):
    """A string key has no integer fast path, so this is the row-encoded `RowKeys` path."""
    lk = [f"k{i}" if i % 7 else None for i in range(_RETURN_ROWS)]
    rk = [f"k{i % (_RETURN_ROWS // 2)}" for i in range(_PROBE_ROWS)]
    left = pa.table(
        {"k": pa.array(lk, pa.string()), "v": pa.array(range(_RETURN_ROWS), pa.int64())}
    )
    right = pa.table({"k": pa.array(rk, pa.string())})
    lds, rds = _register(duck, left, right)
    got = lds.join(rds, left_on=["k"], right_on=["k"], how=how).collect()
    exists = "EXISTS" if how == "semi" else "NOT EXISTS"
    expected = duck.sql(
        f"SELECT lft.k, lft.v FROM lft WHERE {exists} (SELECT 1 FROM rgt WHERE rgt.k = lft.k)"
    )
    assert_same(got, expected)


@pytest.mark.differential
@pytest.mark.parametrize("how", ["semi", "anti"])
def test_composite_int_key_semi_anti_matches_duckdb(duck, how):
    """Two Int64 key columns — the dispatch's `I64x2Keys` path."""
    left = pa.table(
        {
            "a": pa.array([i % 100 for i in range(_RETURN_ROWS)], pa.int64()),
            "b": pa.array([i // 100 for i in range(_RETURN_ROWS)], pa.int64()),
            "v": pa.array(range(_RETURN_ROWS), pa.int64()),
        }
    )
    right = pa.table(
        {
            "a": pa.array([i % 100 for i in range(_PROBE_ROWS)], pa.int64()),
            "b": pa.array([(i // 100) % 10 for i in range(_PROBE_ROWS)], pa.int64()),
        }
    )
    lds, rds = _register(duck, left, right)
    got = lds.join(rds, left_on=["a", "b"], right_on=["a", "b"], how=how).collect()
    exists = "EXISTS" if how == "semi" else "NOT EXISTS"
    expected = duck.sql(
        f"SELECT lft.a, lft.b, lft.v FROM lft WHERE {exists} "
        "(SELECT 1 FROM rgt WHERE rgt.a = lft.a AND rgt.b = lft.b)"
    )
    assert_same(got, expected)


@pytest.mark.differential
@pytest.mark.parametrize("how", ["semi", "anti"])
def test_streamed_and_collected_agree_at_swap_scale(duck, how):
    """`iter_batches` must return the same relation as `collect` at this size.

    The two reach the join through different executors, and the swap lives below both — so
    this is what would catch a change that made only one of them take it.
    """
    left, right = _int_tables(left_nulls=True, right_nulls=True)
    lds, rds = _register(duck, left, right)
    joined = lds.join(rds, left_on=["k"], right_on=["k"], how=how)
    streamed = pa.Table.from_batches(list(joined.iter_batches()), schema=joined.collect().schema)
    exists = "EXISTS" if how == "semi" else "NOT EXISTS"
    expected = duck.sql(
        f"SELECT lft.k, lft.v FROM lft WHERE {exists} (SELECT 1 FROM rgt WHERE rgt.k = lft.k)"
    )
    assert_same(streamed, expected)
