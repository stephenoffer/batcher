"""Differential tests for ``ORDER BY <string> LIMIT k`` against DuckDB.

A one-key string top-N reaches the engine through a path of its own: each morsel is
fully sorted by ``str_sort::stable_sort_indices_str`` and sliced to ``k``, then the
survivors are merged. That is a third route, distinct from the integer radix and from
the general quickselect a multi-key top-N takes, and it is the only one whose ordering
is a hand-written comparator over a packed 8-byte prefix rather than an arrow kernel.
Nothing pinned it end to end, which is what these tests add.

The cases target where a prefix-keyed comparator can go wrong: heavy ties, where the
row-position tie-break alone decides which rows survive; values that share their first
eight bytes or carry a literal NUL, which the packed prefix cannot separate; nulls; and
``k`` at the boundaries of a morsel and of the input.

The payload column ``p`` is the input row position and Batcher's top-N is stable, so
DuckDB's ``ORDER BY s, p`` is a *total* order that Batcher's output must match
row-for-row. That is why these use ``assert_same_ordered``: a multiset comparison would
pass while the wrong tied row survived, which is exactly the defect class here.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same_ordered

# Over the row count that engages the parallel morsel-wise top-N, so each morsel is
# reduced independently and the survivors merged — the path under test.
PARALLEL_ROWS = 200_000


def _table(n: int, distinct: int, *, nulls: bool = False) -> pa.Table:
    """Heavily tied keys (``n // distinct`` duplicates each) with a row-id payload."""
    keys = [
        None if (nulls and i % 97 == 0) else f"str_{(i * 7919) % distinct:05d}" for i in range(n)
    ]
    return pa.table(
        {"s": pa.array(keys, type=pa.string()), "p": pa.array(range(n), type=pa.int64())}
    )


@pytest.mark.differential
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("k", [1, 10, 5_000, 200_001])
def test_string_topn_matches_duckdb(duck, descending, k):
    """`k` spans one row, well inside a morsel, across many morsels, and past the input."""
    t = _table(PARALLEL_ROWS, distinct=5_000)
    duck.register("t", t)
    direction = "DESC" if descending else "ASC"
    out = bt.from_arrow(t).sort("s", descending=descending).limit(k).collect()
    assert_same_ordered(out, duck.sql(f"SELECT * FROM t ORDER BY s {direction}, p ASC LIMIT {k}"))


@pytest.mark.differential
@pytest.mark.parametrize("descending", [False, True])
def test_string_topn_with_nulls_matches_duckdb(duck, descending):
    """Nulls group last in both directions (Batcher's `nulls_first` default is False)."""
    t = _table(PARALLEL_ROWS, distinct=3_000, nulls=True)
    duck.register("t", t)
    direction = "DESC" if descending else "ASC"
    out = bt.from_arrow(t).sort("s", descending=descending).limit(100).collect()
    assert_same_ordered(
        out,
        duck.sql(f"SELECT * FROM t ORDER BY s {direction} NULLS LAST, p ASC LIMIT 100"),
    )


@pytest.mark.differential
def test_string_topn_reaching_into_the_null_run(duck):
    """A `k` larger than the non-null count must fall through into the nulls, in input order.

    Nulls are grouped by `nulls_first` alone and are interchangeable within that group, so
    the first `k` of them in input order are the ones the stable sort keeps. A `k` that
    stops *inside* that run is where an off-by-one would show.
    """
    n = 300
    keys = [f"k{i:03d}" if i < 100 else None for i in range(n)]
    t = pa.table({"s": pa.array(keys, type=pa.string()), "p": pa.array(range(n), type=pa.int64())})
    duck.register("t", t)
    for k in (99, 100, 101, 150, 300):
        out = bt.from_arrow(t).sort("s").limit(k).collect()
        assert_same_ordered(
            out, duck.sql(f"SELECT * FROM t ORDER BY s ASC NULLS LAST, p ASC LIMIT {k}")
        )


@pytest.mark.differential
def test_string_topn_all_keys_equal_keeps_input_order(duck):
    """Every key tied: the surviving rows are decided *entirely* by the row-position
    tie-break, so this fails outright if that tie-break is ever lost."""
    t = _table(PARALLEL_ROWS, distinct=1)
    duck.register("t", t)
    out = bt.from_arrow(t).sort("s").limit(50).collect()
    assert out.column("p").to_pylist() == list(range(50))
    assert_same_ordered(out, duck.sql("SELECT * FROM t ORDER BY s ASC, p ASC LIMIT 50"))


@pytest.mark.differential
@pytest.mark.parametrize("descending", [False, True])
def test_string_topn_pack_collisions_match_duckdb(duck, descending):
    """Values the packed 8-byte prefix key cannot separate.

    The comparator packs a zero-padded 8-byte prefix and reads the bytes only when the
    packs tie. These values make them tie: strings agreeing for eight or more bytes, a
    string that is a prefix of another, and a literal NUL where another value ends — the
    case padding genuinely cannot resolve.
    """
    values = [
        "abcdefghZZZ",
        "abcdefghAAA",
        "abcdefgh",
        "abcdefgh\0",
        "abc",
        "abc\0",
        "",
        "\0",
        "abcdefghi",
        "zzzzzzzzzzzz",
        "ab",
    ]
    n = 40_000
    keys = [values[i % len(values)] for i in range(n)]
    t = pa.table({"s": pa.array(keys, type=pa.string()), "p": pa.array(range(n), type=pa.int64())})
    duck.register("t", t)
    direction = "DESC" if descending else "ASC"
    for k in (1, 7, 1_000):
        out = bt.from_arrow(t).sort("s", descending=descending).limit(k).collect()
        assert_same_ordered(
            out, duck.sql(f"SELECT * FROM t ORDER BY s {direction}, p ASC LIMIT {k}")
        )


@pytest.mark.differential
def test_string_topn_all_null_column(duck):
    """No key is ever compared; the answer is the first `k` rows in input order."""
    n = 5_000
    t = pa.table(
        {
            "s": pa.array([None] * n, type=pa.string()),
            "p": pa.array(range(n), type=pa.int64()),
        }
    )
    duck.register("t", t)
    out = bt.from_arrow(t).sort("s").limit(20).collect()
    assert_same_ordered(out, duck.sql("SELECT * FROM t ORDER BY s ASC NULLS LAST, p ASC LIMIT 20"))


@pytest.mark.differential
def test_string_topn_agrees_with_its_own_two_key_form(duck):
    """The one-key and two-key spellings of the same order must agree.

    They take different engine paths — the one-key form sorts each morsel through the
    string comparator, the two-key form selects through the general quickselect — so this
    is the one assertion here that does not need DuckDB to catch a divergence between
    them. They may differ in cost; they must never differ in result.
    """
    t = _table(PARALLEL_ROWS, distinct=200)
    duck.register("t", t)
    one = bt.from_arrow(t).sort("s").limit(500).collect()
    two = bt.from_arrow(t).sort("s", "p").limit(500).collect()
    assert one.column("p").to_pylist() == two.column("p").to_pylist()
    assert_same_ordered(one, duck.sql("SELECT * FROM t ORDER BY s ASC, p ASC LIMIT 500"))
