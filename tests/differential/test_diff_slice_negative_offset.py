"""A negative `offset` into `str.slice` / `list.slice` counts back from the end.

Both methods are the Polars spelling of a 0-based sub-range, and both got the negative
case wrong in a way that returned data rather than an error -- the worst shape a slicing
bug can take.

`str.slice` lowers to the 1-based `substr`, and shifted *every* offset by one to get
there. That is right for a non-negative offset and wrong for a negative one, which is
already end-relative and denotes the same position in both spellings: `substr` resolves
it as `n + offset + 1`, which is the 0-based `n + offset` the method promises. So each
negative slice landed one character nearer the end -- `slice(-3, 2)` on "abcdef" gave
"ef" instead of "de" -- and `slice(-1)` ran off the end and wrapped back to the whole
string.

`list.slice` did not implement the negative case at all: it clamped the offset with
`.max(0)`, so every negative offset silently returned the list's *head*. `slice(-1)`
returned the whole list. That path is not obscure -- `.list.tail` is not a method, and
the accessor's own `AttributeError` guidance directs users to `.list.slice(-n, n)`.

The oracle is DuckDB's `substr` / `list_slice`, which agree with Polars and with Python's
own slicing on every case here.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col

pytestmark = pytest.mark.differential

# Strings chosen to cover an empty string, a multi-byte grapheme (so a character-oriented
# slice cannot be confused with a byte-oriented one), and leading/trailing whitespace.
_STRINGS = ["abcdef", "", "aXbXc", "  pad  ", "\U0001f600z", "ABC", None]

# Offsets that land inside, exactly on, and past both ends of every string above.
_OFFSETS = (-10, -6, -3, -1, 0, 1, 3, 10)
_LENGTHS = (None, 0, 2, 100)


def _expected_str(s: str | None, offset: int, length: int | None) -> str | None:
    """Python's own slicing, which is what the 0-based contract means."""
    if s is None:
        return None
    start = len(s) + offset if offset < 0 else offset
    stop = len(s) if length is None else start + max(length, 0)
    return s[max(start, 0) : max(min(stop, len(s)), 0)]


@pytest.mark.parametrize("offset", _OFFSETS)
@pytest.mark.parametrize("length", _LENGTHS)
def test_str_slice_matches_python_slicing(offset, length):
    t = pa.table({"s": pa.array(_STRINGS, pa.string())})
    expr = col("s").str.slice(offset) if length is None else col("s").str.slice(offset, length)
    got = bt.from_arrow(t).select(r=expr).collect().to_pydict()["r"]
    assert got == [_expected_str(s, offset, length) for s in _STRINGS]


@pytest.mark.parametrize("offset", _OFFSETS)
@pytest.mark.parametrize("length", _LENGTHS)
def test_str_slice_matches_duckdb_substr(duck, offset, length):
    """The same matrix against DuckDB, whose `substr` is the 1-based spelling."""
    t = pa.table({"s": pa.array(_STRINGS, pa.string())})
    expr = col("s").str.slice(offset) if length is None else col("s").str.slice(offset, length)
    got = bt.from_arrow(t).select(r=expr).collect().to_pydict()["r"]

    # 0-based `offset` -> 1-based `start`: shift a non-negative offset by one; a negative
    # one already denotes the same position in both spellings.
    start = offset + 1 if offset >= 0 else offset
    duck.register("t", t)
    sql = (
        f"SELECT substr(s, {start}) AS r FROM t"
        if length is None
        else f"SELECT substr(s, {start}, {length}) AS r FROM t"
    )
    want = [row[0] for row in duck.sql(sql).fetchall()]
    assert got == want


_LISTS = [[10, 20, 30, 40, 50, 60], [], [1], None]


def _expected_list(xs: list[int] | None, offset: int, length: int | None) -> list[int] | None:
    if xs is None:
        return None
    start = len(xs) + offset if offset < 0 else offset
    stop = len(xs) if length is None else start + max(length, 0)
    return xs[max(start, 0) : max(min(stop, len(xs)), 0)]


@pytest.mark.parametrize("offset", _OFFSETS)
@pytest.mark.parametrize("length", _LENGTHS)
def test_list_slice_matches_python_slicing(offset, length):
    t = pa.table({"xs": pa.array(_LISTS, pa.list_(pa.int64()))})
    expr = col("xs").list.slice(offset) if length is None else col("xs").list.slice(offset, length)
    got = bt.from_arrow(t).select(r=expr).collect().to_pydict()["r"]
    assert got == [_expected_list(xs, offset, length) for xs in _LISTS]


@pytest.mark.parametrize("offset", (-6, -3, -1))
def test_list_slice_to_end_matches_duckdb(duck, offset):
    """`slice(-n)` runs to the end, which is DuckDB's `list_slice(xs, -n, len)`."""
    t = pa.table({"xs": pa.array([[10, 20, 30, 40, 50, 60], [1]], pa.list_(pa.int64()))})
    got = bt.from_arrow(t).select(r=col("xs").list.slice(offset)).collect().to_pydict()["r"]
    duck.register("t", t)
    want = [
        row[0]
        for row in duck.sql(f"SELECT list_slice(xs, {offset}, len(xs)) AS r FROM t").fetchall()
    ]
    assert got == want


def test_list_slice_negative_tail_is_the_documented_tail_idiom():
    """`.list` has no `tail`; its guidance names `slice(-n, n)`, which must be the tail."""
    t = pa.table({"xs": pa.array([[10, 20, 30, 40], [1], []], pa.list_(pa.int64()))})
    got = bt.from_arrow(t).select(r=col("xs").list.slice(-2, 2)).collect().to_pydict()["r"]
    assert got == [[30, 40], [1], []]


def test_str_slice_negative_offset_past_front_is_empty_not_head():
    """An offset reaching past the front collapses, the way `"abc"[-10:-8]` does."""
    t = pa.table({"s": pa.array(["abcdef"], pa.string())})
    got = bt.from_arrow(t).select(r=col("s").str.slice(-10, 2)).collect().to_pydict()["r"]
    assert got == [""]


def test_list_slice_negative_offset_past_front_is_empty_not_head():
    t = pa.table({"xs": pa.array([[10, 20, 30, 40, 50, 60]], pa.list_(pa.int64()))})
    got = bt.from_arrow(t).select(r=col("xs").list.slice(-10, 2)).collect().to_pydict()["r"]
    assert got == [[]]
