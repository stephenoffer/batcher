"""Differential tests vs DuckDB for `.str` edge-case semantics that hid wrong results.

Each test here pins a defect found by the string-function bug hunt: negative/empty
arguments and integer extremes that produced a wrong result or a process-aborting panic
(see docs/architecture/internals/bug_hunt_ledger.md). Kept separate from the happy-path `.str`
suites
so the edge contract is legible.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same
from batcher import col


@pytest.fixture
def t(duck):
    tbl = pa.table(
        {
            "s": pa.array(["", "a", "abc", "a-b-c", "a-b-c-d", "héllo", "中文字", "a😀b", None]),
        }
    )
    duck.register("t", tbl)
    return tbl


def test_right_negative_drops_leading_chars(duck, t):
    """`right(s, -2)` drops the first 2 chars (DuckDB), not returns ''."""
    out = (
        bt.from_arrow(t)
        .select(
            r2=col("s").str.right(-2),
            r0=col("s").str.right(0),
            r100=col("s").str.right(100),
        )
        .collect()
    )
    expected = duck.sql("SELECT right(s, -2) r2, right(s, 0) r0, right(s, 100) r100 FROM t")
    assert_same(out, expected)


def test_split_part_negative_and_empty_delimiter(duck, t):
    """Negative index counts from the right; empty delimiter splits into characters."""
    out = (
        bt.from_arrow(t)
        .select(
            neg1=col("s").str.split_part("-", -1),
            neg2=col("s").str.split_part("-", -2),
            empty2=col("s").str.split_part("", 2),
            empty1=col("s").str.split_part("", 1),
        )
        .collect()
    )
    expected = duck.sql(
        "SELECT split_part(s, '-', -1) neg1, split_part(s, '-', -2) neg2, "
        "split_part(s, '', 2) empty2, split_part(s, '', 1) empty1 FROM t"
    )
    assert_same(out, expected)


def test_split_empty_delimiter_yields_characters(duck, t):
    """`string_split(s, '')` splits into individual characters, no phantom '' ends."""
    out = bt.from_arrow(t).select(parts=col("s").str.split("")).collect()
    expected = duck.sql("SELECT string_split(s, '') parts FROM t")
    assert_same(out, expected)


def test_replace_empty_pattern_is_noop(duck, t):
    """`replace(s, '', 'X')` returns s unchanged (DuckDB), not X-between-every-char."""
    out = bt.from_arrow(t).select(r=col("s").str.replace("", "X")).collect()
    expected = duck.sql("SELECT replace(s, '', 'X') r FROM t")
    assert_same(out, expected)


def test_substring_index_and_overlay_extremes_do_not_crash():
    """i64-extreme counts/positions must clip, not overflow-panic the engine."""
    tbl = pa.table({"s": pa.array(["a-b-c", "hello"])})
    big, small = 9223372036854775807, -9223372036854775808
    out = (
        bt.from_arrow(tbl)
        .select(
            si_min=col("s").str.substring_index("-", small),
            si_max=col("s").str.substring_index("-", big),
            ov_max=col("s").str.overlay("XY", big),
            ov_len=col("s").str.overlay("XY", 2, big),
        )
        .collect()
        .to_pydict()
    )
    # The contract under test is "no panic / clean value"; pin the clipped results.
    assert out["si_min"] == ["a-b-c", "hello"]
    assert out["si_max"] == ["a-b-c", "hello"]


def test_substr_extremes_do_not_crash(duck):
    """substr with i64-extreme start/length clips to the string (DuckDB agrees)."""
    tbl = pa.table({"s": pa.array(["abcdef", "héllo", ""])})
    duck.register("t2", tbl)
    # DuckDB rejects i64-extreme substr args at bind time, so compare the large-but-legal
    # range against DuckDB, and separately assert the extreme case does not panic.
    out = (
        bt.from_arrow(tbl)
        .select(
            a=col("s").str.substr(2, 1000000),
            b=col("s").str.substr(-3, 2),
        )
        .collect()
    )
    expected = duck.sql("SELECT substring(s, 2, 1000000) a, substring(s, -3, 2) b FROM t2")
    assert_same(out, expected)
    # Extreme: must produce a value, not abort the process.
    got = (
        bt.from_arrow(tbl)
        .select(
            c=col("s").str.substr(9223372036854775807, 3),
            d=col("s").str.substr(3, -9223372036854775808),
        )
        .collect()
        .to_pydict()
    )
    assert got["c"] == ["", "", ""]


def test_oversized_repeat_and_pad_error_cleanly():
    """A multi-gigabyte repeat/pad returns a clean error, not an allocator abort."""
    tbl = pa.table({"s": pa.array(["abcdef"])})
    for expr in (
        col("s").str.repeat(1_000_000_000),
        col("s").str.lpad(9223372036854775807, "*"),
        col("s").str.rpad(9223372036854775807, "*"),
    ):
        with pytest.raises(Exception):  # noqa: B017
            bt.from_arrow(tbl).select(v=expr).collect()
    # A bounded size still works.
    ok = bt.from_arrow(tbl).select(v=col("s").str.repeat(3)).collect().to_pydict()
    assert ok["v"] == ["abcdefabcdefabcdef"]
