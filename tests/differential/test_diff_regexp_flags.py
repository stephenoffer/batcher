"""SQL `regexp_matches` option-string (flags) parity vs DuckDB.

Pins the defect where the `_sql` translator silently *dropped* the third
argument of `regexp_matches(s, pattern, options)`, so a case-insensitive
match (`'i'`) or dot-matches-newline (`'s'`) query returned the plain
case-sensitive / newline-sensitive result — a silent wrong answer.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same


@pytest.fixture
def t(duck):
    tbl = pa.table(
        {
            "s": pa.array(
                ["ABC", "abc", "AbC", "a\nc", "A\nC", "xyz", "", None],
                pa.string(),
            )
        }
    )
    duck.register("t", tbl)
    return {"t": tbl}


@pytest.mark.differential
@pytest.mark.parametrize(
    "q",
    [
        # 'i' — case-insensitive: without the fix batcher matched only 'abc'.
        "SELECT s, regexp_matches(s, 'abc', 'i') AS v FROM t",
        # 's' — '.' matches newline: without the fix 'a\nc' did not match.
        "SELECT s, regexp_matches(s, 'a.c', 's') AS v FROM t",
        # combined options.
        "SELECT s, regexp_matches(s, 'a.c', 'is') AS v FROM t",
        # 'c' — explicit case-sensitive (default) is a no-op.
        "SELECT s, regexp_matches(s, 'abc', 'c') AS v FROM t",
        # no options — unchanged behaviour.
        "SELECT s, regexp_matches(s, 'abc') AS v FROM t",
    ],
)
def test_regexp_matches_options(duck, t, q):
    assert_same(bt.sql(q, **t).collect(), duck.sql(q))


@pytest.mark.differential
def test_unsupported_option_raises_not_silent(t):
    # An option we cannot map bit-identically must raise, never silently drop.
    with pytest.raises(NotImplementedError):
        bt.sql("SELECT regexp_matches(s, 'a.c', 'm') AS v FROM t", **t).collect()


@pytest.mark.differential
@pytest.mark.parametrize(
    "q",
    [
        # 'i' — case-insensitive: without the fix the options arg was dropped, so
        # `regexp_replace(s, 'abc', 'X', 'i')` matched case-sensitively (wrong).
        "SELECT s, regexp_replace(s, 'abc', 'X', 'i') AS v FROM rt",
        # 's' — '.' matches newline.
        "SELECT s, regexp_replace(s, 'a.c', 'X', 'is') AS v FROM rt",
        # 'g' — global (replace every match), not just the first.
        "SELECT s, regexp_replace(s, 'a', 'X', 'g') AS v FROM rt",
        # 'ig' — case-insensitive AND global together.
        "SELECT s, regexp_replace(s, 'abc', 'X', 'ig') AS v FROM rt",
        # no options — DuckDB default is first-match-only.
        "SELECT s, regexp_replace(s, 'a', 'X') AS v FROM rt",
    ],
)
def test_regexp_replace_options(duck, q):
    tbl = pa.table({"s": pa.array(["ABC", "abc", "abcabc", "aa", "a\nc", ""], pa.string())})
    duck.register("rt", tbl)
    assert_same(bt.sql(q, rt=bt.from_arrow(tbl)).collect(), duck.sql(q))


@pytest.mark.differential
def test_regexp_replace_unsupported_option_raises(duck):
    tbl = pa.table({"s": ["abc"]})
    duck.register("rt", tbl)
    q = "SELECT regexp_replace(s, 'a', 'X', 'm') AS v FROM rt"
    with pytest.raises(NotImplementedError):
        bt.sql(q, rt=bt.from_arrow(tbl)).collect()
