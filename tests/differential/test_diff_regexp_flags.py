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
from conftest import assert_same


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
