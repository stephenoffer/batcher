"""SQL LIKE / ILIKE string matching vs DuckDB.

LIKE is an anchored match: ``%`` matches any (possibly empty) run of characters,
``_`` matches exactly one character, and every other character (including regex
metacharacters like ``.``) is literal. ILIKE is the case-insensitive form.
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
            "s": pa.array(
                [
                    "abc",
                    "axb",
                    "a.b",
                    "apple",
                    "banana",
                    "xx",
                    "Hxllo",
                    "HELLO",
                    "hello world",
                    "",
                    None,
                ]
            )
        }
    )
    duck.register("t", tbl)
    return tbl


def test_like_wildcards_vs_duckdb(duck, t):
    out = (
        bt.from_arrow(t)
        .select(
            a=col("s").str.like("a%"),
            b=col("s").str.like("%x_"),
            c=col("s").str.ilike("HELLO%"),
        )
        .collect()
    )
    expected = duck.sql("SELECT s LIKE 'a%' a, s LIKE '%x_' b, s ILIKE 'HELLO%' c FROM t")
    assert_same(out, expected)


def test_like_literal_metachars_vs_duckdb(duck, t):
    # The `.` in the pattern is a LITERAL dot, not a regex wildcard: it must match
    # "a.b" only, NOT "axb".
    out = (
        bt.from_arrow(t)
        .select(
            dot=col("s").str.like("a.b"),
            underscore=col("s").str.like("a_b"),
            anchored=col("s").str.like("abc"),
        )
        .collect()
    )
    expected = duck.sql(
        "SELECT s LIKE 'a.b' dot, s LIKE 'a_b' underscore, s LIKE 'abc' anchored FROM t"
    )
    assert_same(out, expected)


@pytest.mark.parametrize(
    "pat",
    ["%%c", "a%%", "%%%", "%%abc", "abc%%", "%%b%%"],
)
@pytest.mark.parametrize("op", ["LIKE", "ILIKE", "NOT LIKE"])
def test_sql_like_consecutive_percent_vs_duckdb(duck, t, op, pat):
    # The SQL fast path peeled a single boundary `%`, so a pattern with two or more
    # consecutive leading/trailing wildcards left an interior `%` that was matched
    # literally: `'abc' LIKE '%%c'` returned false where DuckDB returns true.
    q = f"SELECT (s {op} '{pat}') AS v FROM t"
    assert_same(bt.sql(q, t=t).collect(), duck.sql(q))


def test_like_empty_and_case_vs_duckdb(duck, t):
    out = (
        bt.from_arrow(t)
        .select(
            empty=col("s").str.like(""),
            anything=col("s").str.like("%"),
            contains_x=col("s").str.like("%x%"),
            ilike_h=col("s").str.ilike("h%"),
            like_h=col("s").str.like("h%"),
        )
        .collect()
    )
    expected = duck.sql(
        "SELECT s LIKE '' empty, s LIKE '%' anything, s LIKE '%x%' contains_x, "
        "s ILIKE 'h%' ilike_h, s LIKE 'h%' like_h FROM t"
    )
    assert_same(out, expected)


@pytest.mark.parametrize(
    "pat",
    [
        # Ordered multi-segment patterns: the fast matcher searches each literal segment
        # in order within the region the anchors leave free, instead of a full regex
        # automaton. The prefix/suffix + interior-substring shapes must all agree with
        # DuckDB (this is the TPC-H q13 `%special%requests%` family).
        "%a%b%",
        "a%b",
        "a%b%c",
        "%a%b",
        "a%b%",
        "%ll%o%",
        "h%o",
        "hello%world",
        "%xx%",
        "b%n%n%",
        "a%%b",
        "%a%.%",
    ],
)
@pytest.mark.parametrize("op", ["LIKE", "NOT LIKE"])
def test_sql_like_ordered_segments_vs_duckdb(duck, t, op, pat):
    q = f"SELECT (s {op} '{pat}') AS v FROM t"
    assert_same(bt.sql(q, t=t).collect(), duck.sql(q))


@pytest.mark.parametrize(
    "pred",
    [
        # `%` and `_` are "any character" in SQL, with no exception for a newline, and
        # every one of these shapes must agree with DuckDB on a string containing one.
        # Regression: SQL `LIKE` desugared to a Python-built regex that omitted `(?s)`,
        # so `.`/`.*` stopped at `\n` and `'a\nb' LIKE 'a%b'` was false here, true in
        # DuckDB. The escape-free shapes now lower to the native matcher; the ESCAPE
        # shape still desugars, and its regex carries `(?s)`.
        "s LIKE 'a%b'",
        "s LIKE 'a_b'",
        "s LIKE '%a%b%'",
        "s LIKE 'a%b%c'",
        "s NOT LIKE 'a%b'",
        "s ILIKE 'A%B'",
        r"s LIKE 'a\%b' ESCAPE '\'",
        r"s LIKE 'a%\_b' ESCAPE '\'",
    ],
)
def test_like_matches_newline_like_duckdb(duck, pred):
    """`%`/`_` span a newline — SQL says "any character", with no `\\n` exception."""
    tbl = pa.table({"s": ["a\nb", "axb", "a\nb\nc", "ab", "a%b", "a_b", "plain", "A\nB", None]})
    duck.register("nl", tbl)
    q = f"SELECT ({pred}) AS v FROM nl"
    assert_same(bt.sql(q, nl=tbl).collect(), duck.sql(q))
