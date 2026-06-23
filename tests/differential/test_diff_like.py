"""SQL LIKE / ILIKE string matching vs DuckDB.

LIKE is an anchored match: ``%`` matches any (possibly empty) run of characters,
``_`` matches exactly one character, and every other character (including regex
metacharacters like ``.``) is literal. ILIKE is the case-insensitive form.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
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
    from conftest import assert_same

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
    from conftest import assert_same

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


def test_like_empty_and_case_vs_duckdb(duck, t):
    from conftest import assert_same

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
