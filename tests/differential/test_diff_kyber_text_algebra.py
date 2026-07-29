"""The `text_algebra` rewrites must match DuckDB after the full optimizer runs.

Each case is the *original* spelling: Batcher optimizes it into the absorbed, direct, or
emptiness form and the oracle evaluates it as written. The fixture carries the rows the
rewrites turn on — an empty string, a NULL, a value that satisfies only the weaker of two
patterns, and one that satisfies neither.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
import batcher.kyber.rules.text_algebra
from _harness import assert_same
from batcher import col, lit
from batcher.plan.expr_ir.func_nodes import StrFunc


@pytest.fixture
def t(duck):
    tbl = pa.table(
        {
            "s": pa.array(["a/b/c", "a/x", "", None, "zz", "abc"], type=pa.string()),
            "f": pa.array(["r.tar.gz", "r.gz", "", None, "plain", "abc"], type=pa.string()),
        }
    )
    duck.register("t", tbl)
    return tbl


@pytest.fixture
def empty(duck):
    tbl = pa.table({"s": pa.array([], type=pa.string()), "f": pa.array([], type=pa.string())})
    duck.register("t", tbl)
    return tbl


_CASES = [
    (
        "prefix_conjunction",
        lambda: col("s").str.starts_with("a/") & col("s").str.starts_with("a/b/"),
        "starts_with(s, 'a/') AND starts_with(s, 'a/b/')",
    ),
    (
        "prefix_disjunction",
        lambda: col("s").str.starts_with("a/") | col("s").str.starts_with("a/b/"),
        "starts_with(s, 'a/') OR starts_with(s, 'a/b/')",
    ),
    (
        "suffix_conjunction",
        lambda: col("f").str.ends_with(".gz") & col("f").str.ends_with(".tar.gz"),
        "ends_with(f, '.gz') AND ends_with(f, '.tar.gz')",
    ),
    (
        "suffix_disjunction",
        lambda: col("f").str.ends_with(".gz") | col("f").str.ends_with(".tar.gz"),
        "ends_with(f, '.gz') OR ends_with(f, '.tar.gz')",
    ),
    (
        "substring_conjunction",
        lambda: col("s").str.contains("b") & col("s").str.contains("a/b"),
        "contains(s, 'b') AND contains(s, 'a/b')",
    ),
    (
        "substring_disjunction",
        lambda: col("s").str.contains("b") | col("s").str.contains("a/b"),
        "contains(s, 'b') OR contains(s, 'a/b')",
    ),
    (
        "equality_conjunction",
        lambda: (col("s") == lit("a/b/c")) & col("s").str.starts_with("a/"),
        "s = 'a/b/c' AND starts_with(s, 'a/')",
    ),
    (
        "equality_disjunction",
        lambda: (col("s") == lit("a/b/c")) | col("s").str.starts_with("a/"),
        "s = 'a/b/c' OR starts_with(s, 'a/')",
    ),
    (
        "position_positive",
        lambda: StrFunc("position", col("s"), pattern="b") > lit(0),
        "strpos(s, 'b') > 0",
    ),
    (
        "position_zero",
        lambda: StrFunc("position", col("s"), pattern="b") == lit(0),
        "strpos(s, 'b') = 0",
    ),
    (
        "position_ge_one",
        lambda: StrFunc("position", col("s"), pattern="/") >= lit(1),
        "strpos(s, '/') >= 1",
    ),
    (
        "regexp_count_positive",
        lambda: StrFunc("regexp_count", col("s"), pattern="a+") > lit(0),
        "regexp_matches(s, 'a+')",
    ),
    (
        "leading_slice",
        lambda: StrFunc("substr", col("s"), start=1, length=2) == lit("a/"),
        "substr(s, 1, 2) = 'a/'",
    ),
    (
        "leading_slice_negated",
        lambda: StrFunc("substr", col("s"), start=1, length=2) != lit("a/"),
        "substr(s, 1, 2) <> 'a/'",
    ),
    (
        "trailing_slice",
        lambda: StrFunc("right", col("s"), start=2) == lit("/c"),
        "right(s, 2) = '/c'",
    ),
    (
        "reverse_equality",
        lambda: StrFunc("reverse", col("s")) == lit("cba"),
        "reverse(s) = 'cba'",
    ),
    (
        "reverse_inequality",
        lambda: StrFunc("reverse", col("s")) != lit("cba"),
        "reverse(s) <> 'cba'",
    ),
    ("length_positive", lambda: col("s").str.len() > lit(0), "length(s) > 0"),
    ("length_le_zero", lambda: col("s").str.len() <= lit(0), "length(s) <= 0"),
    ("length_ne_zero", lambda: col("s").str.len() != lit(0), "length(s) <> 0"),
    (
        "octet_length_zero",
        lambda: StrFunc("octet_length", col("s")) == lit(0),
        "strlen(s) = 0",
    ),
    (
        "octet_length_positive",
        lambda: StrFunc("octet_length", col("s")) > lit(0),
        "strlen(s) > 0",
    ),
    (
        "bit_length_positive",
        lambda: StrFunc("bit_length", col("s")) >= lit(1),
        "bit_length(s) >= 1",
    ),
    (
        "double_lpad",
        lambda: StrFunc(
            "lpad", StrFunc("lpad", col("s"), start=8, pattern="0"), start=8, pattern="0"
        ),
        "lpad(lpad(s, 8, '0'), 8, '0')",
    ),
    (
        "double_rpad",
        lambda: StrFunc(
            "rpad", StrFunc("rpad", col("s"), start=8, pattern="."), start=8, pattern="."
        ),
        "rpad(rpad(s, 8, '.'), 8, '.')",
    ),
]


@pytest.mark.parametrize(
    ("expr", "sql"), [(e, s) for _, e, s in _CASES], ids=[n for n, _, _ in _CASES]
)
def test_text_rewrite_matches_duckdb(duck, t, expr, sql):
    out = bt.from_arrow(t).select(r=expr()).collect()
    assert_same(out, duck.sql(f"SELECT {sql} AS r FROM t"))


@pytest.mark.parametrize(
    ("expr", "sql"), [(e, s) for _, e, s in _CASES], ids=[n for n, _, _ in _CASES]
)
def test_text_rewrite_matches_duckdb_on_empty_input(duck, empty, expr, sql):
    out = bt.from_arrow(empty).select(r=expr()).collect()
    assert_same(out, duck.sql(f"SELECT {sql} AS r FROM t"))


def test_absorption_inside_a_filter_matches_duckdb(duck, t):
    out = (
        bt.from_arrow(t)
        .filter(col("s").str.starts_with("a/") & col("s").str.starts_with("a/b/"))
        .select(s=col("s"))
        .collect()
    )
    assert_same(
        out,
        duck.sql("SELECT s FROM t WHERE starts_with(s, 'a/') AND starts_with(s, 'a/b/')"),
    )


def test_unrelated_patterns_are_left_alone_and_still_match(duck, t):
    out = (
        bt.from_arrow(t)
        .select(r=col("s").str.starts_with("a/") & col("s").str.starts_with("zz"))
        .collect()
    )
    assert_same(out, duck.sql("SELECT starts_with(s, 'a/') AND starts_with(s, 'zz') AS r FROM t"))
