"""Differential tests vs DuckDB for regex-replace backreferences, byte-based
`levenshtein`, and image-tensor allocation bounds.

Each test pins a defect found by the wave-2 string/media bug hunt (see
docs/architecture/internals/bug_hunt_ledger.md):

* `regexp_replace`/`regexp_replace_all` passed the rewrite template straight to the
  Rust `regex` crate (`$1` syntax), so DuckDB's RE2 backreferences (`\\1`) came out
  literal and a literal `$` was misinterpreted as a group.
* `levenshtein` counted Unicode scalar values; DuckDB (and PostgreSQL) count bytes.
* `image.to_tensor`/`resize` computed the per-row byte count `w * h * 3` in `usize`
  and cast it to a 32-bit Arrow element length — a product past `i32::MAX` wrapped
  negative and pre-allocated a multi-gigabyte buffer (an OOM bomb from a query param).
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
            "s": pa.array(["ab", "xaby", "abab", "aXbXc", "no-match", "", "héllo", None]),
        }
    )
    duck.register("t", tbl)
    return tbl


def test_regexp_replace_backreferences_match_duckdb(duck, t):
    """RE2 `\\1`/`\\2` backreferences and `\\0` (whole match) substitute like DuckDB."""
    out = (
        bt.from_arrow(t)
        .select(
            swap_first=col("s").str.regexp_replace(r"(a)(b)", r"\2\1"),
            swap_all=col("s").str.regexp_replace_all(r"(a)(b)", r"\2\1"),
            whole=col("s").str.regexp_replace_all(r"(ab)", r"[\0]"),
        )
        .collect()
    )
    expected = duck.sql(
        r"SELECT regexp_replace(s, '(a)(b)', '\2\1') swap_first, "
        r"regexp_replace(s, '(a)(b)', '\2\1', 'g') swap_all, "
        r"regexp_replace(s, '(ab)', '[\0]', 'g') whole FROM t"
    )
    assert_same(out, expected)


def test_regexp_replace_dollar_is_literal(duck, t):
    """A `$` in the replacement is literal in DuckDB (RE2 uses `\\1` for a group, not `$1`);
    an out-of-range group reference (an invalid rewrite) leaves the row unchanged in both.

    (DuckDB's handling of an invalid *escape char* like ``\\q`` is internally inconsistent —
    it drops-and-truncates the template — so that specific edge is a deliberate divergence:
    Batcher treats any invalid rewrite as a no-op. Only the well-defined cases are pinned.)
    """
    out = (
        bt.from_arrow(t)
        .select(
            dollar=col("s").str.regexp_replace_all("a", "$1"),
            bad_group=col("s").str.regexp_replace_all("a", r"\1"),
        )
        .collect()
    )
    expected = duck.sql(
        r"SELECT regexp_replace(s, 'a', '$1', 'g') dollar, "
        r"regexp_replace(s, 'a', '\1', 'g') bad_group FROM t"
    )
    assert_same(out, expected)


def test_levenshtein_is_byte_based(duck):
    """`levenshtein` counts UTF-8 bytes, matching DuckDB (`'héllo'`↔`'abc'` = 6, not 5)."""
    tbl = pa.table({"s": pa.array(["héllo", "café", "日本", "abc", "", "naïve", None])})
    duck.register("lev", tbl)
    out = (
        bt.from_arrow(tbl)
        .select(
            to_abc=col("s").str.levenshtein("abc"),
            to_e=col("s").str.levenshtein("cafe"),
        )
        .collect()
    )
    expected = duck.sql("SELECT levenshtein(s, 'abc') to_abc, levenshtein(s, 'cafe') to_e FROM lev")
    assert_same(out, expected)


def test_to_tensor_rejects_oversized_product():
    """`to_tensor(w, h)` whose `w*h*3` exceeds i32::MAX errors cleanly (no wrap / OOM)."""
    tbl = pa.table({"img": pa.array([b"not an image"], pa.binary())})
    with pytest.raises(Exception):  # noqa: B017 - clean engine error, not a crash
        bt.from_arrow(tbl).select(t=col("img").image.to_tensor(40_000, 40_000)).collect()
    # A bounded request still succeeds (undecodable bytes → null row, not an error).
    out = bt.from_arrow(tbl).select(t=col("img").image.to_tensor(8, 8)).collect()
    assert out.num_rows == 1


def test_jaro_and_jaro_winkler_match_duckdb(duck):
    """`.str.jaro_similarity` / `.str.jaro_winkler_similarity` vs DuckDB's own functions."""
    tbl = pa.table(
        {"s": pa.array(["MARTHA", "DWAYNE", "DIXON", "martha", "abc", "", "jones", None])}
    )
    duck.register("jw", tbl)
    out = (
        bt.from_arrow(tbl)
        .select(
            j=col("s").str.jaro_similarity("MARHTA"),
            w=col("s").str.jaro_winkler_similarity("MARHTA"),
        )
        .collect()
    )
    expected = duck.sql(
        "SELECT jaro_similarity(s, 'MARHTA') j, jaro_winkler_similarity(s, 'MARHTA') w FROM jw"
    )
    assert_same(out, expected)


def test_damerau_levenshtein_matches_duckdb(duck):
    """`.str.damerau_levenshtein` vs DuckDB — including transposition cases (`teh`↔`the`)
    and multi-edit strings where true DL differs from the OSA variant (`ca`→`abc`)."""
    tbl = pa.table({"s": pa.array(["teh", "the", "ca", "kitten", "acb", "abc", "café", "", None])})
    duck.register("dl", tbl)
    out = (
        bt.from_arrow(tbl)
        .select(
            to_the=col("s").str.damerau_levenshtein("the"),
            to_abc=col("s").str.damerau_levenshtein("abc"),
        )
        .collect()
    )
    expected = duck.sql(
        "SELECT damerau_levenshtein(s, 'the') to_the, damerau_levenshtein(s, 'abc') to_abc FROM dl"
    )
    assert_same(out, expected)
