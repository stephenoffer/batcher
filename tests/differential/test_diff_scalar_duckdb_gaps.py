"""Scalar functions DuckDB has and the engine did not — vs DuckDB.

The differential function census (`docs/architecture/internals/competitor_parity_census.md`) sorted
DuckDB's 478 scalar and aggregate builtins into supported, absent, and *wrong*. These are
the absent ones this wave implemented: five math functions, one two-argument math
function, and eleven string functions whose definition comes from an external
specification (RFC 3986, POSIX paths, RE2's `QuoteMeta`) rather than from an operation on
characters.

Arguments are **columns**, not literals, so constant folding cannot answer the query
without the runtime kernel running.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same
from batcher import col


@pytest.fixture
def nums(duck):
    tbl = pa.table(
        {
            "f": pa.array([2.1, -2.1, 2.0, 3.0, 0.5, -0.5, 1.5, None]),
            "g": pa.array([5.0, 1.0, 0.5, 4.0, 2.0, 3.0, 6.0, None]),
            "one": pa.array([1.0, 1.0, -1.0, 0.0, 1e16, 1e-300, -0.0, None]),
            "two": pa.array([2.0, 0.0, -2.0, 1.0, 1e17, 0.0, 1.0, None]),
        }
    )
    duck.register("nums", tbl)
    return tbl


@pytest.fixture
def texts(duck):
    tbl = pa.table(
        {
            "s": pa.array(
                [
                    "a b/c",
                    "a%20b",
                    "1+1",
                    "a.b*c[d]",
                    "plain",
                    "",
                    "ünïcodé",
                    None,
                ]
            ),
            "p": pa.array(
                [
                    "/data/2024/events.parquet",
                    "raw/in.csv",
                    "noslash",
                    "/single",
                    "a/b",
                    "",
                    "/x/y/z.csv",
                    None,
                ]
            ),
            "code": pa.array(["abc", "abd", "xyz", "aab", "cba", "abc", "zzz", None]),
        }
    )
    duck.register("texts", tbl)
    return tbl


@pytest.mark.differential
@pytest.mark.parametrize("method", ["even", "gamma", "lgamma"])
def test_new_math_matches_duckdb(duck, nums, method):
    out = bt.from_arrow(nums).select(r=getattr(col("g"), method)()).collect()
    assert_same(out, duck.sql(f"SELECT {method}(g) r FROM nums"))


@pytest.mark.differential
def test_even_rounds_outward_not_to_nearest(duck, nums):
    """`even` is the one whose direction is easy to get wrong: `3.0` becomes `4.0`."""
    out = bt.from_arrow(nums).select(r=col("f").even()).collect()
    assert_same(out, duck.sql("SELECT even(f) r FROM nums"))
    assert bt.from_arrow(nums).select(r=col("f").even()).to_pydict()["r"][:4] == [
        4.0,
        -4.0,
        2.0,
        4.0,
    ]


@pytest.mark.differential
def test_next_after_matches_duckdb(duck, nums):
    out = bt.from_arrow(nums).select(r=bt.next_after(col("one"), col("two"))).collect()
    assert_same(out, duck.sql("SELECT nextafter(one, two) r FROM nums"))


@pytest.mark.differential
@pytest.mark.parametrize(
    "method",
    ["url_encode", "url_decode", "regexp_escape", "to_binary"],
)
def test_new_string_functions_match_duckdb(duck, texts, method):
    out = bt.from_arrow(texts).select(r=getattr(col("s").str, method)()).collect()
    assert_same(out, duck.sql(f"SELECT {method}(s) r FROM texts"))


@pytest.mark.differential
@pytest.mark.parametrize(
    "method", ["parse_filename", "parse_dirname", "parse_dirpath", "parse_path"]
)
def test_path_functions_match_duckdb(duck, texts, method):
    out = bt.from_arrow(texts).select(r=getattr(col("p").str, method)()).collect()
    assert_same(out, duck.sql(f"SELECT {method}(p) r FROM texts"))


@pytest.mark.differential
def test_hamming_and_jaccard_match_duckdb(duck, texts):
    out = (
        bt.from_arrow(texts)
        .select(h=col("code").str.hamming("abc"), j=col("code").str.jaccard("abc"))
        .collect()
    )
    assert_same(out, duck.sql("SELECT hamming(code, 'abc') h, jaccard(code, 'abc') j FROM texts"))


@pytest.mark.differential
def test_hamming_rejects_unequal_lengths(texts):
    """DuckDB raises rather than comparing a prefix; so does the engine."""
    with pytest.raises(Exception, match="equal length"):
        bt.from_arrow(texts).select(r=col("code").str.hamming("ab")).collect()


@pytest.mark.differential
def test_binary_text_round_trips(texts):
    """`from_binary` inverts `to_binary`; undecodable input is null, not an error."""
    out = bt.from_arrow(texts).select(r=col("s").str.to_binary().str.from_binary()).to_pydict()
    # The empty string round-trips to null: `to_binary('')` is `''`, which is not a whole
    # number of bytes, so `from_binary` nulls it. DuckDB does the same.
    assert out["r"] == [*texts.column("s").to_pylist()[:5], None, "ünïcodé", None]
    bad = bt.from_pydict({"b": ["0110000", "0110000x", "01100001"]})
    assert bad.select(r=col("b").str.from_binary()).to_pydict()["r"] == [None, None, "a"]


@pytest.mark.differential
def test_regexp_escape_output_is_usable_as_a_pattern(texts):
    """The escaped value must match itself — the property the function exists for.

    RE2's rule escapes more than the engine's own matcher needs (a space, a `/`), so this
    also pins that the matcher accepts the wider escaping rather than rejecting it.
    """
    escaped = bt.from_arrow(texts).select(e=col("s").str.regexp_escape()).to_pydict()["e"]
    for value, pattern in zip(texts.column("s").to_pylist(), escaped, strict=True):
        if value is None:
            continue
        one = bt.from_pydict({"s": [value]})
        assert one.select(m=col("s").str.regexp_matches(pattern)).to_pydict()["m"] == [True]


@pytest.mark.differential
@pytest.mark.parametrize(
    ("query", "column"),
    [
        ("SELECT even(g) r FROM nums", "g"),
        ("SELECT gamma(g) r FROM nums", "g"),
        ("SELECT lgamma(g) r FROM nums", "g"),
        ("SELECT nextafter(one, two) r FROM nums", "one"),
        ("SELECT url_encode(s) r FROM texts", "s"),
        ("SELECT url_decode(s) r FROM texts", "s"),
        ("SELECT regexp_escape(s) r FROM texts", "s"),
        ("SELECT to_binary(s) r FROM texts", "s"),
        ("SELECT parse_filename(p) r FROM texts", "p"),
        ("SELECT parse_dirname(p) r FROM texts", "p"),
        ("SELECT parse_dirpath(p) r FROM texts", "p"),
        ("SELECT parse_path(p) r FROM texts", "p"),
        ("SELECT hamming(code, 'abc') r FROM texts", "code"),
        ("SELECT mismatches(code, 'abc') r FROM texts", "code"),
        ("SELECT jaccard(code, 'abc') r FROM texts", "code"),
        ("SELECT prefix(s, 'a') r FROM texts", "s"),
        ("SELECT suffix(s, 'b') r FROM texts", "s"),
    ],
)
def test_reachable_from_sql(duck, nums, texts, query, column):
    """Each is also callable under its DuckDB name from `bt.sql`."""
    assert_same(bt.sql(query, nums=nums, texts=texts).collect(), duck.sql(query))


@pytest.mark.differential
def test_pi_and_today_are_constants(duck, nums):
    """`pi()`/`today()` are nullary in DuckDB and fold to a literal here."""
    assert bt.sql("SELECT pi() r").to_pydict()["r"] == [pytest.approx(3.141592653589793)]
    got = bt.sql("SELECT today() r").to_pydict()["r"][0]
    assert got == duck.sql("SELECT today() r").fetchall()[0][0]
