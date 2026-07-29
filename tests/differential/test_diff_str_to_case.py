"""`str.to_case` — anchored to DuckDB where DuckDB can express the style, else pinned.

DuckDB has no identifier-recasing function, so no style has a direct oracle. Several do
have one on an input whose word boundaries are unambiguous: on a space-separated
lowercase phrase, ``snake``/``kebab``/``dot`` are DuckDB's ``replace(s, ' ', sep)``, and
on an already-snake identifier ``upper_snake`` is ``upper(s)`` and ``snake`` is
``lower(s)``. Those anchor the joining behaviour to the oracle; the remaining styles are
pinned against explicit expectations, and the word *splitting* every style shares is
unit-tested in ``crates/bc-expr/src/eval/str/case.rs``.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same
from batcher import col
from batcher._internal.errors import PlanError

STYLES = [
    "snake",
    "upper_snake",
    "camel",
    "pascal",
    "kebab",
    "upper_kebab",
    "title",
    "sentence",
    "dot",
    "train",
]


@pytest.mark.parametrize(("style", "sep"), [("snake", "_"), ("kebab", "-"), ("dot", ".")])
def test_separator_styles_of_a_plain_phrase_match_duckdb_replace(duck, style, sep):
    t = pa.table({"s": ["hello world", "the quick brown fox", "a", "", None]})
    duck.register("s", t)
    out = bt.from_arrow(t).select(r=col("s").str.to_case(style)).collect()
    assert_same(out, duck.sql(f"SELECT replace(s, ' ', '{sep}') r FROM s"))


@pytest.mark.parametrize(("style", "duck_fn"), [("upper_snake", "upper"), ("snake", "lower")])
def test_snake_identifier_recasing_matches_duckdb_case_folding(duck, style, duck_fn):
    t = pa.table({"s": ["user_id", "a_b_c", "single", "", None]})
    duck.register("s", t)
    out = bt.from_arrow(t).select(r=col("s").str.to_case(style)).collect()
    assert_same(out, duck.sql(f"SELECT {duck_fn}(s) r FROM s"))


@pytest.mark.parametrize(
    ("style", "want"),
    [
        ("snake", "parse_http_response"),
        ("upper_snake", "PARSE_HTTP_RESPONSE"),
        ("camel", "parseHttpResponse"),
        ("pascal", "ParseHttpResponse"),
        ("kebab", "parse-http-response"),
        ("upper_kebab", "PARSE-HTTP-RESPONSE"),
        ("title", "Parse Http Response"),
        ("sentence", "Parse http response"),
        ("dot", "parse.http.response"),
        ("train", "Parse-Http-Response"),
    ],
)
def test_every_style_on_an_acronym_bearing_identifier(style, want):
    t = pa.table({"s": ["parseHTTPResponse"]})
    got = bt.from_arrow(t).select(r=col("s").str.to_case(style)).to_pydict()["r"]
    assert got == [want]


@pytest.mark.parametrize("style", STYLES)
def test_edge_inputs_never_raise_and_keep_nulls(style):
    # Empty, separator-only, single character, digits, and null — the shapes that break a
    # word splitter. A null stays null; anything with no alphanumerics becomes "".
    t = pa.table({"s": ["", "__--..__", "x", "sha256", None]})
    got = bt.from_arrow(t).select(r=col("s").str.to_case(style)).to_pydict()["r"]
    assert got[0] == ""
    assert got[1] == ""
    assert got[4] is None
    assert got[3].lower().replace("-", "").replace("_", "").replace(".", "") == "sha256"


@pytest.mark.parametrize("style", STYLES)
def test_recasing_is_idempotent_in_its_own_style(style):
    # Applying a style to its own output must be a no-op: if it were not, the splitter
    # and the joiner disagree about what a word boundary looks like.
    t = pa.table({"s": ["parseHTTPResponse", "user id", "sha256Digest", "a_bc-de"]})
    ds = bt.from_arrow(t)
    once = ds.select(r=col("s").str.to_case(style)).to_pydict()["r"]
    twice = (
        ds.select(r=col("s").str.to_case(style))
        .select(r=col("r").str.to_case(style))
        .to_pydict()["r"]
    )
    assert once == twice


def test_consecutive_single_letter_words_are_not_recoverable_from_camel_case():
    # A stated limit, not a bug: `camel`/`pascal` join without a separator, so the words
    # of `a_b_c` become `aBC`, which re-splits as two words (`a`, `BC`) because a run of
    # capitals reads as one acronym. Every separator style round-trips this input; the
    # separator-free ones cannot, and no splitter could.
    t = pa.table({"s": ["a_b_c"]})
    ds = bt.from_arrow(t)
    camel = ds.select(r=col("s").str.to_case("camel")).to_pydict()["r"]
    assert camel == ["aBC"]
    again = ds.select(r=col("s").str.to_case("camel")).select(r=col("r").str.to_case("camel"))
    assert again.to_pydict()["r"] == ["aBc"]
    for style in ("snake", "kebab", "dot", "upper_snake", "title", "train"):
        first = ds.select(r=col("s").str.to_case(style)).to_pydict()["r"]
        second = (
            ds.select(r=col("s").str.to_case(style))
            .select(r=col("r").str.to_case(style))
            .to_pydict()["r"]
        )
        assert first == second, style


def test_unknown_style_fails_at_plan_build():
    with pytest.raises(PlanError, match="style must be one of"):
        col("s").str.to_case("SNAKE")


def test_styles_agree_on_word_count():
    # The reason `to_case` is one function with a style rather than ten functions: every
    # style splits the input identically, so the word count never depends on the style.
    t = pa.table({"s": ["parseHTTPResponse", "a_b-c d", "utf8Bytes"]})
    ds = bt.from_arrow(t)
    snake = ds.select(r=col("s").str.to_case("snake")).to_pydict()["r"]
    kebab = ds.select(r=col("s").str.to_case("kebab")).to_pydict()["r"]
    dot = ds.select(r=col("s").str.to_case("dot")).to_pydict()["r"]
    for a, b, c in zip(snake, kebab, dot, strict=True):
        assert a.count("_") == b.count("-") == c.count(".")
