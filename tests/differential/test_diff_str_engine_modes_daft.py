"""The `.str` parameters that restore Daft's reading of a function, run against Daft itself.

The Daft half of `test_diff_str_engine_modes`. The oracle is the installed Daft, which
computes what a migrated script expects; the Batcher spelling a codemod would emit must
reproduce it. Daft is a benchmark dependency rather than a dev one, so this module stands
down where it is absent.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher.plan.expr_ir.namespaces._dialect import escape_rust_regex

daft = pytest.importorskip("daft")
F = pytest.importorskip("daft.functions")

pytestmark = pytest.mark.differential

_WORDS = [
    "hello world",
    "a.b.c",
    "  \tpadded\n ",
    "\xa0nbsp\xa0",
    "",
    None,
    "héllo wörld",
    "HELLO-world",
    "lol",
]

s = bt.col("s")


def _batcher(expr, words=_WORDS) -> list:
    return bt.from_pydict({"s": words}).select(r=expr).to_pydict()["r"]


def _daft(fn, words=_WORDS) -> list:
    frame = daft.from_pydict({"s": words}).select(fn(daft.col("s")).alias("r"))
    return frame.to_pydict()["r"]


def test_all_whitespace_strip_is_daft_strip():
    assert _batcher(s.str.trim(whitespace="all")) == _daft(F.strip)
    assert _batcher(s.str.strip_chars_start(whitespace="all")) == _daft(F.lstrip)
    assert _batcher(s.str.strip_chars_end(whitespace="all")) == _daft(F.rstrip)


@pytest.mark.parametrize(("pattern", "index"), [(r"l(l)?", 0), (r"l(x)?", 1), (r"(\w+) (\w+)", 2)])
def test_extract_missing_null_is_daft_regexp_extract(pattern, index):
    got = _batcher(s.str.extract(pattern, index, missing="null"))
    assert got == _daft(lambda c: F.regexp_extract(c, pattern, index))


@pytest.mark.parametrize(("pattern", "template"), [("(l)", "[$1]"), ("(h)(e)", "${2}${1}")])
def test_dollar_replacement_is_daft_regexp_replace(pattern, template):
    got = _batcher(s.str.replace_all(pattern, template, backrefs="dollar"))
    assert got == _daft(lambda c: F.regexp_replace(c, pattern, template))


@pytest.mark.parametrize("pattern", [".", "l", "lo", "ö", "zz"])
def test_literal_match_count_is_daft_count_matches(pattern):
    got = _batcher(s.str.count_matches(pattern, literal=True))
    assert got == _daft(lambda c: F.count_matches(c, pattern))


@pytest.mark.parametrize("target", ["abc", "hello", ""])
def test_restricted_damerau_levenshtein_is_daft_on_ascii_text(target):
    # Batcher counts UTF-8 bytes (DuckDB) where Daft counts characters, so the two agree on
    # ASCII text; `test_diff_str_engine_modes` holds the byte count on the rest.
    words = ["ca", "abc", "", None, "teh", "hlelo", "HELLO-world"]
    got = _batcher(s.str.damerau_levenshtein(target, restricted=True), words)
    assert got == _daft(lambda c: F.damerau_levenshtein_distance(c, daft.lit(target)), words)


@pytest.mark.parametrize("pattern", ["l", "ö", "", "zz", "."])
def test_find_template_is_daft_find(pattern):
    # Daft `find` is a 0-based byte offset, -1 when absent and null for a null input.
    offset = s.str.regexp_split(escape_rust_regex(pattern), limit=2).list.get(0).str.octet_length()
    template = (
        bt.when(s.str.contains(pattern)).then(offset).when(s.is_not_null()).then(bt.lit(-1))
    ).otherwise(bt.lit(None))
    assert _batcher(template) == _daft(lambda c: F.find(c, pattern))


def test_null_propagating_concat_and_format_are_daft_concat_and_format():
    data = {"a": ["x", None, ""], "b": ["1", "2", None]}
    frame = daft.from_pydict(data)
    want = frame.select(
        F.concat(daft.col("a"), daft.col("b")).alias("c"),
        F.format("<{}|{}>", daft.col("a"), daft.col("b")).alias("f"),
    ).to_pydict()
    a, b = bt.col("a"), bt.col("b")
    got = bt.from_pydict(data).select(
        c=bt.concat_str(a, b, ignore_nulls=False),
        f=bt.format_string("<{}|{}>", a, b, ignore_nulls=False),
    )
    assert got.to_pydict() == want


def test_replace_and_contains_already_are_daft_replace_and_contains():
    # Daft's `replace` and `contains` are literal, as `str.replace`/`str.contains` are.
    assert _batcher(s.str.replace("l", "L")) == _daft(lambda c: F.replace(c, "l", "L"))
    assert _batcher(s.str.contains(".")) == _daft(lambda c: F.contains(c, "."))
