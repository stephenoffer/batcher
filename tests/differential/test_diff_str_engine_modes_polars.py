"""The `.str` parameters that restore Polars' reading of a function, run against Polars itself.

`test_diff_str_engine_modes` holds each parameter against DuckDB. This file runs the other
side for real: the installed Polars computes the answer a migrated script expects, and the
Batcher spelling a codemod would emit must reproduce it value for value, nulls, empty
strings and non-ASCII text included. Where no parameter is needed because an existing
spelling already agrees (`str.slice`), or because a composition of existing ones does
(`find`, `escape_regex`), the composition is what is tested, so the template a codemod
emits is itself pinned.
"""

from __future__ import annotations

import datetime

import polars as pl
import pyarrow as pa
import pytest

import batcher as bt
from batcher.plan.expr_ir.namespaces._dialect import escape_rust_regex

pytestmark = pytest.mark.differential

_WORDS = [
    "hello world",
    "a.b",
    "axb",
    "  \tpadded\n ",
    "\xa0nbsp\u3000",
    "",
    None,
    "héllo wörld",
    "-12",
    "+5",
    "HELLO-world",
    "a1b22",
]


def _batcher(expr, words=_WORDS) -> list:
    return bt.from_pydict({"s": words}).select(r=expr).to_pydict()["r"]


def _polars(expr, words=_WORDS) -> list:
    return pl.DataFrame({"s": words}, schema={"s": pl.String}).select(r=expr)["r"].to_list()


s, p = bt.col("s"), pl.col("s")


@pytest.mark.parametrize("pattern", [".", "a.b", "^h", "l+o", "é", r"\d+"])
def test_contains_regex_is_polars_default(pattern):
    assert _batcher(s.str.contains(pattern, literal=False)) == _polars(p.str.contains(pattern))
    assert _batcher(s.str.contains(pattern)) == _polars(p.str.contains(pattern, literal=True))


def test_strip_all_whitespace_is_polars_strip_chars():
    assert _batcher(s.str.trim(whitespace="all")) == _polars(p.str.strip_chars())
    assert _batcher(s.str.strip_chars_start(whitespace="all")) == _polars(p.str.strip_chars_start())
    assert _batcher(s.str.strip_chars_end(whitespace="all")) == _polars(p.str.strip_chars_end())


@pytest.mark.parametrize("width", [0, 4, 11, 20])
def test_non_truncating_pad_is_polars_pad(width):
    assert _batcher(s.str.lpad(width, "*", truncate=False)) == _polars(p.str.pad_start(width, "*"))
    assert _batcher(s.str.rpad(width, "*", truncate=False)) == _polars(p.str.pad_end(width, "*"))


@pytest.mark.parametrize("width", [0, 3, 5, 12])
def test_zfill_is_polars_zfill(width):
    # Polars counts a non-ASCII string's width in bytes here (and in characters in
    # `pad_start`); `zfill` counts characters, as Python's does, so the two agree on ASCII.
    ascii_words = [w for w in _WORDS if w is None or w.isascii()]
    assert _batcher(s.str.zfill(width), ascii_words) == _polars(p.str.zfill(width), ascii_words)


@pytest.mark.parametrize(("pattern", "group"), [(r"([a-z])(\d+)", 2), (r"l(l)?", 1), (r"(x)?b", 1)])
def test_extract_missing_null_is_polars_extract(pattern, group):
    got = _batcher(s.str.extract(pattern, group, missing="null"))
    assert got == _polars(p.str.extract(pattern, group))


@pytest.mark.parametrize(
    ("pattern", "value"), [("(l)", "[$1]"), ("(?<c>o)", "${c}!"), ("(o)", "$$"), ("(o)", "$1a")]
)
def test_dollar_replacement_is_polars_replace(pattern, value):
    every = s.str.replace_all(pattern, value, backrefs="dollar")
    first = s.str.regexp_replace(pattern, value, backrefs="dollar")
    assert _batcher(every) == _polars(p.str.replace_all(pattern, value))
    assert _batcher(first) == _polars(p.str.replace(pattern, value))


def test_a_pattern_without_groups_takes_its_replacement_literally_in_polars():
    # Polars expands `$` only when the pattern has a capture group; without one, `"$$"` is two
    # dollar signs. The default backslash syntax inserts a `$` literally, so that is the template.
    for value in ["$$", "$1", "x$"]:
        assert _batcher(s.str.replace_all("o", value)) == _polars(p.str.replace_all("o", value))
        assert _batcher(s.str.regexp_replace("o", value)) == _polars(p.str.replace("o", value))


def test_literal_first_replacement_template_is_polars_literal_replace():
    # `str.replace(pat, value, literal=True)` replaces the first occurrence of plain text. The
    # template escapes the pattern and doubles `$` so the replacement stays literal.
    for pattern, value in [(".", "$1"), ("l", "L"), ("", "^")]:
        template = s.str.regexp_replace(
            escape_rust_regex(pattern), value.replace("$", "$$"), backrefs="dollar"
        )
        assert _batcher(template) == _polars(p.str.replace(pattern, value, literal=True))


def test_lowercase_hex_and_binary_decoding_are_polars_encode_and_decode():
    assert _batcher(s.str.hex(case="lower")) == _polars(p.str.encode("hex"))
    assert _batcher(s.str.base64()) == _polars(p.str.encode("base64"))
    hexes = ["6869", "ff00", "", None, "zz", "abc"]
    assert _batcher(s.str.unhex(as_binary=True), hexes) == _polars(
        p.str.decode("hex", strict=False), hexes
    )
    b64 = ["aGk=", "/wA=", "", None, "!!"]
    assert _batcher(s.str.from_base64(as_binary=True), b64) == _polars(
        p.str.decode("base64", strict=False), b64
    )


def test_strict_date_parsing_raises_like_polars():
    good = ["2024-02-15", None]
    assert _batcher(s.str.to_date(strict=True), good) == [datetime.date(2024, 2, 15), None]
    assert _polars(p.str.to_date("%Y-%m-%d"), good) == [datetime.date(2024, 2, 15), None]
    bad = ["2024-02-15", "bad"]
    with pytest.raises(pl.exceptions.InvalidOperationError):
        _polars(p.str.to_date("%Y-%m-%d"), bad)
    with pytest.raises(Exception, match="does not match the format"):
        _batcher(s.str.to_date(strict=True), bad)
    assert _batcher(s.str.to_date(), bad) == _polars(p.str.to_date("%Y-%m-%d", strict=False), bad)


def test_concat_str_without_ignoring_nulls_is_polars_concat_str():
    data = {"a": ["x", None, "", None], "b": ["1", "2", None, None]}
    frame = pl.DataFrame(data)
    got = bt.from_pydict(data).select(r=bt.concat_str(bt.col("a"), bt.col("b"), ignore_nulls=False))
    assert got.to_pydict()["r"] == frame.select(r=pl.concat_str(["a", "b"]))["r"].to_list()
    # Polars' separator form, skipping nulls, is `concat_ws`.
    sep = frame.select(r=pl.concat_str(["a", "b"], separator="-", ignore_nulls=True))["r"]
    got_ws = bt.from_pydict(data).select(r=bt.concat_ws("-", bt.col("a"), bt.col("b")))
    assert got_ws.to_pydict()["r"] == sep.to_list()


def test_format_without_ignoring_nulls_is_polars_format():
    data = {"a": ["x", None, ""], "b": ["1", "2", None]}
    got = bt.from_pydict(data).select(
        r=bt.format_string("<{}|{}>", bt.col("a"), bt.col("b"), ignore_nulls=False)
    )
    want = pl.DataFrame(data).select(r=pl.format("<{}|{}>", "a", "b"))["r"].to_list()
    assert got.to_pydict()["r"] == want


@pytest.mark.parametrize("offset", [-12, -3, -1, 0, 1, 4, 20])
@pytest.mark.parametrize("length", [None, 0, 1, 3, 50])
def test_slice_already_is_polars_slice(offset, length):
    assert _batcher(s.str.slice(offset, length)) == _polars(p.str.slice(offset, length))


def test_categorical_slice_is_slice_over_the_decoded_strings():
    words = ["alpha", "beta", None, "alpha", "", "gämma"]
    table = pa.table({"s": pa.array(words).dictionary_encode()})
    got = bt.from_arrow(table).select(r=s.str.slice(1, 3)).to_pydict()["r"]
    frame = pl.DataFrame({"s": words}, schema={"s": pl.Categorical})
    assert got == frame.select(r=p.cat.slice(1, 3))["r"].to_list()


def test_escape_regex_template_is_polars_escape_regex():
    words = ["a b.c", "x-y~z#&", "", None, "\\ space", "é (1)"]
    template = s.str.escape_regex().str.replace("\\ ", " ")
    assert _batcher(template, words) == _polars(p.str.escape_regex(), words)


@pytest.mark.parametrize("pattern", ["l", "ö", "", "x*", "[0-9]+", "zzz"])
def test_find_template_is_polars_find(pattern):
    # Polars `find` is a 0-based *byte* offset of the first regex match, null when absent.
    before = s.str.regexp_split(pattern, limit=2).list.get(0).str.octet_length()
    regex = bt.when(s.str.contains(pattern, literal=False)).then(before).otherwise(bt.lit(None))
    assert _batcher(regex) == _polars(p.str.find(pattern))
    escaped = escape_rust_regex(pattern)
    plain = s.str.regexp_split(escaped, limit=2).list.get(0).str.octet_length()
    literal = bt.when(s.str.contains(pattern)).then(plain).otherwise(bt.lit(None))
    assert _batcher(literal) == _polars(p.str.find(pattern, literal=True))


def test_titlecase_default_is_polars_to_titlecase_except_after_a_digit():
    # Polars starts a word after a digit too (`'a1b22'` -> `'A1B22'`); DuckDB's `initcap`, the
    # default here, treats a digit as part of the word (`'A1b22'`).
    words = [w for w in _WORDS if w is None or not any(c.isdigit() for c in w)]
    assert _batcher(s.str.to_titlecase(), words) == _polars(p.str.to_titlecase(), words)
    assert _batcher(s.str.to_titlecase(), ["a1b22"]) == ["A1b22"]
    assert _polars(p.str.to_titlecase(), ["a1b22"]) == ["A1B22"]
