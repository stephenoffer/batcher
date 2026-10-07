"""Unicode normalization, case folding, graphemes and struct-valued regex extraction.

Each function is held against an oracle that is not the engine: DuckDB where it has the
same function (`nfc_normalize`, `length_grapheme`, `substring_grapheme`, the struct form of
`regexp_extract`, `replace`), and Python's `unicodedata` / `str.casefold` where it does not.
The Python oracles run on the plan-time literals only, never on engine rows.

Every comparison is positional: a projection over `from_pydict` keeps row order, and each
expected value is computed for the same row, so nothing here is order-independent.
"""

from __future__ import annotations

import unicodedata

import pytest

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.plan.expr_ir.namespaces._dialect import regex_group_names

pytestmark = pytest.mark.differential

#: Composed vs decomposed accents, a compatibility ligature, full-width letters, Hangul
#: (which composes algorithmically), a ZWJ emoji, the empty string, duplicates, and a null.
_TEXTS = [
    "e\u0301",
    "\u00e9",
    "\u00e9",
    "\ufb01ne",
    "\uff21\uff22",
    "\u1100\u1161\u11a8",
    "\U0001f468\u200d\U0001f469\u200d\U0001f467",
    "Stra\u00dfe",
    "plain ascii",
    "",
    None,
]


def _column(expr) -> list:
    return bt.from_pydict({"s": _TEXTS}).select(r=expr).to_pydict()["r"]


def test_nfc_matches_duckdb_nfc_normalize(duck):
    got = _column(bt.col("s").str.normalize())
    want = [duck.execute("SELECT nfc_normalize(?)", [v]).fetchone()[0] for v in _TEXTS]
    assert got == want


@pytest.mark.parametrize("form", ["NFC", "NFD", "NFKC", "NFKD"])
def test_every_form_matches_python_unicodedata(form):
    got = _column(bt.col("s").str.normalize(form))
    assert got == [None if v is None else unicodedata.normalize(form, v) for v in _TEXTS]


def test_normalizing_makes_a_decomposed_key_join(duck):
    left = bt.from_pydict({"k": ["e\u0301"], "a": [1]})
    right = bt.from_pydict({"k": ["\u00e9"], "b": [2]})
    raw = left.join(right, on="k").to_pydict()
    assert raw["a"] == []
    normalized = left.with_columns(k=bt.col("k").str.normalize()).join(
        right.with_columns(k=bt.col("k").str.normalize()), on="k"
    )
    duck_rows = duck.execute(
        "SELECT count(*) FROM (SELECT nfc_normalize(?) AS k) l JOIN (SELECT nfc_normalize(?) AS k) "
        "r USING (k)",
        ["e\u0301", "\u00e9"],
    ).fetchone()[0]
    assert len(normalized.to_pydict()["a"]) == duck_rows == 1


def test_an_unknown_form_is_a_plan_error():
    with pytest.raises(PlanError, match="form"):
        bt.col("s").str.normalize("nfc")


def test_casefold_matches_python_str_casefold():
    texts = [*_TEXTS, "\u03a3\u0391\u03a3", "\u1e9e", "\u0130", "ABC"]
    got = bt.from_pydict({"s": texts}).select(r=bt.col("s").str.casefold()).to_pydict()["r"]
    assert got == [None if v is None else v.casefold() for v in texts]


def test_casefold_differs_from_lower_on_sharp_s():
    out = (
        bt.from_pydict({"s": ["Stra\u00dfe", "STRASSE"]})
        .select(lower=bt.col("s").str.lower(), folded=bt.col("s").str.casefold())
        .to_pydict()
    )
    assert out["lower"] == ["stra\u00dfe", "strasse"]
    assert out["folded"] == ["strasse", "strasse"]


def test_upper_uses_full_case_mapping_unlike_duckdb(duck):
    """Pinned, not hidden: the engine maps the sharp s to `SS` (Unicode full mapping, as Python and
    Polars do) where DuckDB maps per character to the capital sharp s."""
    got = bt.from_pydict({"s": ["Stra\u00dfe"]}).select(r=bt.col("s").str.upper()).to_pydict()
    assert got["r"] == ["STRASSE"] == ["Stra\u00dfe".upper()]
    assert duck.execute("SELECT upper('Stra\u00dfe')").fetchone()[0] == "STRA\u1e9eE"


# --- graphemes ----------------------------------------------------------------------------

_GRAPHEME_TEXTS = [
    "a\u0301bc\U0001f468\u200d\U0001f469\u200d\U0001f467d",
    "abcde",
    "a\nb",
    "\U0001f1fa\U0001f1f8\U0001f1eb\U0001f1f7",
    "",
    None,
]


def test_length_grapheme_matches_duckdb(duck):
    ds = bt.from_pydict({"s": _GRAPHEME_TEXTS})
    got = ds.select(r=bt.col("s").str.len_chars(unit="grapheme")).to_pydict()["r"]
    want = [duck.execute("SELECT length_grapheme(?)", [v]).fetchone()[0] for v in _GRAPHEME_TEXTS]
    assert got == want


def test_crlf_is_one_grapheme_unlike_duckdb(duck):
    """Pinned, not hidden: Unicode's grapheme rules (UAX #29, rule GB3) keep a CR LF pair
    together as one grapheme, and the engine follows them. DuckDB 1.5 counts it as two."""
    ds = bt.from_pydict({"s": ["a\r\nb"]})
    got = ds.select(r=bt.col("s").str.len_chars(unit="grapheme")).to_pydict()["r"]
    assert got == [3]
    assert (
        duck.execute("SELECT length_grapheme('a' || chr(13) || chr(10) || 'b')").fetchone()[0] == 4
    )


def test_byte_unit_is_octet_length(duck):
    ds = bt.from_pydict({"s": _GRAPHEME_TEXTS})
    got = ds.select(r=bt.col("s").str.len_chars(unit="byte")).to_pydict()["r"]
    want = [duck.execute("SELECT strlen(?)", [v]).fetchone()[0] for v in _GRAPHEME_TEXTS]
    assert got == want


@pytest.mark.parametrize("start", [-10, -6, -5, -2, -1, 0, 1, 2, 5, 9])
@pytest.mark.parametrize("length", [None, 0, 1, 3, 8])
def test_substring_grapheme_matches_duckdb(duck, start, length):
    ds = bt.from_pydict({"s": _GRAPHEME_TEXTS})
    got = ds.select(r=bt.col("s").str.substr(start, length, unit="grapheme")).to_pydict()["r"]
    if length is None:
        sql, args = "SELECT substring_grapheme(?, ?)", lambda v: [v, start]
    else:
        sql, args = "SELECT substring_grapheme(?, ?, ?)", lambda v: [v, start, length]
    want = [duck.execute(sql, args(v)).fetchone()[0] for v in _GRAPHEME_TEXTS]
    assert got == want, f"start={start} length={length}"


@pytest.mark.parametrize("start", [-3, -1, 2, 4, 5, 9])
@pytest.mark.parametrize("length", [-1, -2])
def test_a_negative_grapheme_length_reads_like_substr(start, length):
    """A negative length is not compared with DuckDB, whose `substring_grapheme` answers it
    inconsistently: on non-ASCII text `substring_grapheme(s, -2, -2)` returns the whole
    string for one input and two graphemes for another (DuckDB 1.5). The engine gives it
    `substr`'s reading instead, so on ASCII text, where a grapheme is a code point, the two
    units agree."""
    ds = bt.from_pydict({"s": ["abcde", "xy", ""]})
    out = ds.select(
        g=bt.col("s").str.substr(start, length, unit="grapheme"),
        c=bt.col("s").str.substr(start, length),
    ).to_pydict()
    assert out["g"] == out["c"]


def test_slice_in_graphemes_is_zero_based():
    ds = bt.from_pydict({"s": ["a\u0301bc"]})
    got = ds.select(r=bt.col("s").str.slice(0, 1, unit="grapheme")).to_pydict()["r"]
    assert got == ["a\u0301"]


def test_a_byte_slice_is_refused():
    with pytest.raises(PlanError, match="unit"):
        bt.col("s").str.slice(0, 1, unit="byte")


# --- extract_groups -----------------------------------------------------------------------

_LINES = ["GET /a 200", "POST /b/c", "nothing here", "", "GET /a 200", None]
_PATTERN = r"(?P<verb>[A-Z]+) (?P<path>/\S*)(?: (\d+))?"


def test_extract_groups_matches_duckdb_struct_regexp_extract(duck):
    names = regex_group_names(_PATTERN)
    assert names == ["verb", "path", "3"]
    got = (
        bt.from_pydict({"s": _LINES})
        .select(r=bt.col("s").str.extract_groups(_PATTERN))
        .to_pydict()["r"]
    )
    want = [
        duck.execute("SELECT regexp_extract(?, ?, ?)", [v, _PATTERN, names]).fetchone()[0]
        for v in _LINES
    ]
    assert got == want


def test_missing_null_answers_null_fields():
    got = (
        bt.from_pydict({"s": _LINES})
        .select(r=bt.col("s").str.extract_groups(_PATTERN, missing="null"))
        .to_pydict()["r"]
    )
    assert got[1] == {"verb": "POST", "path": "/b/c", "3": None}
    assert got[2] == {"verb": None, "path": None, "3": None}
    assert got[5] is None


@pytest.mark.parametrize(
    "pattern",
    [
        r"(a)(b)",
        r"(?P<x>a)(b)(?<y>c)",
        r"(?:a)(b)(?i:c)(?i)(d)",
        r"[(](a)\((b)",
        r"[]()](a)",
        r"[[:alpha:](](\d)",
    ],
)
def test_declared_fields_agree_with_the_engine(pattern):
    """The struct's field names are computed twice -- by the plan from the pattern text, and
    by the engine from the compiled regex. The declared schema must be what runs."""
    ds = bt.from_pydict({"s": ["ab"]}).select(r=bt.col("s").str.extract_groups(pattern))
    declared = ds.schema.field("r").type
    actual = ds.to_arrow().schema.field("r").type
    assert declared == actual
    assert [f.name for f in actual] == regex_group_names(pattern)


def test_a_pattern_without_groups_is_a_plan_error():
    with pytest.raises(PlanError, match="no capture group"):
        bt.col("s").str.extract_groups("abc")


# --- replace_all(literal=) ----------------------------------------------------------------


def test_replace_all_literal_matches_duckdb_replace(duck):
    texts = ["a.b.c", "abc", "...", "", None]
    ds = bt.from_pydict({"s": texts})
    got = ds.select(r=bt.col("s").str.replace_all(".", "x", literal=True)).to_pydict()["r"]
    want = [duck.execute("SELECT replace(?, '.', 'x')", [v]).fetchone()[0] for v in texts]
    assert got == want
    regex = ds.select(r=bt.col("s").str.replace_all(".", "x")).to_pydict()["r"]
    assert regex[0] == "xxxxx"


def test_replace_all_literal_refuses_dollar_backrefs():
    with pytest.raises(PlanError, match="backrefs"):
        bt.col("s").str.replace_all("a", "$1", literal=True, backrefs="dollar")
