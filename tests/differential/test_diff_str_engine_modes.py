"""The `.str` parameters that restore another engine's reading of a function, against DuckDB.

Batcher's string functions follow DuckDB. Where Polars, Spark, Daft or Ray Data answer the
same call differently, the one Batcher spelling grows a parameter (`contains(literal=)`,
`trim(whitespace=)`, `lpad(truncate=)`, `replace_all(backrefs=)`, ...). Each test here holds
both sides: the **default** still equals DuckDB, and the **parameter** equals DuckDB wherever
DuckDB can spell that other semantics (`regexp_matches`, `trim(s, chars)`, `||`, a `CASE`).
Where it cannot, the oracle is the other engine's own documented example, cited to its source.

The competitor engines that are installed run for real in the sibling files
(`..._polars.py`, `..._daft.py`); this file needs only DuckDB and Arrow.
"""

from __future__ import annotations

import re
import unicodedata

import duckdb
import pyarrow as pa
import pyarrow.compute as pc
import pytest

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.plan.expr_ir.namespaces._dialect import UNICODE_WHITE_SPACE

pytestmark = pytest.mark.differential

#: Nulls, the empty string, whitespace of every kind, non-ASCII text and regex metacharacters.
_WORDS = [
    "hello world",
    "a.b",
    "axb",
    "  \tpadded\n ",
    "\xa0nbsp\xa0",
    "",
    None,
    "héllo wörld",
    "-12",
    "+5",
    "HELLO-world",
    "aaa",
]


def _values(expr, data=None) -> list:
    return bt.from_pydict(data or {"s": _WORDS}).select(r=expr).to_pydict()["r"]


def _duck(duck, sql: str, data=None, params=None) -> list:
    duck.register("t", pa.table(data or {"s": _WORDS}))
    rows = duck.execute(f"SELECT {sql} AS r FROM t", params or []).fetchall()
    return [row[0] for row in rows]


s = bt.col("s")


@pytest.mark.parametrize("pattern", [".", "a.b", "^h", "l+o", "é", "d$"])
def test_contains_is_literal_by_default_and_a_regex_when_asked(duck, pattern):
    assert _values(s.str.contains(pattern)) == _duck(duck, "contains(s, ?)", params=[pattern])
    assert _values(s.str.contains(pattern, literal=False)) == _duck(
        duck, "regexp_matches(s, ?)", params=[pattern]
    )


def test_trim_all_whitespace_is_the_white_space_set_as_characters(duck):
    for method, sql in [
        ("trim", "trim"),
        ("strip_chars_start", "ltrim"),
        ("strip_chars_end", "rtrim"),
    ]:
        default = getattr(s.str, method)()
        every = getattr(s.str, method)(whitespace="all")
        assert _values(default) == _duck(duck, f"{sql}(s)"), method
        assert _values(every) == _duck(duck, f"{sql}(s, ?)", params=[UNICODE_WHITE_SPACE]), method
    # Python's `str.strip()` is the `White_Space` strip Polars and Daft perform.
    assert _values(s.str.trim(whitespace="all")) == [w if w is None else w.strip() for w in _WORDS]


def test_trim_rejects_chars_together_with_all_whitespace():
    with pytest.raises(PlanError, match="not both"):
        s.str.trim("x", whitespace="all")
    with pytest.raises(PlanError, match="whitespace"):
        s.str.trim(whitespace="unicode")


@pytest.mark.parametrize("width", [0, 3, 5, 12])
def test_pad_keeps_a_longer_string_whole_when_not_truncating(duck, width):
    for fn in ("lpad", "rpad"):
        truncating = getattr(s.str, fn)(width, "*")
        whole = getattr(s.str, fn)(width, "*", truncate=False)
        assert _values(truncating) == _duck(duck, f"{fn}(s, {width}, '*')"), fn
        assert _values(whole) == _duck(
            duck, f"CASE WHEN length(s) >= {width} THEN s ELSE {fn}(s, {width}, '*') END"
        ), fn


def test_zfill_keeps_the_sign_in_front_as_python_does():
    data = {"s": ["7", "-12", "+5", "-", "+", "", "1234567", "a.b", "--1", None, "é"]}
    got = _values(bt.col("s").str.zfill(5), data)
    assert got == [w if w is None else w.zfill(5) for w in data["s"]]


def test_hex_lowercase_is_lower_of_hex(duck):
    assert _values(s.str.hex()) == _duck(duck, "hex(s)")
    assert _values(s.str.hex(case="lower")) == _duck(duck, "lower(hex(s))")


def test_decoding_to_binary_is_duckdbs_own_blob_result(duck):
    data = {"s": ["616263", "FF00", "", None, "c3a9"]}
    assert _values(bt.col("s").str.unhex(as_binary=True), data) == _duck(duck, "unhex(s)", data)
    b64 = {"s": ["YWJj", "/wA=", "", None, "w6k="]}
    assert _values(bt.col("s").str.from_base64(as_binary=True), b64) == _duck(
        duck, "from_base64(s)", b64
    )
    # The text form still nulls bytes that are not UTF-8; the binary form keeps them.
    assert _values(bt.col("s").str.unhex(), data) == ["abc", None, "", None, "é"]
    assert _values(bt.col("s").str.unhex(as_binary=True), {"s": ["zz", "abc"]}) == [None, None]


def test_extract_missing_null_is_null_exactly_where_the_pattern_misses(duck):
    pattern = r"([a-z])([0-9]+)"
    data = {"s": ["a12", "b", "", None, "x9y8"]}
    assert _values(bt.col("s").str.extract(pattern, 2), data) == _duck(
        duck, f"regexp_extract(s, '{pattern}', 2)", data
    )
    # Every group takes part whenever this pattern matches, so "missing" is "no match".
    assert _values(bt.col("s").str.extract(pattern, 2, missing="null"), data) == _duck(
        duck,
        f"CASE WHEN regexp_matches(s, '{pattern}') THEN regexp_extract(s, '{pattern}', 2) END",
        data,
    )
    # A group that sat out a match is null too, where the default says ''.
    optional = {"s": ["b", "b1"]}
    assert _values(bt.col("s").str.extract("b([0-9])?", 1), optional) == ["", "1"]
    assert _values(bt.col("s").str.extract("b([0-9])?", 1, missing="null"), optional) == [None, "1"]


def test_extract_all_missing_empty_turns_a_null_element_into_empty(duck):
    pattern = "[a-z]([0-9])?"
    data = {"s": ["a1b", "", None, "c2d3"]}
    assert _values(bt.col("s").str.extract_all(pattern, 1), data) == _duck(
        duck, f"regexp_extract_all(s, '{pattern}', 1)", data
    )
    assert _values(bt.col("s").str.extract_all(pattern, 1, missing="empty"), data) == _duck(
        duck, f"list_transform(regexp_extract_all(s, '{pattern}', 1), x -> coalesce(x, ''))", data
    )


def test_extract_all_group_one_reproduces_sparks_documented_example():
    # `regexpExpressions.scala`, `RegExpExtractAll`:
    # SELECT regexp_extract_all('100-200, 300-400', '(\\d+)-(\\d+)', 1) -> ["100","300"]
    data = {"s": ["100-200, 300-400"]}
    assert _values(bt.col("s").str.extract_all(r"(\d+)-(\d+)", 1), data) == [["100", "300"]]


@pytest.mark.parametrize(
    ("pattern", "rust_template", "re2_template"),
    [
        ("(l)", "[$1]", r"[\1]"),
        ("(h)(e)", "$2$1", r"\2\1"),
        ("o", "$$", "$"),
        ("(l)", "${1}x", r"\1x"),
    ],
)
def test_dollar_backreferences_are_the_re2_template_translated(
    duck, pattern, rust_template, re2_template
):
    first = s.str.regexp_replace(pattern, rust_template, backrefs="dollar")
    every = s.str.replace_all(pattern, rust_template, backrefs="dollar")
    assert _values(first) == _duck(duck, "regexp_replace(s, ?, ?)", params=[pattern, re2_template])
    assert _values(every) == _duck(
        duck, "regexp_replace(s, ?, ?, 'g')", params=[pattern, re2_template]
    )


def test_the_default_replacement_reads_backslash_groups_and_a_literal_dollar(duck):
    assert _values(s.str.replace_all("(l)", "[$1]")) == _duck(
        duck, "regexp_replace(s, '(l)', '[$1]', 'g')"
    )
    assert _values(s.str.regexp_replace("(l)", r"[\1]")) == _duck(
        duck, r"regexp_replace(s, '(l)', '[\1]')"
    )
    with pytest.raises(PlanError, match="backrefs"):
        s.str.replace_all("a", "b", backrefs="java")


def test_regexp_split_limit_reproduces_sparks_documented_examples():
    # `regexpExpressions.scala`, `StringSplit`:
    #   split('oneAtwoBthreeC', '[ABC]')     -> ["one","two","three",""]
    #   split('oneAtwoBthreeC', '[ABC]', -1) -> ["one","two","three",""]
    #   split('oneAtwoBthreeC', '[ABC]', 2)  -> ["one","twoBthreeC"]
    data = {"s": ["oneAtwoBthreeC"]}
    col = bt.col("s").str
    assert _values(col.regexp_split("[ABC]"), data) == [["one", "two", "three", ""]]
    assert _values(col.regexp_split("[ABC]", limit=-1), data) == [["one", "two", "three", ""]]
    assert _values(col.regexp_split("[ABC]", limit=2), data) == [["one", "twoBthreeC"]]


@pytest.mark.parametrize("limit", [1, 2, 3, 10])
def test_regexp_split_limit_is_a_split_count(limit):
    data = {"s": ["a1b22c333d", "", None, "none"]}
    got = _values(bt.col("s").str.regexp_split("[0-9]+", limit=limit), data)
    # Python's `maxsplit=0` means "no limit", so a limit of one piece is the string itself.
    want = [
        None if w is None else re.split("[0-9]+", w, maxsplit=limit - 1) if limit > 1 else [w]
        for w in data["s"]
    ]
    assert got == want


def test_a_limited_regexp_split_is_not_rewritten_into_an_unlimited_split():
    # `regexp_split_plain_to_split` rewrites a metacharacter-free pattern to `split`,
    # which has no limit slot; the rewrite must not fire on a limited call.
    data = {"s": ["a-b-c"]}
    assert _values(bt.col("s").str.regexp_split("-", limit=2), data) == [["a", "b-c"]]
    assert _values(bt.col("s").str.regexp_split("-"), data) == [["a", "b", "c"]]


def test_xxhash64_seed_42_is_sparks_hash_of_one_string_column():
    data = {"s": ["ABC", None]}
    # Spark: SELECT xxhash64('ABC') -> 4105715581806190027 (seed 42). The Rust unit test
    # `xxhash64_seed_reproduces_sparks_documented_example` derives the same seed rule from the
    # documented multi-column example in `hash.scala`.
    assert _values(bt.col("s").str.xxhash64(seed=42), data) == [4105715581806190027, None]
    assert _values(bt.col("s").str.xxhash64(), data)[0] == -1843406881296486760
    assert _values(bt.col("s").str.xxhash64(seed=0), data)[0] == -1843406881296486760
    binary = {"s": [b"ABC"]}
    assert _values(bt.col("s").str.xxhash64(seed=42), binary) == [4105715581806190027]


def test_form_url_coding_reproduces_sparks_documented_examples_and_javas_rules():
    # `urlExpressions.scala`: url_encode('https://spark.apache.org') ->
    # 'https%3A%2F%2Fspark.apache.org', and url_decode of that is the URL again.
    url = {"s": ["https://spark.apache.org"]}
    assert _values(bt.col("s").str.url_encode(form=True), url) == ["https%3A%2F%2Fspark.apache.org"]
    back = {"s": ["https%3A%2F%2Fspark.apache.org"]}
    assert _values(bt.col("s").str.url_decode(form=True), back) == ["https://spark.apache.org"]
    # Java's URLEncoder keeps `.-*_`, writes a space as `+`, and percent-encodes `~`.
    data = {"s": ["a b*~+é", "", None]}
    assert _values(bt.col("s").str.url_encode(form=True), data) == ["a+b*%7E%2B%C3%A9", "", None]
    assert _values(bt.col("s").str.url_encode(), data) == ["a%20b%2A~%2B%C3%A9", "", None]
    plus = {"s": ["a+b%2Bc", None]}
    assert _values(bt.col("s").str.url_decode(form=True), plus) == ["a b+c", None]
    assert _values(bt.col("s").str.url_decode(), plus) == ["a+b+c", None]


def test_titlecase_on_spaces_reproduces_sparks_initcap():
    # `stringExpressions.scala`, `InitCap`: SELECT initcap('sPark sql') -> 'Spark Sql'. Words
    # start after an ASCII space only (`CollationAwareUTF8String.toTitleCaseICU`).
    data = {"s": ["sPark sql", "hello-world", "a\tb", "  two  spaces", "", None, "ÉCOLE élève"]}
    got = _values(bt.col("s").str.to_titlecase(boundary="space"), data)
    assert got == ["Spark Sql", "Hello-world", "A\tb", "  Two  Spaces", "", None, "École Élève"]
    assert _values(bt.col("s").str.to_titlecase(), data)[1] == "Hello-World"


def _osa(a: bytes, b: bytes) -> int:
    d = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        d[i][0] = i
    for j in range(len(b) + 1):
        d[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = a[i - 1] != b[j - 1]
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[len(a)][len(b)]


@pytest.mark.parametrize("target", ["abc", "", "hello", "ca"])
def test_restricted_damerau_levenshtein_is_optimal_string_alignment(duck, target):
    data = {"s": ["ca", "abc", "", None, "teh", "hlelo", "héllo"]}
    assert _values(bt.col("s").str.damerau_levenshtein(target), data) == _duck(
        duck, "damerau_levenshtein(s, ?)", data, [target]
    )
    got = _values(bt.col("s").str.damerau_levenshtein(target, restricted=True), data)
    want = [None if w is None else _osa(w.encode(), target.encode()) for w in data["s"]]
    assert got == want


def test_strict_date_parsing_raises_where_the_lenient_form_nulls(duck):
    good = {"s": ["2024-02-15", None]}
    assert _values(bt.col("s").str.to_date(strict=True), good) == _duck(
        duck, "CAST(strptime(s, '%Y-%m-%d') AS DATE)", good
    )
    bad = {"s": ["2024-02-15", "bad", None]}
    assert _values(bt.col("s").str.to_date(), bad) == _duck(
        duck, "CAST(try_strptime(s, '%Y-%m-%d') AS DATE)", bad
    )
    with pytest.raises(Exception, match="does not match the format"):
        _values(bt.col("s").str.to_date(strict=True), bad)
    with pytest.raises(Exception, match="does not match the format"):
        _values(bt.col("s").str.to_datetime("%Y-%m-%d", strict=True), bad)
    duck.register("t", pa.table(bad))
    with pytest.raises(duckdb.Error):
        duck.execute("SELECT strptime(s, '%Y-%m-%d') FROM t").fetchall()


@pytest.mark.parametrize("pattern", [".", "a", "aa", "l", "ö", "-"])
def test_literal_match_count_counts_non_overlapping_occurrences(duck, pattern):
    literal = s.str.regexp_count(pattern, literal=True)
    assert _values(literal) == _duck(
        duck, "(length(s) - length(replace(s, ?, ''))) // length(?)", params=[pattern, pattern]
    )
    regex = s.str.regexp_count(pattern)
    assert _values(regex) == _duck(duck, "len(regexp_extract_all(s, ?))", params=[pattern])


def test_concat_str_propagates_nulls_when_not_ignoring_them(duck):
    data = {"a": ["x", None, "", None], "b": ["1", "2", None, None]}
    a, b = bt.col("a"), bt.col("b")
    assert _values(bt.concat_str(a, b), data) == _duck(duck, "concat(a, b)", data)
    assert _values(bt.concat_str(a, b, ignore_nulls=False), data) == _duck(duck, "a || b", data)


def test_format_string_propagates_nulls_when_not_ignoring_them(duck):
    data = {"a": ["x", None, ""], "b": [1, 2, None]}
    a, b = bt.col("a"), bt.col("b")
    assert _values(bt.format_string("{}={}", a, b), data) == ["x=1", "=2", "="]
    assert _values(bt.format_string("{}={}", a, b, ignore_nulls=False), data) == _duck(
        duck, "format('{}={}', a, b)", data
    )


def test_require_cased_case_predicates_are_arrow_and_python():
    # Ray Data's `str.is_lower`/`is_upper` are Arrow's `utf8_is_lower`/`utf8_is_upper`
    # (`ray/data/namespace_expressions/string_namespace.py`).
    words = ["", "abc", "ABC", "Abc", "123", "a1", "A1", "ß", "É", "é", " ", None, "ǅ"]
    data = {"s": words}
    array = pa.array(words)
    col = bt.col("s").str
    assert _values(col.is_lower(require_cased=True), data) == pc.utf8_is_lower(array).to_pylist()
    assert _values(col.is_upper(require_cased=True), data) == pc.utf8_is_upper(array).to_pylist()
    assert _values(col.is_lower(require_cased=True), data) == [
        None if w is None else w.islower() for w in words
    ]
    # The default is unchanged: a string without cased characters equals its case forms.
    assert _values(col.is_lower(), data)[:5] == [True, True, False, False, True]


def _spark_mask(value, upper, lower, digit, other):
    table = {"Lu": upper, "Ll": lower, "Nd": digit}
    out = []
    for ch in value:
        units = 2 if ord(ch) > 0xFFFF else 1
        category = unicodedata.category(ch) if units == 1 else "Cs"
        replacement = table.get(category, other)
        out.append(ch if replacement is None else replacement * units)
    return "".join(out)


@pytest.mark.parametrize(
    "classes",
    [("X", "x", "n", None), ("Q", "q", "d", "o"), (None, "q", "d", "o"), (None, None, None, "*")],
)
def test_class_masking_reproduces_sparks_mask(classes):
    # `maskExpressions.scala`: mask('AbCD123-@$#', 'Q', 'q', 'd', 'o') -> 'QqQQdddoooo' and
    # mask('AbCD123-@$#', NULL, 'q', 'd', 'o') -> 'AqCDdddoooo'; the Rust unit test pins every
    # documented example, and this holds the categories on text Spark's examples do not reach.
    upper, lower, digit, other = classes
    data = {"s": ["AbCD123-@$#", "abcd-EFGH-8765-4321", "Éé٣ x", "", None, "a😀B"]}
    got = _values(bt.mask(bt.col("s"), upper=upper, lower=lower, digit=digit, other=other), data)
    want = [None if w is None else _spark_mask(w, upper, lower, digit, other) for w in data["s"]]
    assert got == want


def test_class_masking_refuses_the_reveal_form_beside_it():
    with pytest.raises(PlanError, match="character class"):
        bt.mask(bt.col("s"), upper="X", show_last=4)
    with pytest.raises(PlanError, match="one character"):
        bt.mask(bt.col("s"), upper="XY")
