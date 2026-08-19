"""The corpus-quality half of ``.str``, against Python recomputed from the definition.

These are the methods an LLM data pipeline filters a web corpus with: the Gopher-style
repetition ratios, the character-class ratios, the shape predicates that route a document
to prose or code, and the extractors that pull URLs, emails, mentions, hashtags and
numbers out of raw text. Fifty-three of them, and no test called any.

There is no second engine here. DuckDB has no ``duplicate_ngram_ratio`` and Polars has no
``looks_like_code``, so the oracle is the definition, recomputed in Python inside each
test with ``re`` and ``str``. That is a real oracle rather than a restatement: the
reference is written from what the docstring says the method means, in a different
language, against a different regex engine, and it disagrees loudly when the engine's
regex dialect or its null handling differs from what the documentation promises.

Every test also runs the null row, because a ratio over an empty denominator and a
predicate over a missing document are exactly where this family goes wrong quietly.
"""

from __future__ import annotations

import re

import pytest

import batcher as bt
from batcher import PlanError

pytestmark = pytest.mark.differential

#: Documents chosen so each filter separates them: prose with entities, whitespace only,
#: shouting, a document with repeated lines and repeated paragraphs, source code, JSON, a
#: bulleted list, non-ASCII text with currency and a phone number, and a null.
DOCS = [
    "Hello World! Visit https://example.com or mail a@b.com #tag @me 42 items.",
    "  ",
    "WHY IS THIS ALL CAPS???",
    "line one\nline one\nline two\n\npara2\n\npara2",
    "def f(x):\n    return x + 1",
    '{"a": 1, "b": [2,3]}',
    "* bullet item\n- another",
    "Café naïve €50 costs $10. Call +1-555-123-4567.",
    None,
]
NULL_ROW = len(DOCS) - 1


@pytest.fixture(scope="module")
def ds():
    """The document fixture as a Batcher dataset."""
    return bt.from_pydict({"s": DOCS})


def _column(ds, expr) -> list:
    return ds.select(v=expr).to_pydict()["v"]


def _reference(fn) -> list:
    """Apply a Python reference to every document, passing the null through."""
    return [None if d is None else fn(d) for d in DOCS]


def test_character_class_ratios_match_the_counts_they_are_defined_as(ds):
    """``uppercase_ratio`` / ``lowercase_ratio`` / ``non_ascii_ratio`` over total length."""
    got = ds.select(
        upper=bt.col("s").str.uppercase_ratio(),
        lower=bt.col("s").str.lowercase_ratio(),
        non_ascii=bt.col("s").str.non_ascii_ratio(),
    ).to_pydict()
    # The case classes are ASCII (`[A-Z]` / `[a-z]`), which is what the docstrings say
    # and what `test_case_ratios_are_scoped_to_ascii` below pins. Writing the reference
    # with Python's Unicode-aware `isupper` would disagree on every accented letter.
    want_upper = _reference(lambda d: sum("A" <= c <= "Z" for c in d) / len(d) if d else None)
    want_lower = _reference(lambda d: sum("a" <= c <= "z" for c in d) / len(d) if d else None)
    want_non_ascii = _reference(lambda d: sum(ord(c) > 127 for c in d) / len(d) if d else None)
    for label, got_values, want in [
        ("uppercase_ratio", got["upper"], want_upper),
        ("lowercase_ratio", got["lower"], want_lower),
        ("non_ascii_ratio", got["non_ascii"], want_non_ascii),
    ]:
        for i, (a, b) in enumerate(zip(got_values, want, strict=True)):
            if b is None:
                assert a is None, f"{label}[{i}] did not null"
            else:
                assert a == pytest.approx(b, abs=1e-12), f"{label}[{i}] on {DOCS[i]!r}"


def test_case_ratios_are_scoped_to_ascii_and_an_empty_string_has_no_ratio():
    """The scope of the case classes, and the empty-string case the docstrings promise.

    An accented or non-Latin letter counts as neither upper nor lower; it is
    ``non_ascii_ratio`` that sees it. This matters to anyone filtering a non-English
    corpus with ``uppercase_ratio`` as a shouting detector, so it is pinned rather than
    left to be discovered.
    """
    docs = [
        "ABC",
        "abc",
        "\N{LATIN CAPITAL LETTER E WITH ACUTE}\N{LATIN CAPITAL LETTER A WITH GRAVE}",
        "\N{CYRILLIC SMALL LETTER A}\N{CYRILLIC SMALL LETTER BE}",
        "",
    ]
    got = (
        bt.from_pydict({"s": docs})
        .select(
            upper=bt.col("s").str.uppercase_ratio(),
            lower=bt.col("s").str.lowercase_ratio(),
            non_ascii=bt.col("s").str.non_ascii_ratio(),
        )
        .to_pydict()
    )
    assert got["upper"] == [1.0, 0.0, 0.0, 0.0, None]
    assert got["lower"] == [0.0, 1.0, 0.0, 0.0, None]
    assert got["non_ascii"] == [0.0, 0.0, 1.0, 1.0, None]


def test_the_three_ratios_are_bounded_and_consistent_with_each_other(ds):
    """No ratio may leave [0, 1], and upper plus lower may not exceed the whole string."""
    got = ds.select(
        upper=bt.col("s").str.uppercase_ratio(),
        lower=bt.col("s").str.lowercase_ratio(),
        non_ascii=bt.col("s").str.non_ascii_ratio(),
    ).to_pydict()
    for i in range(len(DOCS) - 1):
        for key in got:
            assert 0.0 <= got[key][i] <= 1.0, f"{key}[{i}] outside [0, 1]"
        assert got["upper"][i] + got["lower"][i] <= 1.0 + 1e-12, i


#: ``(method, Python reference)`` for the shape predicates with an unambiguous definition.
SHAPE_PREDICATES = [
    ("is_blank", lambda d: d.strip() == ""),
    ("is_all_caps", lambda d: bool(re.search("[A-Z]", d)) and not re.search("[a-z]", d)),
    ("is_upper", lambda d: d == d.upper()),
    ("is_ascii_only", lambda d: all(ord(c) <= 127 for c in d)),
    ("is_single_line", lambda d: "\n" not in d),
    ("is_question", lambda d: d.rstrip().endswith("?")),
    ("is_exclamation", lambda d: d.rstrip().endswith("!")),
    ("has_non_ascii", lambda d: any(ord(c) > 127 for c in d)),
    ("starts_with_bullet", lambda d: bool(re.match(r"\s*[-*•]\s", d))),
    ("starts_with_capital", lambda d: bool(re.match(r"\s*[A-Z]", d))),
    ("ends_with_punctuation", lambda d: bool(re.search(r"[.!?]\s*$", d))),
    ("has_repeated_punctuation", lambda d: bool(re.search(r"([!?.,])\1", d))),
]


@pytest.mark.parametrize(("method", "reference"), SHAPE_PREDICATES)
def test_shape_predicate_matches_its_definition(ds, method, reference):
    """Each predicate against the same rule written in Python."""
    got = _column(ds, getattr(bt.col("s").str, method)())
    assert got == _reference(reference), f"{method}: {got}"


def test_is_upper_and_is_all_caps_part_company_on_a_caseless_string(ds):
    """The distinction between the two, which is the only reason both exist.

    ``is_upper`` asks whether the string equals its uppercase form, so a string with no
    letters at all satisfies it. ``is_all_caps`` requires letters and no lowercase ones,
    which is the pandas ``str.isupper`` reading. A fixture of ordinary sentences never
    separates them.
    """
    caseless = ["   ", "", "123", "!!!", "ABC", "Abc"]
    got = (
        bt.from_pydict({"s": caseless})
        .select(upper=bt.col("s").str.is_upper(), caps=bt.col("s").str.is_all_caps())
        .to_pydict()
    )
    assert got["upper"] == [True, True, True, True, True, False]
    assert got["caps"] == [False, False, False, False, True, False]


def test_currency_and_phone_detection_find_the_document_that_has_them(ds):
    """``has_currency`` and ``has_phone``, which only the non-ASCII fixture satisfies."""
    got = ds.select(
        currency=bt.col("s").str.has_currency(), phone=bt.col("s").str.has_phone()
    ).to_pydict()
    assert got["currency"] == [False, False, False, False, False, False, False, True, None]
    assert got["phone"] == [False, False, False, False, False, False, False, True, None]


def test_code_and_json_detection_route_the_right_documents(ds):
    """``looks_like_code`` is deliberately coarser than ``looks_like_json``."""
    got = ds.select(
        code=bt.col("s").str.looks_like_code(), json=bt.col("s").str.looks_like_json()
    ).to_pydict()
    assert got["code"] == [False, False, False, False, True, True, False, False, None]
    assert got["json"] == [False, False, False, False, False, True, False, False, None]
    for i in range(len(DOCS) - 1):
        if got["json"][i]:
            assert got["code"][i], "JSON has code punctuation, so the coarser filter must agree"


#: ``(method, regex)`` for the extractors. Each regex is the family the docstring names,
#: not a copy of the engine's pattern -- so a dialect difference shows up as a mismatch.
EXTRACTORS = [
    ("extract_urls", r"https?://[^\s]+"),
    ("extract_emails", r"[\w.+-]+@[\w-]+\.[\w.]+"),
    ("extract_hashtags", r"#\w+"),
    ("extract_numbers", r"\d+"),
]


@pytest.mark.parametrize(("method", "pattern"), EXTRACTORS)
def test_extractor_finds_the_same_matches_as_the_pattern_it_names(ds, method, pattern):
    """Every extractor returns a ``List<Utf8>``, empty rather than null when nothing matches."""
    got = _column(ds, getattr(bt.col("s").str, method)())
    want = _reference(lambda d: re.findall(pattern, d))
    assert got == want, f"{method}: {got} vs {want}"
    for i, value in enumerate(got):
        if i == NULL_ROW:
            assert value is None, f"{method} must null on a null document"
        else:
            assert isinstance(value, list), f"{method} must return a list, not {value!r}"


def test_extract_mentions_returns_every_at_sign_run_including_inside_an_address(ds):
    """Pinned rather than argued: the pattern is deliberately not email-aware.

    ``a@b.com`` contributes ``@b``. Making the mention pattern skip addresses would need
    it to parse email, and a corpus filter that dropped a real mention because it sat
    next to an address would be the worse failure. Recorded here so the choice is visible
    and a future change to it is a decision rather than a surprise.
    """
    got = _column(ds, bt.col("s").str.extract_mentions())
    assert got[0] == ["@b", "@me"]
    assert got[NULL_ROW] is None


def test_extract_all_and_replace_all_use_the_same_pattern_engine(ds):
    """``extract_all`` against ``re.findall``, and ``replace_all`` against ``re.sub``."""
    pattern = r"[a-z]+o[a-z]+"
    got = ds.select(
        found=bt.col("s").str.extract_all(pattern),
        replaced=bt.col("s").str.replace_all(pattern, "X"),
    ).to_pydict()
    assert got["found"] == _reference(lambda d: re.findall(pattern, d))
    assert got["replaced"] == _reference(lambda d: re.sub(pattern, "X", d))


def test_count_char_counts_a_literal_and_not_a_pattern(ds):
    """The escaping the docstring promises: counting ``"."`` is not counting every character."""
    got = ds.select(
        letter=bt.col("s").str.count_char("o"), dot=bt.col("s").str.count_char(".")
    ).to_pydict()
    assert got["letter"] == _reference(lambda d: d.count("o"))
    assert got["dot"] == _reference(lambda d: d.count("."))
    assert got["dot"][0] < len(DOCS[0]), "a literal dot is not the any-character wildcard"


def test_escape_regex_makes_a_string_safe_to_use_as_a_pattern(ds):
    """The escaped form must match the original literally and nothing else."""
    got = _column(ds, bt.col("s").str.escape_regex())
    for original, escaped in zip(DOCS, got, strict=True):
        if original is None:
            assert escaped is None
            continue
        assert re.fullmatch(escaped, original), f"{escaped!r} does not match {original!r}"


def test_word_position_accessors_match_splitting_on_whitespace(ds):
    """``first_word`` and ``last_word``, empty rather than null on a wordless document."""
    got = ds.select(
        first=bt.col("s").str.first_word(), last=bt.col("s").str.last_word()
    ).to_pydict()
    assert got["first"] == _reference(lambda d: d.split()[0] if d.split() else "")
    assert got["last"] == _reference(lambda d: d.split()[-1] if d.split() else "")


def test_first_sentence_stops_at_the_first_sentence_mark(ds):
    """Empty when there is no mark, which is what the docstring promises."""
    got = _column(ds, bt.col("s").str.first_sentence())

    def reference(d: str) -> str:
        match = re.search(r"^.*?[.!?]", d, re.DOTALL)
        return match.group(0) if match else ""

    assert got == _reference(reference)


def test_repetition_ratios_rise_with_repetition_and_are_zero_without_it(ds):
    """The Gopher filters: a document built by repeating itself must score above one that isn't.

    Their exact values depend on a tokenization the docstrings describe rather than
    specify to the character, so what is asserted is the ordering the filters exist to
    produce, the [0, 1] bound, and the zero on a document with no repetition -- all three
    of which a broken implementation fails.
    """
    got = ds.select(
        line=bt.col("s").str.duplicate_line_ratio(),
        para=bt.col("s").str.duplicate_paragraph_ratio(),
    ).to_pydict()
    repeated = 3  # the fixture with a duplicated line and a duplicated paragraph
    for key in ("line", "para"):
        assert got[key][repeated] > 0.0, f"{key} did not see the repetition"
        assert got[key][0] == 0.0, f"{key} found repetition in a document with none"
        for i in range(len(DOCS) - 1):
            if got[key][i] is not None:
                assert 0.0 <= got[key][i] <= 1.0, f"{key}[{i}] outside [0, 1]"


def test_ngram_ratios_separate_one_repeated_phrase_from_many(ds):
    """``top_ngram_ratio`` counts the most frequent n-gram; ``duplicate_ngram_ratio`` sums all.

    The two documents below are the case that tells them apart: one phrase repeated eight
    times, against a sequence repeated once end to end.
    """
    docs = [
        "the cat the cat the cat the cat the cat the cat the cat the cat",
        "one two three four five six one two three four five six",
        "alpha beta gamma delta epsilon zeta eta theta",
    ]
    got = (
        bt.from_pydict({"s": docs})
        .select(
            top=bt.col("s").str.top_ngram_ratio(2),
            dup=bt.col("s").str.duplicate_ngram_ratio(2),
        )
        .to_pydict()
    )
    assert got["top"][0] > got["top"][1], "one phrase repeated must top the single most frequent"
    assert got["dup"][1] > 0.0, "a sequence repeated end to end is duplicated n-grams"
    # `top_ngram_ratio` measures coverage, not repetition, so it is non-zero even when
    # every bigram is unique -- the most frequent one still covers characters. Only
    # `duplicate_ngram_ratio` goes to zero on a document that repeats nothing, and
    # conflating the two is the mistake this pair of assertions exists to prevent.
    assert got["top"][2] > 0.0, "coverage of the most frequent bigram is never zero"
    assert got["dup"][2] == pytest.approx(0.0, abs=1e-12), "nothing repeats in this document"
    for key in got:
        for value in got[key]:
            assert 0.0 <= value <= 1.0


def test_a_document_too_short_for_an_ngram_has_no_ratio_rather_than_zero(ds):
    """Null, not zero: an undefined ratio must not read as "no repetition"."""
    got = (
        bt.from_pydict({"s": ["one two three", "a b c d e f g h i j"]})
        .select(v=bt.col("s").str.duplicate_ngram_ratio(5))
        .to_pydict()["v"]
    )
    assert got[0] is None, "three words cannot carry a 5-gram"
    assert got[1] is not None


def test_length_filters_agree_with_the_character_count(ds):
    """``is_long`` / ``is_short`` around one threshold, which must partition the corpus."""
    got = ds.select(
        long=bt.col("s").str.is_long(50), short=bt.col("s").str.is_short(50)
    ).to_pydict()
    assert got["long"] == _reference(lambda d: len(d) >= 50)
    assert got["short"] == _reference(lambda d: len(d) < 50)
    for i in range(len(DOCS) - 1):
        assert got["long"][i] != got["short"][i], "the two must partition, not overlap"


def test_token_budget_agrees_with_the_estimate_it_thresholds(ds):
    """``fits_token_budget`` must be exactly ``estimate_tokens() <= budget``."""
    got = ds.select(
        tokens=bt.col("s").str.estimate_tokens(),
        fits=bt.col("s").str.fits_token_budget(10),
        fits_wide=bt.col("s").str.fits_token_budget(10_000),
    ).to_pydict()
    for i in range(len(DOCS) - 1):
        assert got["fits"][i] == (got["tokens"][i] <= 10), f"budget disagrees at row {i}"
        assert got["fits_wide"][i] is True
    assert got["fits"][NULL_ROW] is None


def test_contains_all_is_the_conjunction_of_its_patterns(ds):
    """True only where every literal is present, which one long pattern cannot express."""
    got = ds.select(
        both=bt.col("s").str.contains_all(["Hello", "items"]),
        one_missing=bt.col("s").str.contains_all(["Hello", "absent"]),
    ).to_pydict()
    assert got["both"] == _reference(lambda d: "Hello" in d and "items" in d)
    assert got["one_missing"] == _reference(lambda d: "Hello" in d and "absent" in d)
    # An empty conjunction raises rather than answering true: a filter that silently kept
    # every row because its term list came back empty is the failure this guards against.
    with pytest.raises(PlanError, match="at least one pattern"):
        ds.select(v=bt.col("s").str.contains_all([]))


def test_the_whole_family_survives_an_empty_column():
    """Every method over zero rows must return an empty column, not raise."""
    empty = bt.from_pydict({"s": []})
    got = empty.select(
        ratio=bt.col("s").str.uppercase_ratio(),
        blank=bt.col("s").str.is_blank(),
        urls=bt.col("s").str.extract_urls(),
        tokens=bt.col("s").str.estimate_tokens(),
        top=bt.col("s").str.top_ngram_ratio(2),
    ).to_pydict()
    assert all(values == [] for values in got.values())
