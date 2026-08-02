"""The regular-expression functions on the GPU translator, and the patterns it refuses.

A regex is the one construct in the translator that cannot be checked by construction: the
engine compiles Rust's, the verification backend Python's, and the device cuDF's. The family
was therefore declined whole — which quietly cost a third of the string surface, because
`word_count`, `digit_ratio`, `has_html`, `slugify` and thirty more lower to one of these calls.

`vocab.regex.portable` classifies the *pattern* instead. This module is both halves of that
claim: the accepted patterns match the engine exactly, and the rejected ones are rejected —
because a classifier that lets one dialect-sensitive construct through is worse than no
classifier at all.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col
from batcher.core.gpu_plan import DfBackend, gpu_plan_ops
from batcher.core.gpu_plan.backend import Unsupported
from batcher.core.gpu_plan.execute import run_chain
from batcher.core.gpu_plan.vocab.regex import (
    RESTRICTED_ALPHABET,
    portable,
    shorthand_classes,
)

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def be():
    import pandas as pd

    return DfBackend(pd)


TEXT = pa.table(
    {
        "s": pa.array(
            [
                "ab12cd34",  # several matches
                "xyz",  # none
                "",  # empty, which is not the same as no match
                None,  # missing
                "12",  # a whole-string match
                "a.b",  # a metacharacter as data
                "<p>hi</p> 7",  # markup and a digit
                "Ünïcødé 9",  # non-ASCII, where the shorthand classes diverge
            ],
            pa.string(),
        )
    }
)


def _rows(table: pa.Table) -> list[tuple]:
    return [tuple(r) for r in zip(*table.to_pydict().values(), strict=True)]


def _assert_matches_engine(ds, table: pa.Table, be) -> None:
    spec = gpu_plan_ops(ds._plan)
    assert spec is not None, "the translator declined a plan it is supposed to match"
    expected = ds.collect()
    got = be.to_arrow(run_chain(table, spec[1], be)).select(expected.column_names)
    assert _rows(got) == _rows(expected)
    assert got.schema.types == expected.schema.types


def _declines(ds, table: pa.Table, be) -> None:
    spec = gpu_plan_ops(ds._plan)
    assert spec is not None
    with pytest.raises(Unsupported):
        run_chain(table, spec[1], be)


# --- the classifier ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pattern",
    [
        "[0-9]", "[A-Za-z]", "[A-Za-z0-9]", "[()]", "[.!?]", "[^a-z0-9]+", "<[^>]+>",
        r"[^\x00-\x7F]", "```", r"\n", r"\t", " ", '"', "a|b", "(?:ab)+", "x{2,}", ".",
        r"a\.b", "^abc",
    ],
)  # fmt: skip
def test_a_portable_pattern_is_accepted(pattern):
    """Literals, explicit classes, quantifiers, alternation and non-capturing groups mean the
    same thing in all three engines."""
    assert portable(pattern)


@pytest.mark.parametrize(
    "pattern",
    [
        # A `$` anywhere but the end is a literal in a class or an anchor that cannot match.
        "a$b", "^$x",
        # Nobody implements all three of these the same way, and cuDF implements none.
        r"(?=foo)", r"(?i)abc", r"(a)\1", r"\p{Alpha}", "[[:alpha:]]",
    ],
)  # fmt: skip
def test_a_dialect_sensitive_pattern_is_rejected(pattern):
    """A classifier that lets one of these through is worse than no classifier: it would agree
    with the engine on every ASCII test and disagree on the first accented letter."""
    assert not portable(pattern)


@pytest.mark.parametrize("pattern", ["^[0-9]+$", "abc$"])
def test_a_trailing_end_anchor_is_accepted_but_gated_on_the_data(pattern):
    """`$` is the one construct handled in two halves rather than refused outright.

    The engines disagree about it only over text that *ends in a newline* — Python's matches
    before it, Rust's does not — and that is a property of the data, not of the pattern. So the
    scanner accepts a trailing `$` and `_check_end_anchor` declines at execution time when the
    column actually contains one, which keeps `^[0-9]+$` on the device for the overwhelming
    majority of real text instead of refusing every anchored pattern there is.
    """
    assert portable(pattern)


@pytest.mark.parametrize("pattern", [r"\w+", r"\s", r"\S+", r"\d", r"\W", r"\D", r"\b\w+\b"])
def test_a_shorthand_class_is_accepted_but_gated_on_the_data(pattern):
    """The shorthand classes are portable over a *restricted alphabet* and not in general, so
    they are accepted here and the column is checked against that alphabet before they run."""
    assert portable(pattern)
    assert shorthand_classes(pattern)


@pytest.mark.parametrize("pattern", ["[0-9]", r"a\.b", "(?:ab)+", r"[^\x00-\x7F]"])
def test_a_pattern_without_a_shorthand_class_needs_no_data_check(pattern):
    assert not shorthand_classes(pattern)


def test_the_alphabet_probe_is_itself_a_portable_pattern():
    """The gate is applied with a regex, so it has to be one both backends read the same way —
    otherwise the check would need a check."""
    assert portable(RESTRICTED_ALPHABET)
    assert not shorthand_classes(RESTRICTED_ALPHABET)


def test_an_unfinished_escape_is_rejected():
    assert not portable("abc\\")


# --- the accepted patterns match the engine -------------------------------------------------


PATTERNS = ["[0-9]", "[A-Za-z]", "<[^>]+>", r"[^\x00-\x7F]", "[.!?]", "a|b", "(?:ab)+", "."]


@pytest.mark.parametrize("pattern", PATTERNS)
def test_counting_matches(be, pattern):
    """Non-overlapping matches, and null over a null input rather than zero."""
    ds = bt.from_arrow(TEXT).select(out=col("s").str.regexp_count(pattern))
    _assert_matches_engine(ds, TEXT, be)


@pytest.mark.parametrize("pattern", PATTERNS)
def test_testing_for_a_match(be, pattern):
    """`regexp_matches` is a *search* — true when the pattern occurs anywhere."""
    ds = bt.from_arrow(TEXT).select(out=col("s").str.regexp_matches(pattern))
    _assert_matches_engine(ds, TEXT, be)


@pytest.mark.parametrize("pattern", PATTERNS)
def test_replacing_every_match(be, pattern):
    ds = bt.from_arrow(TEXT).select(out=col("s").str.regexp_replace_all(pattern, "Z"))
    _assert_matches_engine(ds, TEXT, be)


@pytest.mark.parametrize("pattern", PATTERNS)
def test_replacing_only_the_first_match(be, pattern):
    """The two forms differ by one keyword, and swapping them changes a value rather than
    raising."""
    ds = bt.from_arrow(TEXT).select(out=col("s").str.regexp_replace(pattern, "Z"))
    _assert_matches_engine(ds, TEXT, be)


def test_extracting_a_group_declines(be):
    """pandas' Arrow-backed `extract` accepts only *named* capture groups and cuDF's accepts
    only unnamed ones, so the verification backend cannot run the pattern the device would."""
    ds = bt.from_arrow(TEXT).select(out=col("s").str.regexp_extract("([0-9]+)"))
    _declines(ds, TEXT, be)


# --- the text functions built on them --------------------------------------------------------


@pytest.mark.parametrize(
    "fn",
    ["digit_count", "space_count", "tab_count", "newline_count", "quote_count", "paren_count",
     "sentence_count", "line_count", "non_ascii_count", "code_fence_count", "alnum_ratio",
     "alpha_ratio", "digit_ratio", "has_digits", "has_html", "remove_html_tags",
     "remove_digits", "slugify"],
)  # fmt: skip
def test_a_text_function_built_on_a_portable_pattern_reaches_the_device(be, fn):
    """These are the ones whose whole cost was a declined regex — none of them is a regex to
    the user, and every one of them sent its chain to the host."""
    ds = bt.from_arrow(TEXT).select(out=getattr(col("s").str, fn)())
    _assert_matches_engine(ds, TEXT, be)


# --- the shorthand classes, over the alphabet they are portable on ----------------------------

#: The same text as `TEXT`, without the non-ASCII row — which is the whole difference between
#: a shorthand-class pattern the three engines agree on and one they do not.
ASCII_TEXT = pa.table(
    {
        "s": pa.array(
            [
                "Hello World! foo@bar.com",
                "a  b\tc",
                "",
                None,
                "one two three four five",
                "https://x.io/p?q=1",
                "#tag @who 123-456-7890",
            ],
            pa.string(),
        )
    }
)


@pytest.mark.parametrize(
    "fn",
    ["word_count", "whitespace_ratio", "punctuation_ratio", "email_count", "url_count",
     "hashtag_count", "mention_count", "phone_count", "long_word_count",
     "uppercase_word_count", "paragraph_count", "remove_punctuation", "normalize_whitespace",
     "remove_emails", "remove_urls", "has_email", "has_url", "symbol_to_word_ratio",
     "digit_to_word_ratio", "avg_word_length", "avg_sentence_length", "mean_line_length",
     "estimate_tokens", "word_char_ratio"],
)  # fmt: skip
def test_a_shorthand_class_text_function_over_ascii_matches_the_engine(be, fn):
    """Every one of these is a `\\w` or a `\\s` underneath, and every one of them ran on the
    host because the family was declined by its spelling rather than by its data."""
    ds = bt.from_arrow(ASCII_TEXT).select(out=getattr(col("s").str, fn)())
    _assert_matches_engine(ds, ASCII_TEXT, be)


@pytest.mark.parametrize("fn", ["word_count", "whitespace_ratio", "punctuation_ratio"])
def test_the_same_function_declines_over_non_ascii(be, fn):
    """`\\w` is Unicode in the engine and ASCII on the device, so the accented row is exactly
    where they part company — and it is the *data* that decides, not the spelling."""
    ds = bt.from_arrow(TEXT).select(out=getattr(col("s").str, fn)())
    _declines(ds, TEXT, be)


def test_a_vertical_tab_is_enough_to_decline(be):
    """Not only about non-ASCII: `\\s` matches a vertical tab in the engine and not in the host
    backend's RE2, so one control character makes the same pattern count differently."""
    table = pa.table({"s": pa.array(["a\x0bb"], pa.string())})
    ds = bt.from_arrow(table).select(out=col("s").str.word_count())
    _declines(ds, table, be)


def test_a_replacement_carrying_a_group_reference_declines(be):
    """`$1`, `\\1` and a dedicated call — three engines, three spellings."""
    ds = bt.from_arrow(TEXT).select(out=col("s").str.regexp_replace_all("([0-9])", r"\1\1"))
    _declines(ds, TEXT, be)
