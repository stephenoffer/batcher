"""Differential tests for the per-document text-quality measures, against DuckDB.

Two jobs here. The first is `word_count`, which is not a new function: it already existed as
``regexp_count(r"\\S+")`` and now runs as a native single-pass scan. A reimplementation of a
live function is the riskiest change in this area, so it is checked against DuckDB's own
regex count — the same definition the old implementation used — over every shape that could
separate them.

The second is the new Gopher measures. DuckDB has no notion of document quality, so the
oracle is its general string vocabulary spelling out each definition independently. Where a
definition has no SQL spelling (character entropy, the n-gram ratios) the check lives in
``tests/unit/test_text_quality.py`` against worked ground truth instead.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

pytestmark = pytest.mark.differential

#: DuckDB's spelling of "how many words", reused by the oracles that divide by it.
WC = "len(list_filter(str_split_regex(s, '\\s+'), x -> x != ''))"

# Every shape that could separate a whitespace split from a regex: runs of spaces, tabs and
# newlines, leading and trailing whitespace, an all-blank string, an empty string, and null.
_DOCS = [
    "hello big  world",
    "  leading and trailing  ",
    "tabs\tand\nnewlines\r\nmixed",
    "single",
    "   ",
    "",
    None,
]


def _tbl(docs: list[str | None]) -> pa.Table:
    return pa.table({"s": pa.array(docs, pa.string()), "row": list(range(len(docs)))})


def test_word_count_still_matches_the_regex_it_replaced(duck):
    """The native scan == a regex split on whitespace, which is what this function used to be.

    DuckDB has no `regexp_count`, so the oracle splits and filters instead — the same
    definition, reached by a different route, which is what makes it an oracle rather than a
    restatement.
    """
    t = _tbl(_DOCS)
    out = bt.from_arrow(t).select(v=bt.col("s").str.word_count(), row=bt.col("row")).collect()
    duck.register("t", t)
    assert_same(
        out,
        duck.sql(
            "SELECT len(list_filter(regexp_split_to_array(s, '\\s+'), x -> x != '')) AS v,"
            " row FROM t"
        ),
    )


def test_word_count_matches_a_split_and_measure(duck):
    """A second, independent spelling of the same definition, to catch a shared regex quirk."""
    t = _tbl(_DOCS)
    out = bt.from_arrow(t).select(v=bt.col("s").str.word_count(), row=bt.col("row")).collect()
    duck.register("t", t)
    assert_same(
        out,
        duck.sql(
            "SELECT len(list_filter(str_split_regex(s, '\\s+'), x -> x != '')) AS v, row FROM t"
        ),
    )


def test_mean_word_length_matches_characters_over_words(duck):
    """The definition written out: non-whitespace characters divided by the word count."""
    t = _tbl(_DOCS)
    out = bt.from_arrow(t).select(v=bt.col("s").str.mean_word_length(), row=bt.col("row")).collect()
    duck.register("t", t)
    assert_same(
        out,
        duck.sql(
            "SELECT len(regexp_replace(s, '\\s', '', 'g')) * 1.0 "
            f"/ nullif({WC}, 0) AS v, row FROM t"
        ),
    )


def test_symbol_ratio_matches_a_delete_and_measure(duck):
    """`(# + …) / words`, counting both spellings of an ellipsis."""
    t = _tbl(["a... b... c... d", "x # y # z", "no symbols here", "", None])
    out = bt.from_arrow(t).select(v=bt.col("s").str.symbol_ratio(), row=bt.col("row")).collect()
    duck.register("t", t)
    assert_same(
        out,
        duck.sql(
            "SELECT ((len(s) - len(replace(s, '#', '')))"
            "      + (len(s) - len(replace(s, '...', ''))) / 3"
            "      + (len(s) - len(replace(s, '…', '')))) * 1.0"
            f" / nullif({WC}, 0) AS v, row FROM t"
        ),
    )


def test_alpha_word_ratio_matches_a_filtered_word_list(duck):
    """The fraction of words containing a letter, counted by filtering the split."""
    t = _tbl(["the quick brown fox", "1.99 2.50 3.75 four", "123 456", "", None])
    out = bt.from_arrow(t).select(v=bt.col("s").str.alpha_word_ratio(), row=bt.col("row")).collect()
    duck.register("t", t)
    assert_same(
        out,
        duck.sql(
            "SELECT len(list_filter(w, x -> regexp_matches(x, '[a-zA-Z]'))) * 1.0 "
            "/ nullif(len(w), 0) AS v, row FROM ("
            "  SELECT list_filter(str_split_regex(s, '\\s+'), x -> x != '') AS w, row FROM t)"
        ),
    )


def test_bullet_and_ellipsis_line_ratios_match_a_filtered_line_list(duck):
    """Each is a count over the non-empty lines, which DuckDB can split and filter."""
    t = _tbl(["- a\n- b\nprose", "one...\ntwo…\nthree", "plain", "", None])
    ds = bt.from_arrow(t)
    duck.register("t", t)
    bullets = ds.select(v=bt.col("s").str.bullet_line_ratio(), row=bt.col("row")).collect()
    assert_same(
        bullets,
        duck.sql(
            "SELECT len(list_filter(l, x -> regexp_matches("
            "  x, '^[-*\\x{2022}\\x{2023}\\x{2043}\\x{2219}]'))) * 1.0 "
            "/ nullif(len(l), 0) AS v, row FROM ("
            "  SELECT list_filter(list_transform(str_split(s, chr(10)), x -> trim(x)),"
            "                     x -> x != '') AS l, row FROM t)"
        ),
    )
    ellipses = ds.select(v=bt.col("s").str.ellipsis_line_ratio(), row=bt.col("row")).collect()
    assert_same(
        ellipses,
        duck.sql(
            "SELECT len(list_filter(l, x -> x LIKE '%...' OR x LIKE '%…')) * 1.0 "
            "/ nullif(len(l), 0) AS v, row FROM ("
            "  SELECT list_filter(list_transform(str_split(s, chr(10)), x -> trim(x)),"
            "                     x -> x != '') AS l, row FROM t)"
        ),
    )


def test_stopword_count_matches_a_distinct_membership_count(duck):
    """Distinct stop words present, not occurrences — so a repeated word counts once."""
    t = _tbl(["the the the", "the cat sat with a hat", "(The) and,", "nothing", "", None])
    out = bt.from_arrow(t).select(v=bt.col("s").str.stopword_count(), row=bt.col("row")).collect()
    duck.register("t", t)
    assert_same(
        out,
        duck.sql(
            "SELECT CASE WHEN s IS NULL THEN NULL ELSE len(list_filter("
            "  ['the','be','to','of','and','that','have','with'],"
            "  sw -> list_contains(list_distinct(list_transform("
            "     list_filter(str_split_regex(lower(s), '\\s+'), x -> x != ''),"
            "     x -> regexp_replace(x, '^[^a-z0-9]+|[^a-z0-9]+$', '', 'g'))), sw))) END"
            " AS v, row FROM t"
        ),
    )


def test_the_quality_measures_survive_the_execution_schedulings(duck):
    """`collect()`, `collect(spill=True)` and `iter_batches()` agree, past one morsel.

    Every measure here is a per-row scalar, so no scheduling can change it — which is why
    this is worth pinning: a kernel that accumulated across rows instead of within one would
    pass every small test and split its answers at a morsel boundary.
    """
    from _harness import assert_tables_equal

    t = _tbl(_DOCS * 4000)
    ds = bt.from_arrow(t).select(
        w=bt.col("s").str.word_count(),
        m=bt.col("s").str.mean_word_length(),
        e=bt.col("s").str.char_entropy(),
        row=bt.col("row"),
    )
    single = ds.collect()
    assert_tables_equal(ds.collect(spill=True), single)
    streamed = pa.Table.from_batches(list(ds.iter_batches()), schema=single.schema)
    assert_tables_equal(streamed, single)
