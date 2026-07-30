"""`str.token_ngrams` and `list.multiset_overlap` against DuckDB.

Both primitives have a DuckDB spelling — n-grams as a windowed split, the clipped overlap as
a join of two per-value counts taking the lesser — so the engine's answer is checked against
an oracle rather than against a hand-written expectation. That matters most for the clip,
which is the one behaviour a set intersection gets wrong.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

pytestmark = pytest.mark.differential

# Rows chosen to cover the edges the clip cares about: repeats on one side, repeats on both,
# disjoint values, an empty list, and a single shared value.
_PAIRS = pa.table(
    {
        "id": [1, 2, 3, 4, 5, 6],
        "a": [
            ["the", "the", "cat"],
            ["a", "a", "a"],
            ["x", "y", "z"],
            [],
            ["a", "b", "a", "b"],
            ["solo"],
        ],
        "b": [
            ["the", "cat"],
            ["a", "a"],
            ["p", "q"],
            ["a"],
            ["b", "a", "b", "a"],
            ["solo"],
        ],
    }
)

_DOCS = pa.table(
    {
        "id": [1, 2, 3, 4],
        "txt": [
            "the quick brown fox jumps",
            "one two",
            "single",
            "  padded   spacing  here ",
        ],
    }
)


def test_multiset_overlap_matches_duckdb_clipped_counts(duck):
    duck.register("pairs", _PAIRS)
    expected = duck.sql(
        """
        SELECT id, (
          SELECT coalesce(sum(least(l.c, r.c)), 0)
          FROM (SELECT x, count(*) c FROM unnest(t.a) AS u(x) GROUP BY x) l
          JOIN (SELECT x, count(*) c FROM unnest(t.b) AS u(x) GROUP BY x) r ON l.x = r.x
        )::DOUBLE AS overlap
        FROM pairs t
        """
    )
    got = (
        bt.from_arrow(_PAIRS)
        .select("id", overlap=bt.col("a").list.multiset_overlap(bt.col("b")))
        .collect()
    )
    assert_same(got, expected)


def test_multiset_overlap_never_exceeds_either_list_length():
    """The bound a clipped intersection must respect, on the same rows."""
    got = (
        bt.from_arrow(_PAIRS)
        .select(
            o=bt.col("a").list.multiset_overlap(bt.col("b")),
            bound=bt.least(bt.col("a").list.len(), bt.col("b").list.len()),
        )
        .to_pydict()
    )
    assert all(o <= b for o, b in zip(got["o"], got["bound"], strict=True))


def test_token_ngrams_matches_a_duckdb_windowed_split(duck):
    duck.register("docs", _DOCS)
    # DuckDB's spelling: split on whitespace, drop the empty parts a run of spaces leaves,
    # then join each 2-token window. A document with fewer than two tokens still yields one
    # gram, which `greatest(len(t) - 1, 1)` reproduces.
    expected = duck.sql(
        r"""
        WITH toks AS (
          SELECT id, list_filter(regexp_split_to_array(trim(txt), '\s+'), x -> x <> '') AS t
          FROM docs
        )
        SELECT id, list(gram ORDER BY i) AS grams FROM (
          SELECT id, i, array_to_string(t[i : i + 1], ' ') AS gram
          FROM toks, generate_series(1, greatest(len(t) - 1, 1)) AS g(i)
        ) GROUP BY id
        """
    )
    got = bt.from_arrow(_DOCS).select("id", grams=bt.col("txt").str.token_ngrams(2)).collect()
    assert_same(got, expected)


def test_unigram_precision_matches_a_duckdb_clipped_ratio(duck):
    """The whole metric, end to end: `ngram_precision` against DuckDB's own clipped count."""
    pairs = pa.table(
        {
            "p": ["cat cat cat cat", "cat sat down", "zebra"],
            "r": ["cat sat down", "cat sat down", "cat sat down"],
        }
    )
    duck.register("gen", pairs)
    expected = duck.sql(
        r"""
        WITH toks AS (
          SELECT regexp_split_to_array(lower(p), '\s+') AS pt,
                 regexp_split_to_array(lower(r), '\s+') AS rt
          FROM gen
        ),
        scored AS (
          SELECT len(pt) AS n, (
            SELECT coalesce(sum(least(l.c, x.c)), 0)
            FROM (SELECT v, count(*) c FROM unnest(t.pt) AS u(v) GROUP BY v) l
            JOIN (SELECT v, count(*) c FROM unnest(t.rt) AS u(v) GROUP BY v) x ON l.v = x.v
          ) AS overlap
          FROM toks t
        )
        SELECT avg(CASE WHEN n > 0 THEN overlap / n ELSE 0 END)::DOUBLE AS prec FROM scored
        """
    )
    got = bt.from_arrow(pairs).agg(prec=bt.ngram_precision("p", "r")).collect()
    assert_same(got, expected)
