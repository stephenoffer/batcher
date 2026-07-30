"""`list.gather` against DuckDB's `list_select`, and the rerank the two build together.

DuckDB spells the same operation `list_select(values, indices)`, differing only in that its
indices are 1-based. Checking against it covers the ordinary path; the null and out-of-range
edges are pinned in the Rust unit tests, where DuckDB's own behaviour differs.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

pytestmark = pytest.mark.differential

_ROWS = pa.table(
    {
        "id": [1, 2, 3],
        "docs": [["a", "b", "c"], ["p", "q"], ["solo"]],
        "picks": [[2, 0], [1, 0], [0]],  # 0-based
        "scores": [[0.1, 0.9, 0.5], [0.7, 0.2], [1.0]],
    }
)


def test_gather_matches_duckdb_list_select(duck):
    duck.register("rows", _ROWS)
    # DuckDB's `list_select` is 1-based, so the same positions shift by one.
    expected = duck.sql(
        """
        SELECT id, list_select(docs, [i + 1 FOR i IN picks]) AS taken
        FROM rows
        """
    )
    got = (
        bt.from_arrow(_ROWS)
        .select("id", taken=bt.col("docs").list.gather(bt.col("picks")))
        .collect()
    )
    assert_same(got, expected)


def test_a_score_ordered_rerank_matches_duckdb(duck):
    """The end-to-end shape: rank by one column, take from another, cut at k."""
    duck.register("rows", _ROWS)
    expected = duck.sql(
        """
        SELECT id, list_select(
                     docs,
                     list_slice(
                       list_reverse(list_grade_up(scores)),
                       1, 2
                     )
                   ) AS top2
        FROM rows
        """
    )
    best_first = bt.col("scores").list.arg_sort().list.reverse()
    got = (
        bt.from_arrow(_ROWS)
        .select("id", top2=bt.col("docs").list.gather(best_first.list.head(2)))
        .collect()
    )
    assert_same(got, expected)
