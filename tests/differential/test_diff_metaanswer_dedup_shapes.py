"""Metadata scalar terminals must decline over row-dropping / dedup shapes — and equal DuckDB.

A whole-relation source statistic (a column's footer min/max, null count, or distinct
count) describes *every* row. Three plan shapes drop or collapse rows in a way that
statistic cannot see, so a metadata answer read straight from it would disagree with
executing the query:

- ``distinct(subset)`` and ``QUALIFY <rank> <= k`` keep only the top row(s) per partition
  (they lower to `Filter(<rank> <= k)` over a ranking `Window`). A *non-key* column's
  whole-relation min/max/null-count then describes rows the result no longer contains —
  e.g. ``min(x)`` over ``distinct(["g"], keep="first", order_by="y")`` reading the global
  minimum instead of the surviving rows' minimum.
- ``UNION`` (``union(distinct=True)``) deduplicates across branches, folding repeated null
  rows into one — yet the source statistics *sum* each branch's null count, so a metadata
  ``n_null`` over-reports (two branches with one null each give 2, the deduplicated result
  holds 1).

The fast path must decline (return ``None``) for exactly these, and the executed answer
must equal DuckDB. Regression cases assert the shortcut still *fires* where dedup preserves
the statistic (a value-set-preserving ``UNION``'s min; a row-preserving ranking window).

Pins the fix in `api/terminal/metadata_answer/_core.py` (`_has_rank_reduction`,
`_dedups_nulls`).
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from batcher.api.terminal.metadata_answer import (
    metadata_max,
    metadata_min,
    metadata_null_count,
)
from batcher.plan.expr_ir.nodes import row_number

pytestmark = pytest.mark.differential

# A key column `g` with repeats, a non-key `x` whose extremes fall on non-surviving rows,
# an order column `y`, and two nulls in `x` (so dedup can collapse them).
_TABLE = pa.table(
    {
        "g": pa.array([1, 1, 2, 2, 3, 3], pa.int64()),
        "x": pa.array([100, 1, 200, 2, None, None], pa.int64()),
        "y": pa.array([1, 2, 3, 4, 5, 6], pa.int64()),
    }
)


@pytest.fixture
def pq_path(tmp_path):
    path = str(tmp_path / "t.parquet")
    pq.write_table(_TABLE, path, row_group_size=2)  # multiple row groups
    return path


def _sources(pq_path):
    return {"parquet": bt.read.parquet(pq_path), "memory": bt.from_arrow(_TABLE)}


# ---------------------------------------------------------------------------------------
# distinct(subset): a per-partition top-N. min/max/null_count of a NON-key column decline.
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("keep", ["first", "last"])
def test_distinct_subset_scalar_declines_and_matches_duckdb(pq_path, duck, keep):
    duck.register("t", _TABLE)
    # DuckDB's equivalent of distinct(["g"], keep, order_by=y): row_number() partitioned by g
    # ordered by y (ascending for `first`, descending for `last`), keeping rank 1.
    direction = "DESC" if keep == "last" else "ASC"
    sub = (
        f"SELECT * FROM (SELECT *, row_number() OVER (PARTITION BY g ORDER BY y {direction}) rn "
        f"FROM t) WHERE rn = 1"
    )
    dmin, dmax, dnull, dnun = duck.execute(
        f"SELECT min(x), max(x), count(*) - count(x), count(DISTINCT x) FROM ({sub})"
    ).fetchone()

    for ds in _sources(pq_path).values():
        d = ds.distinct(["g"], keep=keep, order_by="y")
        # The fast path must DECLINE — a non-key column's whole-relation stat is not the answer.
        assert metadata_min(d._plan, d._sources, "x") is None
        assert metadata_max(d._plan, d._sources, "x") is None
        assert metadata_null_count(d._plan, d._sources, "x") is None
        # ... and the executed terminal equals DuckDB.
        assert d.min("x") == dmin
        assert d.max("x") == dmax
        assert d.n_null("x") == dnull
        assert d.n_unique("x") == dnun


def test_distinct_subset_keep_any_scalar_declines(pq_path):
    """`keep="any"` picks an arbitrary survivor per key, so the whole-relation stat is still
    wrong; the fast path declines (the picked value is non-deterministic, so no value oracle)."""
    for ds in _sources(pq_path).values():
        d = ds.distinct(["g"])
        assert metadata_min(d._plan, d._sources, "x") is None
        assert metadata_max(d._plan, d._sources, "x") is None
        assert metadata_null_count(d._plan, d._sources, "x") is None


def test_qualify_pattern_scalar_declines(pq_path):
    """The un-fused shape — a `Filter` on a ranking `Window`'s output — declines too."""
    for ds in _sources(pq_path).values():
        q = ds.with_columns(row_number().over(partition_by="g", order_by="y").alias("rn")).filter(
            bt.col("rn") <= 1
        )
        assert metadata_min(q._plan, q._sources, "x") is None
        assert metadata_max(q._plan, q._sources, "x") is None


def test_row_preserving_ranking_window_still_answers(pq_path):
    """A ranking window that is *not* filtered preserves every row — the shortcut still fires."""
    for ds in _sources(pq_path).values():
        w = ds.with_columns(row_number().over(partition_by="g", order_by="y").alias("rn"))
        # min(x) over a row-preserving window is the whole-relation min — answerable.
        assert metadata_min(w._plan, w._sources, "x") == 1
        assert w.min("x") == 1


# ---------------------------------------------------------------------------------------
# union(distinct=True): dedup collapses repeated null rows. n_null declines; min still fires.
# ---------------------------------------------------------------------------------------
def test_union_distinct_null_count_declines_and_matches_duckdb(pq_path, duck):
    duck.register("t", _TABLE)
    dnull = duck.execute(
        "SELECT count(*) - count(x) FROM (SELECT * FROM t UNION SELECT * FROM t)"
    ).fetchone()[0]
    for ds in _sources(pq_path).values():
        u = ds.union(ds, distinct=True)
        # Summed source null count over-reports under dedup — must decline.
        assert metadata_null_count(u._plan, u._sources, "x") is None
        assert u.n_null("x") == dnull  # executed answer equals DuckDB


def test_union_all_null_count_still_answers(pq_path):
    """`UNION ALL` sums null counts correctly, so the shortcut still fires (no over-count)."""
    # _TABLE has two nulls in x; UNION ALL of two copies has four.
    for ds in _sources(pq_path).values():
        u = ds.union(ds)  # distinct=False
        assert metadata_null_count(u._plan, u._sources, "x") == 4
        assert u.n_null("x") == 4


def test_union_distinct_min_still_answers(pq_path):
    """Dedup preserves the value *set*, so min/max/n_unique over `UNION` stay answerable."""
    for ds in _sources(pq_path).values():
        u = ds.union(ds, distinct=True)
        assert metadata_min(u._plan, u._sources, "x") == 1  # fired, no scan
        assert u.min("x") == 1
