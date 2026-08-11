"""``GROUPING(x)`` alongside ROLLUP/CUBE/GROUPING SETS — the standard reporting shape.

Each grouping level is translated as its own GROUP BY and the levels are combined with
UNION ALL. ``GROUPING(x)`` is a per-level *constant*, so the substitution left a different
integer literal in each level — and an un-aliased select item is named after the
expression it holds, so the levels disagreed on the output name (``(0)`` against ``(1)``).
The UNION then refused them:

    PlanError: union inputs must have identical columns:
        ['g', '(0)', 'count_star()'] vs ['g', '(1)', 'count_star()']

So ``SELECT g, GROUPING(g), count(*) FROM t GROUP BY ROLLUP(g)`` — the canonical way to
tell a rollup's subtotal row from a real one — could not run. Naming the item once, before
the levels are generated, pins it across all of them.

The aliased spellings are here as well as the bare one because they took different paths:
an explicit ``AS`` already pinned the name, so only the un-aliased form failed, and a fix
that covered just the bare case would leave ``GROUPING(g) + 1`` still broken.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

pytestmark = pytest.mark.differential


def _t() -> pa.Table:
    return pa.table(
        {
            "g": pa.array(["a", "b", "a", "b"]),
            "b": pa.array([True, False, True, None]),
            "x": pa.array([1, 2, 3, 4], pa.int64()),
        }
    )


@pytest.fixture
def tables(duck):
    duck.register("t", _t())
    return {"t": bt.from_arrow(_t())}


@pytest.mark.parametrize(
    "query",
    [
        "SELECT g, GROUPING(g) AS gr, count(*) AS c FROM t GROUP BY ROLLUP(g)",
        "SELECT g, GROUPING_ID(g) AS gi, count(*) AS c FROM t GROUP BY ROLLUP(g)",
        # Two arguments: the bit vector, first argument the most significant bit.
        "SELECT g, b, GROUPING(g, b) AS gr, count(*) AS c FROM t GROUP BY CUBE(g, b)",
        "SELECT g, b, GROUPING(g) AS a, GROUPING(b) AS d, count(*) AS c FROM t GROUP BY CUBE(g, b)",
        "SELECT g, GROUPING(g) AS gi, count(*) AS c FROM t GROUP BY GROUPING SETS ((g), ())",
        # GROUPING inside a larger expression — the whole item needs the pinned name.
        "SELECT g, GROUPING(g) + 1 AS gp, count(*) AS c FROM t GROUP BY ROLLUP(g)",
        # A plain GROUP BY rolls nothing up, so GROUPING is the constant 0.
        "SELECT g, GROUPING(g) AS gr, count(*) AS c FROM t GROUP BY g",
        # Filtering on the flag is how a report drops (or keeps) the subtotal rows.
        "SELECT g, count(*) AS c FROM t GROUP BY ROLLUP(g) HAVING GROUPING(g) = 0",
    ],
)
def test_grouping_matches_duckdb(tables, duck, query):
    assert_same(bt.sql(query, **tables).collect(), duck.sql(query))


def test_unaliased_grouping_item_runs_and_keeps_one_name(tables):
    """The exact shape that raised: no AS anywhere, so every level had to agree on a name."""
    got = bt.sql("SELECT g, GROUPING(g), count(*) FROM t GROUP BY ROLLUP(g)", **tables).collect()
    assert got.num_rows == 3  # two groups plus the grand total
    assert len(got.column_names) == 3
    flags = sorted(got.column_names)
    assert any("grouping" in c for c in flags), got.column_names


def test_rollup_subtotal_row_is_marked(tables):
    """The flag has to actually distinguish the subtotal row from a group with a NULL key."""
    rows = (
        bt.sql("SELECT g, GROUPING(g) AS gr, count(*) AS c FROM t GROUP BY ROLLUP(g)", **tables)
        .collect()
        .to_pydict()
    )
    marked = [(g, c) for g, gr, c in zip(rows["g"], rows["gr"], rows["c"], strict=True) if gr == 1]
    assert marked == [(None, 4)]
