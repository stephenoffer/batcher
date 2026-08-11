"""Column pruning under an outer explode must not discard the rows `outer` exists to keep.

`rewrite_projection` rebuilt an `Unnest` positionally — ``Unnest(child, column, alias)`` —
and the node has **five** fields. So whenever pruning rewrote the child of an explode,
`outer` fell back to its default (False) and `index_alias` to None, both silently:

    ds.with_columns(x=col("xs")).explode("x", outer=True).select("id", "x")

returned only the rows whose list was non-empty. The row kept for an empty list, and the
row kept for a NULL list, were gone — the entire point of ``outer``. Nothing raised, and
the *unoptimized* plan was correct, so the defect was visible only after optimization.

It reaches users through both front ends. The DataFrame spelling is above; the SQL one is
``SELECT id, x FROM t LEFT JOIN UNNEST(t.xs) AS u(x) ON TRUE``, which is how SQL asks for
an outer unnest.

Why no existing test caught it: pruning only rewrites the child when some column is
actually dropped, and every explode test projected everything it produced. The trigger is
a *projection* over an outer explode, which is what these add.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same
from batcher.plan.expr_ir import col

pytestmark = pytest.mark.differential

#: Both list edges that only `outer` keeps: an empty list and a NULL one.
_ROWS = {"id": [1, 2, 3, 4], "xs": [[10, 20], [30], [], None]}


def _t() -> pa.Table:
    return pa.table(
        {
            "id": pa.array(_ROWS["id"], pa.int64()),
            "xs": pa.array(_ROWS["xs"], pa.list_(pa.int64())),
        }
    )


def test_outer_explode_keeps_its_rows_under_a_projection():
    ds = bt.from_pydict(_ROWS)
    full = ds.explode("xs", alias="x", outer=True).collect().num_rows
    pruned = ds.with_columns(x=col("xs")).explode("x", outer=True).select("id", "x")
    # Two elements from the first list, one from the second, and one row each for the
    # empty and the NULL list — the last two being exactly what `outer` adds.
    assert pruned.collect().num_rows == full == 5
    assert pruned.collect().to_pydict() == {"id": [1, 1, 2, 3, 4], "x": [10, 20, 30, None, None]}


def test_dropping_the_source_column_keeps_the_outer_rows():
    """`drop` prunes through the same path `select` does."""
    ds = bt.from_pydict(_ROWS)
    got = ds.with_columns(x=col("xs")).explode("x", outer=True).drop("xs").collect()
    assert got.to_pydict() == {"id": [1, 1, 2, 3, 4], "x": [10, 20, 30, None, None]}


def test_position_index_survives_pruning():
    """`index_alias` was dropped by the same rebuild, so the position column vanished."""
    ds = bt.from_pydict(_ROWS)
    got = ds.explode("xs", alias="x", outer=True, index="i").select("id", "x", "i").collect()
    assert got.column_names == ["id", "x", "i"]
    assert got.to_pydict()["i"] == [0, 1, 0, None, None]


def test_sql_outer_unnest_with_a_projection_matches_duckdb(duck):
    duck.register("t", _t())
    query = "SELECT id, x FROM t LEFT JOIN UNNEST(t.xs) AS u(x) ON TRUE"
    assert_same(bt.sql(query, t=bt.from_arrow(_t())).collect(), duck.sql(query))


def test_sql_outer_unnest_with_ordinality_and_a_projection_matches_duckdb(duck):
    duck.register("t", _t())
    query = "SELECT id, x, i FROM t LEFT JOIN UNNEST(t.xs) WITH ORDINALITY AS u(x, i) ON TRUE"
    assert_same(bt.sql(query, t=bt.from_arrow(_t())).collect(), duck.sql(query))


def test_non_outer_explode_still_drops_empty_and_null_lists():
    """The default must not drift the other way while fixing the outer case."""
    ds = bt.from_pydict(_ROWS)
    got = ds.with_columns(x=col("xs")).explode("x").select("id", "x").collect()
    assert got.to_pydict() == {"id": [1, 1, 2], "x": [10, 20, 30]}
