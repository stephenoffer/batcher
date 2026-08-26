"""A library aggregate has to be recognized before the query is built, not after.

`plan.functions` exports ~200 aggregates -- `all_caps_rate`, `pii_rate`, `cohen_kappa`,
`geometric_mean` -- and every one of them raised "unknown function" in SQL while answering
through `bt.<name>(...)`. Wiring them through the *scalar* path was not enough and was worse
than nothing: an aggregate reached that way answered correctly with no grouping and then
failed under `GROUP BY` with "window function '__bt_win_0' references unknown column",
because aggregate *collection* never saw it and built the GROUP BY without it. Right in the
shape nobody writes, broken in the shape everybody does.

They are classified by name now, in `is_agg_node`, which is where collection asks. Most are
composite -- expressions over aggregate leaves, the way `sem` and the `regr_*` family are --
and `GroupBy.agg` already hoists those into hidden columns, so the machinery was there.

Both the grouped and the ungrouped shape are asserted for each, because passing either one
alone is exactly how the half-working version looked correct.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_tables_equal
from batcher import col

pytestmark = pytest.mark.differential


@pytest.fixture
def library_table() -> pa.Table:
    return pa.table(
        {
            "g": pa.array(["a", "a", "b", "b"]),
            "f": pa.array([1.0, 2.0, 3.0, 4.0]),
            "s": pa.array(["Alpha beta", "x y", "GAMMA d", "e f"]),
        }
    )


@pytest.mark.parametrize("name", ["all_caps_rate", "pii_rate", "emoji_rate", "refusal_rate"])
def test_a_library_aggregate_groups_as_well_as_it_reduces(library_table, name):
    """An aggregate has to be recognized *before* the query is built, not after.

    Reached through the scalar path it answered correctly with no grouping and then failed
    under `GROUP BY` with "window function '__bt_win_0' references unknown column" — right
    in the shape nobody writes, broken in the shape everybody does. It is classified by
    name now, so aggregate *collection* sees it and builds the GROUP BY around it.

    Both shapes are asserted, because passing either one alone is exactly how the half-
    working version looked correct.
    """
    ds = bt.from_arrow(library_table)
    builder = getattr(bt, name)
    whole = bt.sql(f"SELECT {name}(s) AS v FROM t", t=library_table).to_pydict()
    assert whole == ds.agg(v=builder(col("s"))).to_pydict()

    grouped = bt.sql(f"SELECT g, {name}(s) AS v FROM t GROUP BY g", t=library_table)
    assert_tables_equal(grouped.collect(), ds.group_by("g").agg(v=builder(col("s"))).collect())


def test_a_library_aggregate_is_refused_under_distinct(library_table):
    """`DISTINCT` de-duplicates one input; a composite aggregate reduces several columns.

    There is no single column to de-duplicate, so it is refused by name rather than
    silently ignored — the rule `sem` and the `regr_*` family already follow.
    """
    with pytest.raises(NotImplementedError, match="DISTINCT"):
        bt.sql("SELECT all_caps_rate(DISTINCT s) AS v FROM t", t=library_table).collect()
