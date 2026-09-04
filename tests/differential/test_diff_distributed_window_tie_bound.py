"""How far a `row_number` tie may diverge between one node and many — and no further.

`.claude/rules/python-control-plane.md` states two exceptions to "distributed == single-node".
The second is this one: `row_number()` assigns 1..n within a window partition, and when several
rows share the ordering key, SQL does not say which of them gets which number. Single-node
resolves it by arrival within the partition; distributed resolves it by arrival through the
shuffle. The two disagree, and both are right.

A documented exception with no test is the thing that file warns about elsewhere — "an
invariant nobody can fail is one nobody reads". So what is pinned here is the **bound**, not
the divergence:

* the set of window partitions is the same on both sides;
* each partition holds the same number of rows on both sides;
* each partition's rank multiset is exactly ``1..n`` on both sides.

Everything a partitioning defect would break is therefore asserted, and only the assignment of
a rank to a *tied* row is left free. A hash that split one partition across two reducers, a
shuffle that dropped or duplicated a row, or a reducer that restarted its counter would each
fail here — which is what makes the exception a bound rather than a licence.

The companion assertion is that the freedom **disappears** once the query removes it: with an
order key that determines every row's position, the two results are identical row for row. If
that ever stops holding, the divergence is not the documented one.

The input is built so that *every* row is in a tied group, because a tie that occurs only
sometimes makes this test pass for the wrong reason on a lucky shuffle.
"""

from __future__ import annotations

import collections

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_tables_equal

pytestmark = pytest.mark.differential

pytest.importorskip("ray", reason="the distributed path needs Ray")

_ROWS = 900
_PARTITIONS = 8
#: Only 15 distinct order-key values over 900 rows in 8 partitions, so every row shares its
#: (partition, order key) with at least one other.
_ORDER_VALUES = 15
_WORKERS = 3


@pytest.fixture(scope="module")
def rows() -> pa.Table:
    return pa.table(
        {
            "k": pa.array([i % (_PARTITIONS * 7) for i in range(_ROWS)], pa.int64()),
            "o": pa.array([i % _ORDER_VALUES for i in range(_ROWS)], pa.int64()),
            "tiebreak": pa.array(range(_ROWS), pa.int64()),
        }
    )


def _ranks_by_partition(table: pa.Table) -> dict[int, list[int]]:
    by_partition: dict[int, list[int]] = collections.defaultdict(list)
    for key, rank in zip(table.column("k").to_pylist(), table.column("r").to_pylist(), strict=True):
        by_partition[key % _PARTITIONS].append(rank)
    return {p: sorted(v) for p, v in by_partition.items()}


def test_every_row_is_tied_so_the_freedom_is_actually_exercised(rows):
    """The control. Without it this file could pass on an input that has no ties at all."""
    groups = collections.Counter(
        zip(
            [k % _PARTITIONS for k in rows.column("k").to_pylist()],
            rows.column("o").to_pylist(),
            strict=True,
        )
    )
    assert sum(n for n in groups.values() if n > 1) == rows.num_rows


def test_a_tied_row_number_diverges_only_in_which_tied_row_got_which_rank(rows):
    ds = bt.from_arrow(rows).window(
        partition_by=[bt.col("k") % _PARTITIONS], order_by=["o"], functions={"r": "row_number"}
    )
    single = ds.collect()
    distributed = ds.collect(distributed=True, num_workers=_WORKERS)

    assert distributed.column_names == single.column_names
    assert distributed.num_rows == single.num_rows

    one, many = _ranks_by_partition(single), _ranks_by_partition(distributed)
    assert set(one) == set(many), "the shuffle produced a different set of window partitions"
    for partition, ranks in sorted(one.items()):
        assert len(many[partition]) == len(ranks), (
            f"partition {partition} holds {len(many[partition])} rows distributed and "
            f"{len(ranks)} single-node"
        )
        expected = list(range(1, len(ranks) + 1))
        assert ranks == expected, f"single-node partition {partition} is not 1..n"
        assert many[partition] == expected, (
            f"distributed partition {partition} is not 1..n — a reducer restarted its counter "
            "or a partition was split across two of them"
        )


def test_the_freedom_disappears_when_the_order_key_determines_every_position(rows):
    """With no tie left to break, the two results must be identical row for row."""
    ds = bt.from_arrow(rows).window(
        partition_by=[bt.col("k") % _PARTITIONS],
        order_by=["o", "tiebreak"],
        functions={"r": "row_number"},
    )
    single = ds.collect().sort_by([("tiebreak", "ascending")])
    distributed = ds.collect(distributed=True, num_workers=_WORKERS).sort_by(
        [("tiebreak", "ascending")]
    )
    assert_tables_equal(distributed, single, ordered=True)


def test_rank_and_dense_rank_have_no_such_freedom(rows):
    """They give tied rows the same value by definition, so they must agree exactly."""
    for func in ("rank", "dense_rank"):
        ds = bt.from_arrow(rows).window(
            partition_by=[bt.col("k") % _PARTITIONS], order_by=["o"], functions={"r": func}
        )
        single = ds.collect().sort_by([("tiebreak", "ascending")])
        distributed = ds.collect(distributed=True, num_workers=_WORKERS).sort_by(
            [("tiebreak", "ascending")]
        )
        assert_tables_equal(distributed, single, ordered=True)
