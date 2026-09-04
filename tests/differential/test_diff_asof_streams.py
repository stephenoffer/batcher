"""A keyless ASOF join must stream the left past a materialized right, not build both.

`join_asof(..., by=...)` co-partitions on `by` and rides the grace join, which is bounded on
both sides. A **keyless** ASOF has no group to hash, so that decomposition is unavailable —
and the shape materialized the whole join.

It does not need a decomposition, because it is not a fold. Each left row's match is a
**lookup**: the nearest right row at or before it (or after it, within `tolerance`), which
depends on that row's `on` value and the right side alone. No state carries from one left row
to the next. So running the operator per left batch against the whole right side yields the
collected rows in the collected order, and peak memory falls from the join to the right side —
which is the ordinary shape here, a large fact stream against a smaller quote table.

**Every assertion is row-for-row, not a multiset.** Output order is the property a per-batch
rewrite is most likely to disturb, and the comparison that would not see it is the one this
repo reaches for by default. The unsorted-left case is here for the same reason: the operator
emits in left-row order regardless of the `on` ordering, so a rewrite that quietly sorted
would pass a set comparison and fail this one.
"""

from __future__ import annotations

import random

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_tables_equal

pytestmark = pytest.mark.differential

_N = 3000


@pytest.fixture(scope="module")
def sides() -> tuple[pa.Table, pa.Table]:
    rng = random.Random(20260826)
    left = pa.table(
        {
            "rid": pa.array(range(_N), pa.int64()),
            "t": pa.array(sorted(rng.sample(range(100_000), _N)), pa.int64()),
            "v": pa.array([rng.random() for _ in range(_N)], pa.float64()),
        }
    )
    right = pa.table(
        {
            "t": pa.array(sorted(rng.sample(range(100_000), 400)), pa.int64()),
            "q": pa.array([rng.random() for _ in range(400)], pa.float64()),
        }
    )
    return left, right


#: Every knob that decides which right row a match picks. Each is resolved inside the lookup,
#: so a rewrite that respected the shape but mishandled one would return plausible rows.
_SETTINGS = {
    "backward": {},
    "forward": {"direction": "forward"},
    "nearest": {"direction": "nearest"},
    "tolerance": {"tolerance": 500},
    "forward_tolerance": {"direction": "forward", "tolerance": 500},
    "no_exact_matches": {"allow_exact_matches": False},
}


def _streamed(ds) -> pa.Table:
    batches = list(ds.iter_batches())
    return pa.Table.from_batches(batches) if batches else ds.collect().slice(0, 0)


@pytest.mark.parametrize("setting", sorted(_SETTINGS))
def test_a_streamed_keyless_asof_equals_the_collected_one(sides, setting):
    left, right = sides
    ds = bt.from_arrow(left).join_asof(bt.from_arrow(right), on="t", **_SETTINGS[setting])
    assert_tables_equal(_streamed(ds), ds.collect(), ordered=True)


def test_it_actually_streams_rather_than_materializing(sides):
    """The point of the change. Every assertion above passes on the materializing path."""
    import batcher.api.terminal.stream.bounded as bounded

    left, right = sides
    fired: list[int] = []
    original = bounded._asof_batches
    bounded._asof_batches = lambda *a, **k: (fired.append(1), original(*a, **k))[1]
    try:
        ds = bt.from_arrow(left).join_asof(bt.from_arrow(right), on="t")
        list(ds.iter_batches())
    finally:
        bounded._asof_batches = original
    assert fired, "the keyless ASOF materialized instead of streaming the left"


def test_a_by_keyed_asof_keeps_the_grace_join(sides):
    """The control. A `by`-keyed ASOF must **not** take this path.

    It has a decomposition that bounds *both* sides, which is the better plan when the right
    is large; the lookup driver holds the whole right in memory. Routing it here would still
    return the right rows, so nothing but this assertion would notice the regression.
    """
    import batcher.api.terminal.stream.bounded as bounded

    left, right = sides
    keyed_left = left.append_column("g", pa.array([f"s{i % 5}" for i in range(left.num_rows)]))
    keyed_right = right.append_column("g", pa.array([f"s{i % 5}" for i in range(right.num_rows)]))
    fired: list[int] = []
    original = bounded._asof_batches
    bounded._asof_batches = lambda *a, **k: (fired.append(1), original(*a, **k))[1]
    try:
        ds = bt.from_arrow(keyed_left).join_asof(bt.from_arrow(keyed_right), on="t", by="g")
        streamed = _streamed(ds)
    finally:
        bounded._asof_batches = original
    assert not fired, "a by-keyed ASOF took the lookup driver instead of the grace join"
    assert_tables_equal(
        streamed.sort_by([("rid", "ascending")]),
        ds.collect().sort_by([("rid", "ascending")]),
        ordered=True,
    )


def test_an_unsorted_left_keeps_its_own_row_order(sides):
    """The operator emits in left-row order whatever the `on` ordering, so a per-batch rewrite
    that quietly sorted would pass a multiset comparison and fail this one."""
    left, right = sides
    rng = random.Random(7)
    order = list(range(left.num_rows))
    rng.shuffle(order)
    shuffled = left.take(pa.array(order, pa.int64()))
    ds = bt.from_arrow(shuffled).join_asof(bt.from_arrow(right), on="t")
    assert_tables_equal(_streamed(ds), ds.collect(), ordered=True)
