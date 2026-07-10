"""`ds.scd.apply_changes` across successive batches — the properties a single apply cannot show.

The differential suite pins one apply against DuckDB. What matters operationally is what
happens over *many* applies, because a change feed redelivers rows and reorders them:

* **Idempotent** — re-applying a batch leaves the target as it was after the first apply.
* **Monotone in sequence** — applying batches in non-decreasing sequence order converges
  on the source's state; a change older than the stored one is rejected.

It is deliberately *not* commutative, and one test pins that. A delete is physical rather
than a tombstone, so a deleted key stores no sequence to compare against: replaying an old
insert for it resurrects the key. Delta Live Tables' SCD-type-1 apply has the same shape
and the same caveat. Asserting the behavior keeps it from changing silently.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import PlanError

pytestmark = pytest.mark.integration

DELETES = bt.col("op") == "DELETE"

# An explicit schema, so an empty batch is still typed (an untyped empty column is Arrow
# `null`, which has no ordering and cannot be sequenced).
_FEED_SCHEMA = pa.schema(
    [("id", pa.int64()), ("city", pa.string()), ("op", pa.string()), ("seq", pa.int64())]
)


def _feed(rows):
    """A change feed of ``(id, city, op, seq)`` tuples."""
    columns = list(zip(*rows, strict=True)) if rows else ([], [], [], [])
    arrays = [pa.array(c, f.type) for c, f in zip(columns, _FEED_SCHEMA, strict=True)]
    return bt.from_arrow(pa.table(arrays, schema=_FEED_SCHEMA))


def _apply(target, rows):
    return _feed(rows).scd.apply_changes(
        target, keys="id", sequence_by="seq", deletes=DELETES, columns=["id", "city"]
    )


def _state(target):
    ds = bt.read.parquet(target).sort("id")
    return list(zip(ds.to_pydict()["id"], ds.to_pydict()["city"], strict=True))


@pytest.fixture
def target(tmp_path):
    return str(tmp_path / "customers.parquet")


def test_a_first_apply_creates_the_target_from_the_inserts(target):
    _apply(target, [(1, "NYC", "INSERT", 1), (2, "LA", "INSERT", 2)])
    assert _state(target) == [(1, "NYC"), (2, "LA")]


def test_a_delete_in_the_first_batch_inserts_nothing(target):
    _apply(target, [(1, "NYC", "INSERT", 1), (2, "LA", "DELETE", 2)])
    assert _state(target) == [(1, "NYC")]


def test_the_sequence_column_is_persisted_so_later_batches_can_reject_stale_changes(target):
    _apply(target, [(1, "NYC", "INSERT", 5)])
    assert bt.read.parquet(target).to_pydict()["seq"] == [5]


def test_a_later_batch_updates_and_deletes(target):
    _apply(target, [(1, "NYC", "INSERT", 1), (2, "LA", "INSERT", 2)])
    _apply(target, [(1, "SF", "UPDATE", 3), (2, "LA", "DELETE", 4)])
    assert _state(target) == [(1, "SF")]


def test_a_stale_change_from_a_later_batch_is_rejected(target):
    """The out-of-order arrival that would otherwise resurrect old data."""
    _apply(target, [(1, "SF", "UPDATE", 3)])
    _apply(target, [(1, "OLD", "UPDATE", 0)])
    assert _state(target) == [(1, "SF")]


def test_a_stale_delete_does_not_remove_a_newer_row(target):
    _apply(target, [(1, "SF", "UPDATE", 5)])
    _apply(target, [(1, "SF", "DELETE", 1)])
    assert _state(target) == [(1, "SF")]


def test_applying_a_batch_twice_is_idempotent(target):
    batch = [(1, "NYC", "INSERT", 1), (2, "LA", "INSERT", 2)]
    _apply(target, batch)
    once = _state(target)
    _apply(target, batch)
    assert _state(target) == once


_FEED = [
    (1, "NYC", "INSERT", 1),
    (2, "LA", "INSERT", 2),
    (1, "SF", "UPDATE", 3),
    (2, "LA", "DELETE", 4),
]


def test_applying_a_feed_one_change_at_a_time_matches_applying_it_at_once(tmp_path):
    """Batch boundaries are not semantics: the batching of a feed must not change it."""
    streamed, at_once = str(tmp_path / "s.parquet"), str(tmp_path / "a.parquet")
    for change in _FEED:
        _apply(streamed, [change])
    _apply(at_once, _FEED)
    assert _state(streamed) == _state(at_once) == [(1, "SF")]


def test_re_applying_the_last_batch_of_a_feed_is_a_no_op(target):
    """Idempotence is what a reader that cannot commit its offset atomically relies on."""
    for change in _FEED:
        _apply(target, [change])
    once = _state(target)
    _apply(target, [_FEED[-1]])
    assert _state(target) == once


def test_replaying_an_old_insert_for_a_deleted_key_resurrects_it(target):
    """The documented non-commutativity: a physical delete leaves no sequence to compare.

    Pinned so the limitation cannot change silently. Fixing it would mean storing a
    tombstone per deleted key, which grows without bound and needs a compaction story.
    """
    _apply(target, [(1, "NYC", "INSERT", 5)])
    _apply(target, [(1, "NYC", "DELETE", 6)])
    assert _state(target) == []
    _apply(target, [(1, "NYC", "INSERT", 5)])  # an old change, replayed
    assert _state(target) == [(1, "NYC")]


def test_an_empty_change_batch_leaves_the_target_untouched(target):
    """The routine incremental case: this run's feed had nothing in it."""
    _apply(target, [(1, "SF", "UPDATE", 3)])
    _apply(target, [])
    assert _state(target) == [(1, "SF")]


def test_a_deleted_key_can_be_reinserted_by_a_newer_change(target):
    _apply(target, [(1, "NYC", "INSERT", 1)])
    _apply(target, [(1, "NYC", "DELETE", 2)])
    assert _state(target) == []
    _apply(target, [(1, "SD", "INSERT", 3)])
    assert _state(target) == [(1, "SD")]


def test_a_tombstone_for_a_key_that_was_never_seen_changes_nothing(target):
    _apply(target, [(1, "SF", "INSERT", 1)])
    _apply(target, [(9, "X", "DELETE", 2)])
    assert _state(target) == [(1, "SF")]


def test_a_feed_without_deletes_is_a_sequenced_upsert(tmp_path):
    path = str(tmp_path / "t.parquet")
    bt.from_pydict({"id": [1, 1], "v": [10, 20], "seq": [1, 2]}).scd.apply_changes(
        path, keys="id", sequence_by="seq"
    )
    assert bt.read.parquet(path).to_pydict() == {"id": [1], "v": [20], "seq": [2]}


def test_composite_keys(tmp_path):
    path = str(tmp_path / "t.parquet")
    bt.from_pydict(
        {"a": [1, 1, 2], "b": ["x", "x", "y"], "v": [1, 2, 3], "seq": [1, 2, 3]}
    ).scd.apply_changes(path, keys=["a", "b"], sequence_by="seq")
    got = bt.read.parquet(path).sort("a").to_pydict()
    assert got["v"] == [2, 3]


def test_control_columns_are_excluded_from_the_target(target):
    """`op` drives the delete predicate but must not be stored."""
    _apply(target, [(1, "NYC", "INSERT", 1)])
    assert bt.read.parquet(target).columns == ["id", "city", "seq"]


def test_a_target_whose_columns_do_not_match_is_rejected(tmp_path):
    path = str(tmp_path / "t.parquet")
    bt.from_pydict({"id": [1], "other": ["x"], "seq": [1]}).write(path, format="parquet")
    with pytest.raises(PlanError, match="columns do not match"):
        bt.from_pydict({"id": [1], "v": [2], "seq": [2]}).scd.apply_changes(
            path, keys="id", sequence_by="seq"
        )


@pytest.mark.parametrize("reserved", ["__bc_cdc_delete", "__bc_cdc_target_seq"])
def test_a_feed_column_named_like_an_internal_helper_is_rejected(tmp_path, reserved):
    """The composition adds these columns; overwriting one would silently destroy data."""
    path = str(tmp_path / "t.parquet")
    with pytest.raises(PlanError, match="reserved"):
        bt.from_pydict({"id": [1], "seq": [1], reserved: [9]}).scd.apply_changes(
            path, keys="id", sequence_by="seq"
        )
