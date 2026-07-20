"""`ds.scd.apply_changes` — CDC reconciliation cross-checked against DuckDB.

The change-feed apply is not one relational operator, it is a *specification*: collapse
to the greatest sequence per key, reject a change that is not newer than what is stored,
let deletes remove and everything else upsert. DuckDB can state that specification
directly in SQL, so it is a genuine oracle — Batcher's composition of `distinct` /
`join` / `union` must produce the same relation as DuckDB's window + anti-join.

`seq` is unique per key in every fixture here, because the contract says ties are broken
arbitrarily and an oracle cannot check an arbitrary choice.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same
from batcher.api.merge import cdc_stored_columns, compose_cdc_apply
from batcher.plan.expr_ir import Col

pytestmark = pytest.mark.differential

# The DuckDB statement of the same specification. `applicable` is the set of changes that
# win: newest-per-key within the batch, and not older than the sequence already stored.
_ORACLE = """
WITH latest AS (
    SELECT * FROM (
        SELECT *, row_number() OVER (PARTITION BY id ORDER BY seq DESC) AS rn FROM changes
    ) WHERE rn = 1
),
applicable AS (
    SELECT l.* FROM latest l
    LEFT JOIN target t ON l.id = t.id
    WHERE t.seq IS NULL OR l.seq >= t.seq
)
SELECT id, city, seq FROM target WHERE id NOT IN (SELECT id FROM applicable)
UNION ALL
SELECT id, city, seq FROM applicable WHERE op IS DISTINCT FROM 'DELETE'
"""

_KEYS = ["id"]
_SEQ = "seq"
_COLUMNS = ["id", "city"]


def _apply(duck, changes: pa.Table, target: pa.Table | None):
    """Run Batcher's composition and DuckDB's specification over the same inputs."""
    duck.register("changes", changes)
    duck.register("target", target if target is not None else changes.schema.empty_table())
    changes_ds = bt.from_arrow(changes)
    stored = cdc_stored_columns(changes_ds.columns, _KEYS, _SEQ, _COLUMNS)
    target_ds = None if target is None else bt.from_arrow(target)
    got = compose_cdc_apply(
        changes_ds, target_ds, _KEYS, _SEQ, stored, Col("op") == "DELETE"
    ).to_arrow()
    return got


def _changes(ids, cities, ops, seqs) -> pa.Table:
    return pa.table(
        {
            "id": pa.array(ids, pa.int64()),
            "city": pa.array(cities, pa.string()),
            "op": pa.array(ops, pa.string()),
            "seq": pa.array(seqs, pa.int64()),
        }
    )


def _target(ids, cities, seqs) -> pa.Table:
    return pa.table(
        {
            "id": pa.array(ids, pa.int64()),
            "city": pa.array(cities, pa.string()),
            "seq": pa.array(seqs, pa.int64()),
        }
    )


def test_inserts_and_updates_collapse_to_the_greatest_sequence(duck):
    changes = _changes([1, 1, 2], ["NYC", "SF", "LA"], ["INSERT", "UPDATE", "INSERT"], [1, 3, 2])
    got = _apply(duck, changes, _target([], [], []))
    assert_same(got, duck.sql(_ORACLE))


def test_a_delete_removes_a_matched_row(duck):
    changes = _changes([2], ["LA"], ["DELETE"], [4])
    got = _apply(duck, changes, _target([1, 2], ["SF", "LA"], [3, 2]))
    assert_same(got, duck.sql(_ORACLE))


def test_a_stale_change_is_rejected(duck):
    """A change older than the stored sequence must not resurrect old data."""
    changes = _changes([1], ["OLD"], ["UPDATE"], [0])
    got = _apply(duck, changes, _target([1], ["SF"], [3]))
    assert_same(got, duck.sql(_ORACLE))


def test_redelivering_the_exact_change_is_a_no_op(duck):
    """`>=` rather than `>`: a feed that replays a batch must converge, not reject."""
    changes = _changes([1], ["SF"], ["UPDATE"], [3])
    got = _apply(duck, changes, _target([1], ["SF"], [3]))
    assert_same(got, duck.sql(_ORACLE))


def test_a_stale_delete_does_not_remove_a_newer_row(duck):
    changes = _changes([1], ["SF"], ["DELETE"], [1])
    got = _apply(duck, changes, _target([1], ["SF"], [5]))
    assert_same(got, duck.sql(_ORACLE))


def test_a_tombstone_for_an_absent_key_changes_nothing(duck):
    changes = _changes([9], ["X"], ["DELETE"], [6])
    got = _apply(duck, changes, _target([1], ["SF"], [3]))
    assert_same(got, duck.sql(_ORACLE))


def test_a_key_deleted_then_reinserted_within_one_batch_survives(duck):
    """The greatest sequence wins, and it is the insert."""
    changes = _changes([1, 1], ["OLD", "NEW"], ["DELETE", "INSERT"], [4, 5])
    got = _apply(duck, changes, _target([1], ["SF"], [3]))
    assert_same(got, duck.sql(_ORACLE))


def test_a_key_inserted_then_deleted_within_one_batch_is_gone(duck):
    changes = _changes([1, 1], ["NEW", "NEW"], ["INSERT", "DELETE"], [4, 5])
    got = _apply(duck, changes, _target([1], ["SF"], [3]))
    assert_same(got, duck.sql(_ORACLE))


def test_a_null_operation_is_not_a_delete(duck):
    """`NULL == 'DELETE'` is unknown; the row must upsert, not vanish."""
    changes = _changes([1, 2], ["A", "B"], ["INSERT", None], [1, 2])
    got = _apply(duck, changes, _target([], [], []))
    assert_same(got, duck.sql(_ORACLE))


def test_untouched_target_rows_survive(duck):
    changes = _changes([1], ["NEW"], ["UPDATE"], [9])
    got = _apply(duck, changes, _target([1, 2, 3], ["A", "B", "C"], [1, 1, 1]))
    assert_same(got, duck.sql(_ORACLE))


def test_an_empty_change_feed_leaves_the_target_alone(duck):
    got = _apply(duck, _changes([], [], [], []), _target([1], ["SF"], [3]))
    assert_same(got, duck.sql(_ORACLE))


def test_a_mixed_batch_of_every_case_at_once(duck):
    """Insert, update, delete, stale, tombstone, and an untouched row, in one apply."""
    changes = _changes(
        [1, 2, 3, 4, 5],
        ["stale", "updated", "gone", "brand-new", "tombstone"],
        ["UPDATE", "UPDATE", "DELETE", "INSERT", "DELETE"],
        [0, 7, 8, 9, 10],
    )
    got = _apply(
        duck, changes, _target([1, 2, 3, 6], ["keep", "old", "doomed", "idle"], [5, 1, 1, 1])
    )
    assert_same(got, duck.sql(_ORACLE))
