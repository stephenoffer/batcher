"""Generic MERGE/upsert composition for non-transactional (file) targets.

Transactional sinks (Delta) use their native ``MERGE``; a plain file target has no
transaction log, so an upsert is a copy-on-write composition over existing
relational ops — anti/semi-join the source against the current target, ``union`` the
surviving/updated/new rows, and atomically overwrite. No new IR.

Because it is read-modify-write over a whole path, a file merge is **not** safe under
concurrent writers (that is exactly what the transactional sinks are for); it is the
single-writer batch-upsert pattern (Delta-style copy-on-write, by hand).

`compose_cdc_apply` is the same idea one level up: reconciling a *change feed* — which
carries deletes, redeliveries, and out-of-order rows — rather than a clean snapshot.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher._internal.errors import PlanError
from batcher.plan.expr_ir import Col, Expr, lit, when

if TYPE_CHECKING:
    from batcher.api.dataset import Dataset

__all__ = [
    "SEQUENCE_COMPARE_COL",
    "cdc_stored_columns",
    "compose_cdc_apply",
    "compose_file_merge",
    "execute_merge",
    "merge_predicate_for",
]

# Helper columns, dropped before the result is written. Named with a reserved prefix so
# they cannot collide with a user column of the change feed.
_DELETE_FLAG = "__bc_cdc_delete"
SEQUENCE_COMPARE_COL = "__bc_cdc_target_seq"
_RESERVED = frozenset({_DELETE_FLAG, SEQUENCE_COMPARE_COL})


def merge_predicate_for(keys: str | list[str]) -> str:
    """Build a Delta ``MERGE`` match predicate from key column(s).

    ``merge_predicate_for(["id", "day"])`` →
    ``"target.id = source.id AND target.day = source.day"`` — the engine's delta
    sink aliases the existing table ``target`` and the new data ``source``.
    """
    cols = [keys] if isinstance(keys, str) else list(keys)
    if not cols:
        raise PlanError("merge_on requires at least one key column")
    return " AND ".join(f"target.{c} = source.{c}" for c in cols)


def execute_merge(
    writer,
    target: str,
    *,
    on: str | list[str],
    when_matched: str,
    when_not_matched: str,
    format: str | None,
    native_sinks: frozenset[str],
    opts: dict,
):
    """Dispatch a `Writer.merge`: native transactional MERGE for `native_sinks`,
    else a copy-on-write file merge (read target → `compose_file_merge` → overwrite).
    Kept here so `Writer` stays a thin namespace."""
    from batcher.api.session import read as _read
    from batcher.io.detect import detect_format
    from batcher.io.filesystem import resolve_filesystem

    fmt = detect_format(target, format)
    keys = [on] if isinstance(on, str) else list(on)
    if fmt in native_sinks:
        return writer.delta(target, merge_on=keys, **opts)
    if not resolve_filesystem(target).exists(target):
        if when_not_matched == "ignore":
            from batcher.io.manifest import WriteManifest

            return WriteManifest()
        return writer(target, fmt, mode="overwrite", **opts)
    target_ds = _read(target, format=fmt)
    merged = compose_file_merge(writer._ds, target_ds, keys, when_matched, when_not_matched)
    return merged.write(target, fmt, mode="overwrite", **opts)


_WHEN_MATCHED = ("update", "delete")
_WHEN_NOT_MATCHED = ("insert", "ignore")


def compose_file_merge(
    source: Dataset,
    target: Dataset,
    on: list[str],
    when_matched: str,
    when_not_matched: str,
) -> Dataset:
    """Compose the merged relation of `source` into `target`, keyed on `on`.

    - ``when_matched="update"``: a matching source row replaces the target row.
    - ``when_matched="delete"``: a matched target row is removed.
    - ``when_not_matched="insert"``: a source row with no target match is added.
    - ``when_not_matched="ignore"``: such source rows are dropped.
    """
    if when_matched not in _WHEN_MATCHED:
        raise PlanError(
            f"merge(): when_matched must be one of {_WHEN_MATCHED}, got {when_matched!r}"
        )
    if when_not_matched not in _WHEN_NOT_MATCHED:
        raise PlanError(
            f"merge(): when_not_matched must be one of {_WHEN_NOT_MATCHED}, "
            f"got {when_not_matched!r}"
        )
    if sorted(source.columns) != sorted(target.columns):
        raise PlanError(
            "merge(): source and target must have the same columns "
            f"(source={source.columns}, target={target.columns})"
        )

    source_keys = source.select(*on).distinct()
    # Target rows with no matching source key always survive untouched.
    survivors = target.join(source_keys, on=on, how="anti")

    if when_matched == "delete":
        # Matched target rows are removed; optionally insert brand-new source rows.
        if when_not_matched == "insert":
            target_keys = target.select(*on).distinct()
            new_rows = source.join(target_keys, on=on, how="anti")
            return survivors.union(new_rows)
        return survivors

    # when_matched == "update": the source row wins for matched keys.
    if when_not_matched == "insert":
        contributed = source  # every source row (updates matched, inserts the rest)
    else:
        target_keys = target.select(*on).distinct()
        contributed = source.join(target_keys, on=on, how="semi")  # only updates
    return survivors.union(contributed)


def cdc_stored_columns(
    source_columns: list[str], keys: list[str], sequence_by: str, columns: list[str] | None
) -> list[str]:
    """The target's column list: `columns` (or every source column), plus keys and sequence.

    The sequence column is *persisted* in the target, not merely used to order this
    batch. That is what lets a later batch recognize a change it has already applied, or
    one that arrives after a newer change already landed — see `compose_cdc_apply`.
    """
    reserved = sorted(set(source_columns) & _RESERVED)
    if reserved:
        # The composition adds these as helper columns. Overwriting a feed column of the
        # same name would silently replace the user's data with a boolean flag.
        raise PlanError(f"apply_changes(): {reserved} are reserved column names; rename them")
    chosen = list(source_columns) if columns is None else list(columns)
    unknown = [c for c in chosen if c not in source_columns]
    if unknown:
        raise PlanError(f"apply_changes(): unknown column(s) {unknown} in the change feed")
    # `dict.fromkeys` dedupes while preserving first-seen order.
    stored = list(dict.fromkeys([*keys, *chosen, sequence_by]))
    missing = [c for c in (*keys, sequence_by) if c not in source_columns]
    if missing:
        raise PlanError(f"apply_changes(): change feed is missing column(s) {missing}")
    return stored


def _latest_per_key(changes: Dataset, keys: list[str], sequence_by: str) -> Dataset:
    """One row per key: the change with the greatest `sequence_by`.

    Collapses redeliveries and out-of-order arrivals *within* the batch. When two
    changes for a key share a sequence value the winner is arbitrary, which is why the
    sequence must be unique per key (the same requirement Delta Live Tables imposes).
    """
    return changes.distinct(subset=keys, keep="last", order_by=sequence_by)


def compose_cdc_apply(
    changes: Dataset,
    target: Dataset | None,
    keys: list[str],
    sequence_by: str,
    stored: list[str],
    deletes: Expr | None,
) -> Dataset:
    """Compose the state of `target` after applying the change feed `changes`.

    A change feed is not a snapshot. It carries deletes, it redelivers rows, and its rows
    arrive out of order. Four rules reconcile it:

    1. **Collapse within the batch** — keep only the greatest-sequence change per key, so
       redeliveries and out-of-order rows *inside* one batch resolve to the latest.
    2. **Reject the stale** — a change applies only if its key is new to the target, or
       its sequence is at least the sequence already stored for that key. ``>=`` rather
       than ``>`` so redelivering the exact change is a no-op rather than a rejection.
    3. **Deletes remove** — an applicable delete drops the target row and inserts
       nothing. A delete for a key that is absent is a tombstone and changes nothing.
    4. **Everything else upserts.**

    Together these make re-applying a batch **idempotent**, and make applying batches in
    non-decreasing sequence order — what a CDC reader produces — converge on the source's
    state.

    They do **not** make the apply commutative across batches. Rule 2 compares against the
    sequence stored *on the row*, and a deleted row stores nothing: the delete is physical,
    not a tombstone. So an old insert replayed after its key was deleted resurrects that
    key, and replaying the whole feed from the beginning need not converge. Delta Live
    Tables' ``STORED AS SCD TYPE 1`` has exactly this shape and exactly this caveat; a
    tombstone column would fix it at the cost of unbounded growth and a compaction story.

    `target` is None when the table does not exist yet, in which case every non-delete
    change is an insert.
    """
    flag = lit(False) if deletes is None else when(deletes).then(lit(True)).otherwise(lit(False))
    # Force a non-null boolean: a NULL delete predicate must mean "not a delete", not
    # "drop this row", which is what a raw `filter(~pred)` on a NULL would do.
    latest = _latest_per_key(changes, keys, sequence_by).with_columns(**{_DELETE_FLAG: flag})
    latest = latest.select(*stored, _DELETE_FLAG)

    if target is None:
        return latest.filter(~Col(_DELETE_FLAG)).select(*stored)

    if sorted(target.columns) != sorted(stored):
        raise PlanError(
            "apply_changes(): the target's columns do not match the ones being applied "
            f"(target={sorted(target.columns)}, applying={sorted(stored)})"
        )

    # The target's current sequence per key, renamed so the left join collides on nothing.
    target_seq = target.select(*keys, sequence_by).rename({sequence_by: SEQUENCE_COMPARE_COL})
    candidates = latest.join(target_seq, on=keys, how="left")
    applicable = candidates.filter(
        Col(SEQUENCE_COMPARE_COL).is_null() | (Col(sequence_by) >= Col(SEQUENCE_COMPARE_COL))
    )

    # A key whose change was rejected as stale keeps its target row; a key whose change
    # applied has its target row replaced (by the upsert) or removed (by the delete).
    applied_keys = applicable.select(*keys).distinct()
    survivors = target.join(applied_keys, on=keys, how="anti")
    upserts = applicable.filter(~Col(_DELETE_FLAG)).select(*stored)
    return survivors.union(upserts)
