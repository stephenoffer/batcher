"""Generic MERGE/upsert composition for non-transactional (file) targets.

Transactional sinks (Delta) use their native ``MERGE``; a plain file target has no
transaction log, so an upsert is a copy-on-write composition over existing
relational ops — anti/semi-join the source against the current target, ``union`` the
surviving/updated/new rows, and atomically overwrite. No new IR.

Because it is read-modify-write over a whole path, a file merge is **not** safe under
concurrent writers (that is exactly what the transactional sinks are for); it is the
single-writer batch-upsert pattern (Delta-style copy-on-write, by hand).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher._internal.errors import PlanError

if TYPE_CHECKING:
    from batcher.api.dataset import Dataset

__all__ = ["compose_file_merge", "execute_merge", "merge_predicate_for"]


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
