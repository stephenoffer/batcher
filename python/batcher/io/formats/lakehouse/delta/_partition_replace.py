"""Replacing several Delta partitions in one commit.

delta-rs scopes an overwrite with ``partition_filters``, and those are one conjunction:
``(p = 'a') AND (q = 'x')``. Replacing two days is a disjunction, which it cannot take, so a
reload of several partitions used to be one commit per partition. Each was atomic and the
set was not: a reader between two commits saw some partitions reloaded and others not.

A Delta commit is a list of actions, and nothing requires the removals to come from a
filter. So a disjunction of partition equalities is resolved here against the current
snapshot's add-actions into explicit `RemoveAction`s, and the caller commits them together
with the new files' `AddAction`s. One log entry, one version: a reader sees the old
partitions or the new ones, never a mixture.

Only equality on partition columns qualifies, because that is what can be decided from the
log alone (a partition value is its path segment). Anything else returns ``None`` and the
caller keeps its existing behaviour.

Layer: io (neutral).
"""

from __future__ import annotations

import time
from typing import Any

from batcher.io.formats.lakehouse.delta._predicate import to_partition_filters

__all__ = ["partition_removes", "to_partition_dnf"]

Conjunct = list[tuple[str, str, str]]


def to_partition_dnf(
    ir: dict[str, Any] | None, partition_columns: list[str]
) -> list[Conjunct] | None:
    """`ir` as an OR of partition-equality conjunctions, or None when it is not one.

    Args:
        ir: The predicate IR.
        partition_columns: The table's partition columns.

    Returns:
        One ``[(column, "=", value), ...]`` list per disjunct, or None.
    """
    if ir is None or not partition_columns:
        return None
    disjuncts: list[dict[str, Any]] = []

    def split(node: dict[str, Any]) -> None:
        if node.get("e") == "binary" and node.get("op") == "or":
            split(node["left"])
            split(node["right"])
        else:
            disjuncts.append(node)

    split(ir)
    out: list[Conjunct] = []
    for node in disjuncts:
        conjunct = to_partition_filters(node, partition_columns)
        if conjunct is None or any(op != "=" for _, op, _ in conjunct):
            return None
        out.append(conjunct)
    return out


def partition_removes(table: Any, dnf: list[Conjunct]) -> list[Any]:
    """`RemoveAction`s for every live file whose partition matches any conjunct of `dnf`.

    Args:
        table: A loaded `deltalake.DeltaTable` at the snapshot being replaced.
        dnf: The disjunction from `to_partition_dnf`.

    Returns:
        One removal per matching data file.
    """
    import pyarrow as pa
    from deltalake.transaction import RemoveAction

    adds = pa.table(table.get_add_actions(flatten=True)).to_pylist()
    columns = sorted({column for conjunct in dnf for column, _, _ in conjunct})
    now = int(time.time() * 1000)
    removes = []
    for add in adds:
        values = {c: add.get(f"partition.{c}") for c in columns}
        if not any(_matches(values, conjunct) for conjunct in dnf):
            continue
        removes.append(
            RemoveAction(
                path=add["path"],
                data_change=True,
                deletion_timestamp=now,
                size=add.get("size_bytes"),
                partition_values={
                    key[len("partition.") :]: (None if value is None else str(value))
                    for key, value in add.items()
                    if key.startswith("partition.")
                },
            )
        )
    return removes


def _matches(values: dict[str, Any], conjunct: Conjunct) -> bool:
    return all(
        values.get(column) is not None and str(values[column]) == value
        for column, _, value in conjunct
    )
