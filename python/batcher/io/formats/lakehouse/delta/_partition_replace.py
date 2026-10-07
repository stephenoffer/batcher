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

__all__ = ["partition_removes", "partitions_outside", "to_partition_dnf"]

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


def partitions_outside(ir: dict[str, Any], files: Any, schema: Any) -> list[dict[str, Any]]:
    """The partition values of each written file a ``replace_where`` does not cover.

    A partition-scoped overwrite retires only the partitions the predicate names, while the
    commit adds *every* file the write produced. A file landing in some other partition is
    therefore appended beside that partition's existing rows rather than replacing them:
    backfilling ``region = 'us'`` with a stray ``eu`` row leaves ``eu`` holding both its old
    row and the new one, and nothing reports it. Spark's Delta refuses such a write by
    default, and so does this.

    Decided from metadata alone: each file holds one partition, so its partition values are
    the whole question, and they are evaluated with the predicate's own typed semantics
    (one row per file) rather than restated as string comparisons. A NULL partition value
    makes the predicate NULL, which is not a match, exactly as it is for a row.

    Args:
        ir: The ``replace_where`` predicate IR, already known to be partition-scoped.
        files: The `WrittenFile`s the commit is about to add.
        schema: The write's Arrow schema, which types the partition values.

    Returns:
        One ``{column: value}`` per out-of-scope file that holds rows; empty when all fit.
    """
    import pyarrow as pa

    from batcher.io.predicate import to_pyarrow_expression

    written = [f for f in files if f.rows]
    if not written:
        return []
    columns = sorted({c for f in written for c in f.partition_values})
    fields = [
        schema.field(c) if schema is not None and c in schema.names else pa.field(c, pa.string())
        for c in columns
    ]
    probe = pa.table(
        {
            **{
                field.name: pa.array([f.partition_values.get(field.name) for f in written]).cast(
                    field.type
                )
                for field in fields
            },
            "__file__": pa.array(range(len(written)), pa.int64()),
        }
    )
    expression = to_pyarrow_expression(ir, probe.schema)
    if expression is None:
        raise ValueError("the replace_where predicate has no pyarrow form")
    inside = set(probe.filter(expression).column("__file__").to_pylist())
    return [dict(f.partition_values) for i, f in enumerate(written) if i not in inside]


def _matches(values: dict[str, Any], conjunct: Conjunct) -> bool:
    return all(
        values.get(column) is not None and str(values[column]) == value
        for column, _, value in conjunct
    )
