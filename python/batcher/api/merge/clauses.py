"""The clause model of a SQL ``MERGE`` — and how a clause names a source vs a target column.

A ``MERGE`` row lives in one of exactly three populations, and SQL gives each its own
family of clauses:

- **matched** — the key exists on both sides (``WHEN MATCHED``): update it or delete it.
- **not matched** — a source key with no target row (``WHEN NOT MATCHED``): insert it.
- **not matched by source** — a target key absent from the source
  (``WHEN NOT MATCHED BY SOURCE``): update it or delete it. This is the clause SCD-2
  expiry, soft-deletes, and full-snapshot reconciliation are built on, and the one a
  naive upsert forgets.

Within a population, clauses are tried **in order and the first whose condition holds
wins** — which is precisely `Expr`'s chained ``CASE``, so the whole of ``MERGE`` composes
out of the IR that already exists (see `compose`). A row matched by *no* clause is left
exactly as it was (or, for an insert population, simply not inserted).

## Naming the two sides

A matched clause can see both rows, so ``amount`` is ambiguous — SQL disambiguates with
``source.amount`` / ``target.amount``. `source_col` and `target_col` are that, as
expressions. They are ordinary `Col` references under the hood: the composition renames
every source column to a reserved prefix before joining, so the two sides can never
collide and no suffix guessing is involved.

`target_col` is meaningless in an INSERT clause (there is no target row yet) and
`source_col` is meaningless in a NOT MATCHED BY SOURCE clause (there is no source row);
`validate_clause` rejects both rather than letting them silently read a null.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from batcher._internal.errors import PlanError
from batcher.plan.expr_ir import Col, Expr

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "MATCHED",
    "NOT_MATCHED",
    "NOT_MATCHED_BY_SOURCE",
    "SOURCE_PREFIX",
    "MergeClause",
    "source_col",
    "source_name",
    "target_col",
    "validate_clause",
]

# Every source column is renamed to this prefix inside the composed relation, so a source
# and a target column of the same name coexist without a join suffix and without either
# shadowing the other. Reserved: a user column already carrying it is rejected up front.
SOURCE_PREFIX = "__bc_src_"

MATCHED = "matched"
NOT_MATCHED = "not_matched"
NOT_MATCHED_BY_SOURCE = "not_matched_by_source"

_ACTIONS = {
    MATCHED: ("update", "delete"),
    NOT_MATCHED: ("insert",),
    NOT_MATCHED_BY_SOURCE: ("update", "delete"),
}


def source_name(column: str) -> str:
    """The reserved name `column` from the source side takes inside a composed merge."""
    return f"{SOURCE_PREFIX}{column}"


def source_col(column: str) -> Col:
    """Reference the **source** row's `column` inside a merge clause (SQL ``source.col``).

    Valid in ``when_matched`` and ``when_not_matched`` clauses. A
    ``when_not_matched_by_source`` clause has no source row, so using it there is an error.

    Examples:
        .. doctest::

            >>> from batcher import source_col
            >>> source_col("amount").to_ir()["name"]
            '__bc_src_amount'

    Args:
        column: The source column's name, as it is spelled in the source dataset.

    Returns:
        An expression reading that column from the source side of the merge.
    """
    return Col(source_name(column))


def target_col(column: str) -> Col:
    """Reference the **target** row's `column` inside a merge clause (SQL ``target.col``).

    Valid in ``when_matched`` and ``when_not_matched_by_source`` clauses. A
    ``when_not_matched`` (insert) clause has no target row, so using it there is an error.

    Examples:
        .. doctest::

            >>> from batcher import target_col
            >>> target_col("amount").to_ir()["name"]
            'amount'

    Args:
        column: The target column's name, as it is spelled in the target table.

    Returns:
        An expression reading that column from the target side of the merge.
    """
    return Col(column)


@dataclass(frozen=True, slots=True)
class MergeClause:
    """One ``WHEN …`` clause: a population, an action, a guard, and the values it writes.

    `values` maps a **target** column to the expression producing its new value. `None`
    means "all columns", i.e. SQL's ``UPDATE SET *`` / ``INSERT *``: every target column
    is taken from the source column of the same name.
    """

    kind: str
    action: str
    condition: Expr | None = None
    values: Mapping[str, Expr] | None = None

    @property
    def is_delete(self) -> bool:
        """True when this clause removes the row rather than writing one."""
        return self.action == "delete"


def validate_clause(clause: MergeClause, target_columns: list[str]) -> None:
    """Reject a clause that names an unknown target column or the wrong side of the merge.

    Caught here rather than at execution because the failure modes are silent ones: an
    expression referencing the absent side would read a null, and a `values` key that is
    not a target column would be dropped by the final projection.
    """
    if clause.kind not in _ACTIONS:
        raise PlanError(f"merge(): unknown clause kind {clause.kind!r}")
    if clause.action not in _ACTIONS[clause.kind]:
        raise PlanError(
            f"merge(): a {clause.kind!r} clause cannot {clause.action!r}; "
            f"it must be one of {_ACTIONS[clause.kind]}"
        )

    for column in clause.values or {}:
        if column not in target_columns:
            raise PlanError(
                f"merge(): {clause.kind} clause writes unknown target column {column!r} "
                f"(target columns: {target_columns})"
            )

    referenced = _referenced(clause)
    if clause.kind == NOT_MATCHED_BY_SOURCE:
        offending = sorted(c for c in referenced if c.startswith(SOURCE_PREFIX))
        if offending:
            names = [c.removeprefix(SOURCE_PREFIX) for c in offending]
            raise PlanError(
                "merge(): a not-matched-by-source clause has no source row, so it cannot "
                f"reference source_col({names[0]!r}) — use target_col() instead"
            )
    if clause.kind == NOT_MATCHED:
        # An insert has no target row: any *bare* target reference would read a null.
        offending = sorted(
            c for c in referenced if not c.startswith(SOURCE_PREFIX) and c in target_columns
        )
        if offending:
            raise PlanError(
                "merge(): a not-matched (insert) clause has no target row, so it cannot "
                f"reference target_col({offending[0]!r}) — use source_col() instead"
            )


def _referenced(clause: MergeClause) -> set[str]:
    """Every column name the clause's condition and value expressions read."""
    names: set[str] = set()
    exprs = [*(clause.values or {}).values()]
    if clause.condition is not None:
        exprs.append(clause.condition)
    for expr in exprs:
        _walk(expr, names)
    return names


def _walk(expr: Expr, names: set[str]) -> None:
    """Collect `Col` names from `expr`, walking the IR dict so every node kind is covered.

    Serializing to IR is how the expression tree is traversed generically — hand-rolling a
    visitor here would need a case per `Expr` subclass and would silently miss new ones.
    """
    _collect(expr.to_ir(), names)


def _collect(node: object, names: set[str]) -> None:
    if isinstance(node, dict):
        if node.get("e") == "col" and isinstance(node.get("name"), str):
            names.add(node["name"])
        for value in node.values():
            _collect(value, names)
    elif isinstance(node, list):
        for value in node:
            _collect(value, names)
