"""The DML entry point: dispatch a statement to its rewrite and gather its ``RETURNING``.

The rewrites themselves live by statement: `rewrite` (INSERT, DELETE, UPDATE), `using`
(``DELETE ... USING``), `upsert` (``INSERT ... ON CONFLICT``) and `merge` (``MERGE
INTO``). This module is only the dispatch, so a reader looking for one statement's
semantics finds it in one place.
"""

from __future__ import annotations

from typing import Any, NamedTuple

from sqlglot import expressions as exp

from batcher._internal.errors import PlanError
from batcher._sql.dml import rewrite
from batcher._sql.dml.merge import merge
from batcher._sql.dml.upsert import upsert
from batcher.api.dataset import Dataset

__all__ = ["DmlResult", "apply_dml"]


class DmlResult(NamedTuple):
    """What a DML statement produced.

    Attributes:
        name: The target table's name, which the caller rebinds.
        state: The target's new state.
        returning: The ``RETURNING`` projection, or None when the statement has none.
    """

    name: str
    state: Dataset
    returning: Dataset | None


def apply_dml(node: Any, registry: dict[str, Dataset], functions: dict[str, Any]) -> DmlResult:
    """Rewrite an INSERT / DELETE / UPDATE / MERGE into the target's new state.

    Args:
        node: The parsed statement.
        registry: Every visible table name and its bound `Dataset` (the session catalog
            plus per-call overrides).
        functions: Registered Python functions.

    Returns:
        The target's name, its new state and the statement's ``RETURNING`` rows. Nothing
        is rebound here; the caller does that.

    Raises:
        PlanError: The statement is not a DML statement this front end serves.
    """
    if isinstance(node, exp.Insert):
        name, current, rows = rewrite.inserted_rows(node, registry, functions)
        if node.args.get("conflict"):
            return DmlResult(name, upsert(node, name, current, rows, registry, functions), None)
        state = current.union(rows, distinct=False)
        return DmlResult(name, state, rewrite.returning(node, rows, registry, functions))
    if isinstance(node, exp.Delete):
        name, kept, deleted = rewrite.delete(node, registry, functions)
        return DmlResult(name, kept, rewrite.returning(node, deleted, registry, functions))
    if isinstance(node, exp.Update):
        name, state, updated = rewrite.update(node, registry, functions)
        return DmlResult(name, state, rewrite.returning(node, updated, registry, functions))
    if isinstance(node, exp.Merge):
        name, state = merge(node, registry, functions)
        return DmlResult(name, state, None)
    raise PlanError(f"unsupported DML statement: {type(node).__name__}")
