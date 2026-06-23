"""`LogicalPlan` — the base class for declarative plan nodes.

Immutable node tree. Each fluent `Dataset` operation returns a new `LogicalPlan`
wrapping the previous one. Validation (column references resolve against the
input's available columns) happens at build time so mistakes fail fast, before
the optimizer or engine ever runs. Logical plans lower to the relational IR JSON
via `to_ir()`; types of derived columns are resolved by the engine.
"""

from __future__ import annotations

from typing import Any

from batcher._internal.errors import PlanError
from batcher.plan.expr_ir import Expr
from batcher.plan.expr_ir import referenced_columns as _referenced_columns

__all__ = ["LogicalPlan"]


def _validate_refs(expr: Expr, available: set[str], *, what: str) -> None:
    """Raise `PlanError` if `expr` references a column not in `available`.

    The single source of the "unknown column(s)" validation message; `what`
    labels the site (e.g. ``"filter"``, ``f"projection {alias!r}"``).
    """
    missing = _referenced_columns(expr) - available
    if missing:
        raise PlanError(
            f"{what} references unknown column(s) {sorted(missing)}; available: {sorted(available)}"
        )


class LogicalPlan:
    """Base class for logical plan nodes."""

    def to_ir(self) -> dict[str, Any]:  # pragma: no cover - overridden
        raise NotImplementedError

    def available_columns(self) -> list[str]:  # pragma: no cover - overridden
        raise NotImplementedError

    def _check(self, expr: Expr) -> None:
        """Raise `PlanError` if `expr` references a column not produced by input."""
        _validate_refs(expr, set(self.available_columns()), what="expression")
