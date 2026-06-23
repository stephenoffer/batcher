"""Grouping and ordering logical nodes: `Aggregate` and `Sort` (and their specs).

Both are pipeline breakers in spirit — `Aggregate` groups and computes mergeable
aggregates; `Sort` orders rows (and carries an optional top-N `limit`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from batcher.plan.expr_ir import AggExpr, Expr
from batcher.plan.ir_tags import Op
from batcher.plan.logical.base import LogicalPlan, _validate_refs
from batcher.plan.logical.relational import Projection

__all__ = ["Aggregate", "AggregateSpec", "Sort", "SortKeySpec"]


@dataclass(frozen=True, slots=True)
class AggregateSpec:
    """One aggregate output: a name, function, and optional input expression."""

    alias: str
    agg: AggExpr


@dataclass(frozen=True, slots=True)
class Aggregate(LogicalPlan):
    """Group by key expressions and compute aggregates. A pipeline breaker."""

    input: LogicalPlan
    group_keys: tuple[Projection, ...]
    aggregates: tuple[AggregateSpec, ...]

    def __post_init__(self) -> None:
        available = set(self.input.available_columns())
        for key in self.group_keys:
            _validate_refs(key.expr, available, what=f"group_by key {key.alias!r}")
        for spec in self.aggregates:
            if spec.agg.input is not None:
                _validate_refs(spec.agg.input, available, what=f"aggregate {spec.alias!r}")

    def to_ir(self) -> dict[str, Any]:
        return {
            "op": Op.AGGREGATE,
            "input": self.input.to_ir(),
            "group_keys": [{"expr": k.expr.to_ir(), "alias": k.alias} for k in self.group_keys],
            "aggregates": [s.agg.to_ir(s.alias) for s in self.aggregates],
        }

    def available_columns(self) -> list[str]:
        return [k.alias for k in self.group_keys] + [s.alias for s in self.aggregates]


@dataclass(frozen=True, slots=True)
class SortKeySpec:
    """One sort key: an expression and its ordering."""

    expr: Expr
    descending: bool = False
    nulls_first: bool = False


@dataclass(frozen=True, slots=True)
class Sort(LogicalPlan):
    """Order rows by sort keys. Preserves the input schema.

    `limit` (set by the top-N fusion pass when a `Limit` sits directly above)
    turns this into a top-N: the engine produces only the first `limit` rows via
    a partial sort instead of fully sorting.
    """

    input: LogicalPlan
    keys: tuple[SortKeySpec, ...]
    limit: int | None = None

    def __post_init__(self) -> None:
        available = set(self.input.available_columns())
        for key in self.keys:
            _validate_refs(key.expr, available, what="sort key")

    def to_ir(self) -> dict[str, Any]:
        return {
            "op": Op.SORT,
            "input": self.input.to_ir(),
            "keys": [
                {
                    "expr": k.expr.to_ir(),
                    "descending": k.descending,
                    "nulls_first": k.nulls_first,
                }
                for k in self.keys
            ],
            "limit": self.limit,
        }

    def available_columns(self) -> list[str]:
        return self.input.available_columns()
