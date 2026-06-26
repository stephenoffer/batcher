"""Grouping and ordering logical nodes: `Aggregate` and `Sort` (and their specs).

Both are pipeline breakers in spirit — `Aggregate` groups and computes mergeable
aggregates; `Sort` orders rows (and carries an optional top-N `limit`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pyarrow as pa

from batcher.plan.expr_ir import AggExpr, Expr
from batcher.plan.ir_tags import Op
from batcher.plan.logical.base import LogicalPlan, _validate_refs
from batcher.plan.logical.relational import Projection
from batcher.plan.schema import SchemaRef
from batcher.plan.streaming import Watermark
from batcher.plan.types import infer_type, widen

__all__ = ["Aggregate", "AggregateSpec", "Sort", "SortKeySpec"]

# Aggregate function → output-type category (the engine's result types).
_AGG_INT = frozenset({"count", "count_distinct", "count_star", "approx_count_distinct"})
_AGG_FLOAT = frozenset(
    {
        "mean",
        "median",
        "quantile",
        "approx_quantile",
        "stddev",
        "var",
        "corr",
        "covar_pop",
        "covar_samp",
        "skewness",
        "kurtosis",
    }
)
_AGG_BOOL = frozenset({"bool_and", "bool_or"})
_AGG_INPUT = frozenset({"min", "max", "mode", "arg_min", "arg_max"})  # preserve input type
_AGG_WIDEN_INPUT = frozenset({"sum", "product", "bit_and", "bit_or", "bit_xor"})  # widen(input)


def _agg_output_type(agg: AggExpr, input_schema: SchemaRef) -> pa.DataType | None:
    """The Arrow type an aggregate produces, or ``None`` if not certain."""
    func = agg.func
    if func in _AGG_INT:
        return pa.int64()
    if func in _AGG_FLOAT:
        return pa.float64()
    if func in _AGG_BOOL:
        return pa.bool_()
    if func in _AGG_INPUT or func in _AGG_WIDEN_INPUT:
        if agg.input is None:
            return None
        t = infer_type(agg.input, input_schema)
        if t is None:
            return None
        return widen(t) if func in _AGG_WIDEN_INPUT else t
    return None  # histogram, list_agg, … — leave to the engine


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
    # Driver-only event-time watermark (set via `Dataset.with_watermark` ahead of the
    # group-by); bounds streaming windowed-aggregation state. Never serialized to IR.
    watermark: Watermark | None = None

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

    def available_schema(self) -> SchemaRef | None:
        inp = self.input.available_schema()
        if inp is None:
            return None
        fields: list[pa.Field] = []
        for key in self.group_keys:
            t = infer_type(key.expr, inp)
            if t is None:
                return None
            fields.append(pa.field(key.alias, t))
        for spec in self.aggregates:
            t = _agg_output_type(spec.agg, inp)
            if t is None:
                return None
            fields.append(pa.field(spec.alias, t))
        return SchemaRef.from_arrow(pa.schema(fields))


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

    def available_schema(self) -> SchemaRef | None:
        return self.input.available_schema()
