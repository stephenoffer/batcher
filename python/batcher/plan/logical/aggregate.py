"""Grouping and ordering logical nodes: `Aggregate` and `Sort` (and their specs).

Both are pipeline breakers in spirit — `Aggregate` groups and computes mergeable
aggregates; `Sort` orders rows (and carries an optional top-N `limit`).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import chain
from typing import Any

import pyarrow as pa

from batcher._internal.errors import PlanError
from batcher.plan.expr_ir import AggExpr
from batcher.plan.ir_specs import aggregates_ir, group_keys_ir, sort_keys_ir
from batcher.plan.ir_tags import AGG_FNS, Op
from batcher.plan.logical.base import (
    LogicalPlan,
    SortKeySpec,
    _reject_duplicate_aliases,
    _validate_refs,
    available_column_set,
    validate_key_domains,
)
from batcher.plan.logical.relational import Projection
from batcher.plan.schema import SchemaRef
from batcher.plan.streaming import Watermark
from batcher.plan.types import infer_type, widen

__all__ = ["Aggregate", "AggregateSpec", "Sort", "SortKeySpec"]

# Aggregate function → output-type category (the engine's result types).
_AGG_INT = frozenset(
    # `l_count` is a number of contigs, so Int64 — reporting it as a float would be the
    # same mistake as a fractional row count.
    {"count", "count_distinct", "count_star", "approx_count_distinct", "l_count"}
)
_AGG_FLOAT = frozenset(
    {
        "mean",
        "median",
        "quantile",
        "approx_quantile",
        # Contiguity lengths: `n_length` is a contig length and `aun` a base-weighted mean,
        # both Float64 whatever the input length column's type.
        "n_length",
        "aun",
        "stddev",
        "var",
        "corr",
        "covar_pop",
        "covar_samp",
        "skewness",
        "kurtosis",
        # `product` is unconditionally Float64 in the engine (Rust `AggFunc::Product`),
        # not `widen(input)` — an int column's product still comes back as double.
        "product",
    }
)
_AGG_BOOL = frozenset({"bool_and", "bool_or"})
# `l_count` is a number of contigs, so Int64 — reporting it as a float would be the
# same mistake as a fractional row count. `n_length`/`aun` are lengths and land in
# `_AGG_FLOAT` beside the other length-valued statistics.
_AGG_INPUT = frozenset({"min", "max", "mode", "arg_min", "arg_max"})  # preserve input type
_AGG_WIDEN_INPUT = frozenset({"sum", "bit_and", "bit_or", "bit_xor"})  # widen(input)


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
        if pa.types.is_null(t):
            # A `null`-typed input is the one case where these do not preserve their input:
            # the accumulators have no null-typed slot to gather into, so they materialize
            # the group's values as Int64 and the column comes back `int64` all-null. The
            # column *does* survive as `null` when nothing aggregates it (a passthrough
            # crosses the boundary untouched), so this is an aggregate rule and not a
            # widening one — `widen` is right to leave `null` alone.
            return pa.int64()
        return widen(t) if func in _AGG_WIDEN_INPUT else t
    return None  # histogram, list_agg, … — leave to the engine


def _validate_agg_input_types(source: LogicalPlan, aggregates: tuple[AggregateSpec, ...]) -> None:
    """Reject an aggregate over an input type it cannot mean anything over.

    The type counterpart to `_validate_refs` above, and it runs on the same terms: at build
    time when the input's types are known, silently when they are not. What it catches used
    to surface two ways, both bad. A pair the accumulators reject (``SUM`` over a string)
    reached the user as a raw ``RuntimeError`` from Rust *after the scan* -- hours in, on a
    large job -- while `Dataset.schema` had been reporting an output type for it all along.
    A pair they tolerate but cannot interpret (``PRODUCT`` over a timestamp, which
    multiplies epoch microseconds; ``PRODUCT`` over a string, which returns all nulls)
    reached them as a wrong answer with no diagnostic at all.

    The rule lives in `plan.types.domains` beside the rest of the type vocabulary; this
    is only the plan-tree half, resolving each aggregate's input expression to a type.
    """
    from batcher.plan.types.domains import aggregate_domain_error

    schema = source.available_schema()
    if schema is None:
        return
    for spec in aggregates:
        expr = spec.agg.input
        if expr is None:
            continue
        dt = infer_type(expr, schema)
        if dt is None:
            continue
        problem = aggregate_domain_error(spec.agg.func, _input_label(expr, spec.alias), widen(dt))
        if problem is not None:
            raise PlanError(problem)


def _input_label(expr, alias: str) -> str:
    """How to name an aggregate's input in an error: its column, or the output it feeds."""
    from batcher.plan.expr_ir import Col

    return f"column {expr.name!r}" if isinstance(expr, Col) else f"the input to {alias!r}"


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
        available = available_column_set(self.input)
        for key in self.group_keys:
            _validate_refs(key.expr, available, what=f"group_by key {key.alias!r}")
        for spec in self.aggregates:
            if spec.agg.func not in AGG_FNS:
                raise PlanError(
                    f"unknown aggregate function {spec.agg.func!r} for {spec.alias!r}; "
                    f"expected one of the tags in plan/ir_tags.py::AGG_FNS"
                )
            if spec.agg.input is not None:
                _validate_refs(spec.agg.input, available, what=f"aggregate {spec.alias!r}")
        _validate_agg_input_types(self.input, self.aggregates)
        # A `map` key reaches the engine's row encoder as an internal "Row format support
        # not yet implemented" dump, after the scan, naming neither the column nor the
        # clause. Same argument as `_validate_agg_input_types` above, for the other half of
        # a `group_by`.
        validate_key_domains(
            self.input, [(k.expr, k.alias) for k in self.group_keys], operation="group_by()"
        )
        _reject_duplicate_aliases(
            [k.alias for k in self.group_keys] + [s.alias for s in self.aggregates],
            what="group_by().agg()",
        )

    def to_ir(self) -> dict[str, Any]:
        return {
            "op": Op.AGGREGATE,
            "input": self.input.to_ir(),
            "group_keys": group_keys_ir(self.group_keys),
            "aggregates": aggregates_ir(self.aggregates),
        }

    def available_columns(self) -> list[str]:
        return [k.alias for k in self.group_keys] + [s.alias for s in self.aggregates]

    def available_schema(self) -> SchemaRef | None:
        inp = self.input.available_schema()
        if inp is None:
            return None
        # Group keys first, then aggregates — the output order `available_columns` promises.
        return SchemaRef.from_typed_fields(
            chain(
                ((key.alias, infer_type(key.expr, inp)) for key in self.group_keys),
                ((spec.alias, _agg_output_type(spec.agg, inp)) for spec in self.aggregates),
            )
        )


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
        available = available_column_set(self.input)
        for key in self.keys:
            _validate_refs(key.expr, available, what="sort key")

    def shape_ir(self) -> dict[str, Any]:
        """Every IR field but the input — see `Join.shape_ir` for why this seam exists.

        Four places rebuild a `sort` node with a per-task scan substituted: the shuffle
        sort, both Flight sort paths, and the streaming top-N driver (which also overrides
        `limit`, since its per-batch trim is not the plan's). They were each restating the
        field list; the keys at least went through the shared `sort_keys_ir`, so only a new
        field on `Sort` could drift — which is exactly how `Join` lost `strategy`.
        """
        return {
            "op": Op.SORT,
            "keys": sort_keys_ir(self.keys),
            "limit": self.limit,
        }

    def to_ir(self) -> dict[str, Any]:
        return {**self.shape_ir(), "input": self.input.to_ir()}

    def available_columns(self) -> list[str]:
        return self.input.available_columns()

    def available_schema(self) -> SchemaRef | None:
        return self.input.available_schema()
