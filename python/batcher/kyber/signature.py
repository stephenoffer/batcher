"""Structural plan signatures.

A signature identifies "the same operator in the same query shape" across runs,
ignoring literal values — so learned statistics (a filter's selectivity, a join's
output size) recorded on one execution apply to the next execution of the same
shape. This is the key the MetadataHub feedback loop is indexed by.
"""

from __future__ import annotations

import hashlib
import json

from batcher.plan.logical import (
    Aggregate,
    AsofJoin,
    Distinct,
    Filter,
    Join,
    Limit,
    LogicalPlan,
    MapBatches,
    Project,
    Sample,
    Scan,
    Sort,
    Union,
    Unnest,
    Unpivot,
)

__all__ = ["plan_signature"]


def plan_signature(node: LogicalPlan) -> str:
    """A stable short hash of a node's structure (literal values normalized)."""
    payload = json.dumps(_struct(node), sort_keys=True, default=str)
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def _struct(node: LogicalPlan):
    if isinstance(node, Scan):
        return ["scan"]
    if isinstance(node, Filter):
        return ["filter", _norm(node.predicate.to_ir()), _struct(node.input)]
    if isinstance(node, Project):
        return ["project", [i.alias for i in node.items], _struct(node.input)]
    if isinstance(node, Aggregate):
        return [
            "agg",
            [k.alias for k in node.group_keys],
            [(s.alias, s.agg.func) for s in node.aggregates],
            _struct(node.input),
        ]
    if isinstance(node, Join):
        return [
            "join",
            node.join_type,
            list(node.left_keys),
            list(node.right_keys),
            _struct(node.left),
            _struct(node.right),
        ]
    if isinstance(node, AsofJoin):
        return [
            "asof_join",
            node.left_on,
            node.right_on,
            list(node.left_by),
            list(node.right_by),
            node.direction,
            _struct(node.left),
            _struct(node.right),
        ]
    if isinstance(node, Sort):
        return ["sort", _struct(node.input)]
    if isinstance(node, Limit):
        return ["limit", _struct(node.input)]
    if isinstance(node, Distinct):
        return ["distinct", _struct(node.input)]
    if isinstance(node, Union):
        return ["union", [_struct(i) for i in node.inputs]]
    if isinstance(node, MapBatches):
        # The UDF's identity is part of the shape: without it every `map_batches` over the
        # same input signature collapses to one learned entry, so a filtering UDF (0.1x) and
        # an exploding one (20x) — the two ends of an AI pipeline — would share, and
        # whichever ran last would answer for the other. Exactly the scan-collision defect,
        # and the reason `MapBatches` can only be a learnable (`_CORRECTABLE`) fan-out once
        # its signature tells the UDFs apart. Best-effort by qualified name (stable across
        # runs for a named function/class); anonymous lambdas still collide, the floor.
        return ["map_batches", _udf_identity(node.fn), _struct(node.input)]
    if isinstance(node, Unnest):
        return ["unnest", node.column, node.alias, _struct(node.input)]
    if isinstance(node, Unpivot):
        return [
            "unpivot",
            list(node.index),
            list(node.on),
            node.variable_name,
            node.value_name,
            _struct(node.input),
        ]
    if isinstance(node, Sample):
        return ["sample", node.fraction, node.seed, _struct(node.input)]
    return [type(node).__name__]


def _udf_identity(fn: object) -> str:
    """A best-effort, run-stable identity for a `map_batches` UDF.

    A named function or factory class is identified by ``module.qualname`` — stable across
    processes, so a correction learned on one run applies to the next. A callable *instance*
    (the "build the model once" factory pattern) is identified by its class, since that is
    what determines its behaviour. An anonymous lambda degrades to ``<lambda>`` and may
    collide with another lambda over the same input — the unavoidable floor for a UDF with
    no name — which is still strictly better than every `map_batches` sharing one key.
    """
    qual = getattr(fn, "__qualname__", None)
    if qual is not None:
        module = getattr(fn, "__module__", "") or ""
    else:  # a callable instance: identify it by its type
        cls = type(fn)
        qual = getattr(cls, "__qualname__", cls.__name__)
        module = getattr(cls, "__module__", "") or ""
    return f"{module}.{qual}"


def _norm(ir):
    """Normalize an expression IR, replacing literal values with a placeholder."""
    if isinstance(ir, dict):
        if ir.get("e") == "lit":
            return {"e": "lit"}
        return {k: _norm(v) for k, v in ir.items()}
    if isinstance(ir, list):
        return [_norm(x) for x in ir]
    return ir
