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
        return ["map_batches", _struct(node.input)]
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


def _norm(ir):
    """Normalize an expression IR, replacing literal values with a placeholder."""
    if isinstance(ir, dict):
        if ir.get("e") == "lit":
            return {"e": "lit"}
        return {k: _norm(v) for k, v in ir.items()}
    if isinstance(ir, list):
        return [_norm(x) for x in ir]
    return ir
