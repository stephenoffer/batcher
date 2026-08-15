"""The shared sub-document shapes of the JSON IR — group keys, aggregates, sort keys.

`ir_tags.py` owns the operator *tags*; this module owns the operator *payloads* that
more than one caller has to build. Three shapes are not private to `to_ir()`, because
the engine also exposes them as standalone mergeable primitives (`partial_aggregate`,
`combine`, `combine_finalize`, the sort/top-N entry points) that the distributed and
streaming drivers call directly, without a whole `RelOp` around them:

- a group key      → ``{"expr": ..., "alias": ...}``
- an aggregate     → ``AggExpr.to_ir(alias)``
- a sort key       → ``{"expr": ..., "descending": ..., "nulls_first": ...}``
- a task's input   → ``{"op": "scan", "source_id": N}``, and the two shapes that wrap it

Before this module those shapes were hand-rolled at fourteen call sites across
`plan`, `dist`, and `core` — including inside `Aggregate.to_ir`/`Sort.to_ir`
themselves. That made the *wire contract* something with no single definition: a
change to the Rust `serde` shape had to be mirrored by hand in fourteen places, and
missing one would desynchronize the streaming or distributed path from the batch path
while every batch test stayed green (invariant #8, and exactly the class of
silent failure `CLAUDE.md` warns about).

`plan` is neutral (layer 1), so every layer — `core`, `dist`, `kyber`, `api` — shares
these definitions rather than copying them. Batch and streaming lower an aggregate
through the *same* code, which is what makes "streaming is batch with a different
schedule" true at the wire level rather than merely aspirational.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable

    from batcher.plan.logical.aggregate import Aggregate, AggregateSpec, SortKeySpec
    from batcher.plan.logical.relational import Projection

__all__ = [
    "agg_spec_json",
    "aggregates_ir",
    "binary_task_ir",
    "group_keys_ir",
    "sort_keys_ir",
    "task_scan_ir",
    "unary_task_ir",
]


def group_keys_ir(keys: Iterable[Projection]) -> list[dict[str, Any]]:
    """Lower group-key projections to their IR fragments.

    Args:
        keys: The grouping projections, in output order.

    Returns:
        One ``{"expr": ..., "alias": ...}`` document per key.
    """
    return [{"expr": k.expr.to_ir(), "alias": k.alias} for k in keys]


def aggregates_ir(aggregates: Iterable[AggregateSpec]) -> list[dict[str, Any]]:
    """Lower aggregate specs to their IR fragments.

    Args:
        aggregates: The aggregate specs, in output order.

    Returns:
        One aggregate document per spec, each already carrying its output alias.
    """
    return [s.agg.to_ir(s.alias) for s in aggregates]


def sort_keys_ir(keys: Iterable[SortKeySpec]) -> list[dict[str, Any]]:
    """Lower sort keys to their IR fragments.

    Args:
        keys: The sort keys, in precedence order.

    Returns:
        One ``{"expr": ..., "descending": ..., "nulls_first": ...}`` document per key.
    """
    return [
        {"expr": k.expr.to_ir(), "descending": k.descending, "nulls_first": k.nulls_first}
        for k in keys
    ]


def agg_spec_json(agg: Aggregate) -> tuple[str, str]:
    """Serialize an aggregate's group keys and aggregates for the mergeable primitives.

    ``partial_aggregate``, ``combine``, ``combine_finalize``, and
    ``combine_finalize_spilling`` all take this same ``(group_keys, aggregates)`` JSON
    pair. Producing it here — rather than at each call site — is what keeps the
    single-node, streaming, and distributed folds provably the *same* aggregation.

    Args:
        agg: The aggregate node to lower.

    Returns:
        The ``(group_keys_json, aggregates_json)`` pair, ready to hand to the engine.
    """
    return (
        json.dumps(group_keys_ir(agg.group_keys)),
        json.dumps(aggregates_ir(agg.aggregates)),
    )


def task_scan_ir(source_id: int = 0) -> dict[str, Any]:
    """The IR for one of the inputs a worker's per-task plan reads.

    A distributed task does not scan a file: it scans whatever the driver handed it, and
    `bc_ir::RelOp::Scan`'s ``source_id`` is how the plan names which of those inputs to read.
    Every reducer, broadcast task and spilling breaker substitutes one of these for the
    original child before shipping the plan.

    It was written out as a bare ``{"op": "scan", "source_id": 0}`` literal at **twenty** sites
    across `dist`, which is the one thing the module docstring above says must not happen to a
    wire shape: three tag strings restated twenty times, in the package whose defining risk is
    a worker's plan disagreeing with the engine's contract. `ir_tags.Op.SCAN` existed and none
    of them used it.

    Args:
        source_id: Which of the task's inputs to read. `0` is the only input of a unary
            operator, and the left input of a binary one.

    Returns:
        The scan node's IR.
    """
    from batcher.plan.ir_tags import Op

    return {"op": Op.SCAN, "source_id": source_id}


def unary_task_ir(node: Any) -> dict[str, Any]:
    """A unary operator's own shape with its child replaced by the task's input.

    Built from the node's `shape_ir()` rather than by listing its fields, so a field added to
    the operator crosses the cluster without anyone remembering to add it here — the property
    the per-node copies of this were each written to preserve, and the reason to keep it in one
    place now.

    Args:
        node: The plan node, which must expose `shape_ir()` and take an ``input``.

    Returns:
        The per-task IR document.
    """
    return {**node.shape_ir(), "input": task_scan_ir(0)}


def binary_task_ir(node: Any) -> dict[str, Any]:
    """A binary operator's own shape with both children replaced by the task's inputs.

    The join-shaped counterpart of `unary_task_ir`: left is source 0, right is source 1. That
    ordering is the contract between the driver, which decides what to hand the task in which
    slot, and the plan the task runs — so it is stated once here rather than at each of the
    five call sites that used to state it.

    Args:
        node: The plan node, which must expose `shape_ir()` and take ``left``/``right``.

    Returns:
        The per-task IR document.
    """
    return {**node.shape_ir(), "left": task_scan_ir(0), "right": task_scan_ir(1)}
