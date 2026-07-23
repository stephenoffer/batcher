"""The shared sub-document shapes of the JSON IR — group keys, aggregates, sort keys.

`ir_tags.py` owns the operator *tags*; this module owns the operator *payloads* that
more than one caller has to build. Three shapes are not private to `to_ir()`, because
the engine also exposes them as standalone mergeable primitives (`partial_aggregate`,
`combine`, `combine_finalize`, the sort/top-N entry points) that the distributed and
streaming drivers call directly, without a whole `RelOp` around them:

- a group key      → ``{"expr": ..., "alias": ...}``
- an aggregate     → ``AggExpr.to_ir(alias)``
- a sort key       → ``{"expr": ..., "descending": ..., "nulls_first": ...}``

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
    "group_keys_ir",
    "sort_keys_ir",
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
