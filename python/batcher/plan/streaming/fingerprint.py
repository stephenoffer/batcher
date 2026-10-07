"""The fingerprint a streaming checkpoint binds its plan by, opaque nodes included.

A checkpoint refuses a stateful restart under a different plan (`checkpoint.identity`),
because restoring state folded by one computation into another resumes it from the wrong
answer. The fingerprint it compares is `LogicalPlan.content_key`, which is a hash of the
lowered IR — and a plan with an *opaque* node (`transform_with_state`, `map_batches`) has no
IR, so its key is the node's object identity and differs in every process. Such a plan used
to bind no fingerprint at all, which skipped the check for every keyed-state query: one
restarted with `group_keys=("user", "region")` over state keyed by ``("user",)`` restored
one-element keys that never match a two-element one, and the old state sat there unread.

This keys what *is* stable about an opaque node — its kind, the columns it produces, the
keys its state is grouped by, and everything beneath it, which serializes normally — and
leaves out its Python function, whose body has no stable identity across processes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from batcher.plan.logical import LogicalPlan
from batcher.plan.schema import SchemaRef
from batcher.plan.visitor import children, with_children

__all__ = ["restart_fingerprint"]

#: The fields an opaque node keys its state by (`transform_with_state`'s `group_keys`, a
#: streaming session window's `partition_by`). Restored state is looked up by these, so a
#: change to them is exactly what makes a checkpoint unrestorable. Resource knobs such as
#: a `map_batches` worker count are left out on purpose: they move no state.
_STATE_KEY_FIELDS = ("group_keys", "partition_by")


@dataclass(frozen=True, slots=True)
class _OpaqueShape(LogicalPlan):
    """An opaque node reduced to what a restart can check, so a parent serializes over it.

    Reports the original's columns and schema, so the parent rebuilt onto it validates
    exactly as it did against the real node. It is never executed or sent to the engine.
    """

    kind: str
    columns: tuple[str, ...]
    #: JSON of the node's `_STATE_KEY_FIELDS`. A string, not a tuple, so
    #: the visitor can never mistake it for a child-bearing field.
    shape: str
    inputs: tuple[LogicalPlan, ...]
    schema: SchemaRef | None = None

    def to_ir(self) -> dict[str, Any]:
        return {
            "opaque": self.kind,
            "columns": list(self.columns),
            "shape": self.shape,
            "inputs": [i.to_ir() for i in self.inputs],
        }

    def available_columns(self) -> list[str]:
        return list(self.columns)

    def available_schema(self) -> SchemaRef | None:
        return self.schema


def _stable(node: LogicalPlan) -> LogicalPlan:
    """`node` with every opaque node beneath (and including) it replaced by its shape."""
    if node.ir_json() is not None:
        return node
    rebuilt = with_children(node, [_stable(child) for child in children(node)])
    if rebuilt.ir_json() is not None:
        return rebuilt  # opaque only through a child, which is now a shape
    shape = {name: list(getattr(node, name, None) or ()) for name in _STATE_KEY_FIELDS}
    return _OpaqueShape(
        kind=type(node).__name__,
        columns=tuple(node.available_columns()),
        shape=json.dumps(shape),
        inputs=tuple(children(rebuilt)),
        schema=node.available_schema(),
    )


def restart_fingerprint(plan: LogicalPlan) -> str:
    """The plan's fingerprint for checkpoint compatibility, stable across processes.

    Equal to `plan.content_key()` for a plan with no opaque node, so a checkpoint written
    before opaque plans were fingerprinted still binds to the same value.

    Args:
        plan: The streaming plan the checkpoint records.

    Returns:
        A content hash of the plan, with each opaque node keyed by its kind, output
        columns, group keys, and inputs, and never by its Python function.
    """
    return _stable(plan).content_key()
