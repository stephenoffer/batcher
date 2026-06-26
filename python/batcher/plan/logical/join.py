"""Join logical nodes: `JoinOutputCol` and `Join`.

Equi-join of two relations — a pipeline breaker with two inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pyarrow as pa

from batcher._internal.errors import PlanError
from batcher.plan.ir_tags import Op
from batcher.plan.logical.base import LogicalPlan
from batcher.plan.schema import SchemaRef

__all__ = ["AsofJoin", "Join", "JoinOutputCol", "WatermarkStreamJoin"]


@dataclass(frozen=True, slots=True)
class JoinOutputCol:
    """One output column of a join: which side, the source name, the output name."""

    side: str  # "left" | "right"
    name: str
    alias: str


def _join_output_schema(
    left: LogicalPlan, right: LogicalPlan, output: tuple[JoinOutputCol, ...]
) -> SchemaRef | None:
    """Assemble a join's output schema from each side's inferred schema.

    Each output column takes its type from its source side (an outer join only
    relaxes nullability, not the value type), so the type carries through. Returns
    ``None`` if either side's schema is not inferable.
    """
    left_schema = left.available_schema()
    right_schema = right.available_schema()
    if left_schema is None or right_schema is None:
        return None
    fields: list[pa.Field] = []
    for o in output:
        src = left_schema if o.side == "left" else right_schema
        if not src.has(o.name):
            return None
        fields.append(pa.field(o.alias, src.field(o.name).type))
    return SchemaRef.from_arrow(pa.schema(fields))


# The join semantics the engine understands — the `join_type` wire vocabulary,
# mirroring `bc_ir`'s join kinds. The user-facing `how="outer"` is normalized to
# `"full"` and `cross_join` lowers to an `"inner"` equi-join, so neither reaches a
# node; this is the complete set a `Join` may carry.
JOIN_TYPES = frozenset({"inner", "left", "right", "full", "semi", "anti"})

# Physical join algorithms the engine understands. A planner hint, not a semantic
# change — every strategy yields the same relation (see `bc_ir::JoinStrategy`).
JOIN_STRATEGIES = frozenset({"hash", "broadcast", "sort_merge"})

# The nearest-match directions an ASOF join may search in.
ASOF_DIRECTIONS = frozenset({"backward", "forward"})


@dataclass(frozen=True, slots=True)
class Join(LogicalPlan):
    """Equi-join of two relations. A pipeline breaker with two inputs."""

    left: LogicalPlan
    right: LogicalPlan
    left_keys: tuple[str, ...]
    right_keys: tuple[str, ...]
    join_type: str  # one of JOIN_TYPES (inner|left|right|full|semi|anti)
    output: tuple[JoinOutputCol, ...]
    # Physical algorithm chosen by Kyber's SELECTION phase. Defaults to "hash"
    # (shuffle hash join); "broadcast" replicates the small build side. Both
    # produce identical results, so the engine may fall back to hash for any
    # strategy it cannot honor.
    strategy: str = "hash"

    def __post_init__(self) -> None:
        left_cols = set(self.left.available_columns())
        right_cols = set(self.right.available_columns())
        for k in self.left_keys:
            if k not in left_cols:
                raise PlanError(f"join left key {k!r} not in left columns {sorted(left_cols)}")
        for k in self.right_keys:
            if k not in right_cols:
                raise PlanError(f"join right key {k!r} not in right columns {sorted(right_cols)}")
        if len(self.left_keys) != len(self.right_keys):
            raise PlanError("join requires the same number of left and right keys")
        if self.join_type not in JOIN_TYPES:
            raise PlanError(f"unknown join type {self.join_type!r}; expected {sorted(JOIN_TYPES)}")
        if self.strategy not in JOIN_STRATEGIES:
            raise PlanError(f"unknown join strategy {self.strategy!r}; expected {JOIN_STRATEGIES}")

    def to_ir(self) -> dict[str, Any]:
        return {
            "op": Op.HASH_JOIN,
            "left": self.left.to_ir(),
            "right": self.right.to_ir(),
            "left_keys": list(self.left_keys),
            "right_keys": list(self.right_keys),
            "join_type": self.join_type,
            "output": [{"side": o.side, "name": o.name, "alias": o.alias} for o in self.output],
            "strategy": self.strategy,
        }

    def available_columns(self) -> list[str]:
        return [o.alias for o in self.output]

    def available_schema(self) -> SchemaRef | None:
        return _join_output_schema(self.left, self.right, self.output)


@dataclass(frozen=True, slots=True)
class WatermarkStreamJoin(LogicalPlan):
    """A watermark-bounded stream-stream interval inner join (Spark stream-stream join).

    Joins two streams on equality keys *and* an event-time interval
    (``|left_time - right_time| <= within``), which is what lets buffered state be
    evicted once the watermark guarantees no future match — keeping memory bounded.
    A streaming-only node executed by the driver (over bounded sources a plain `join`
    is used), so it is never lowered to the Rust IR.
    """

    left: LogicalPlan
    right: LogicalPlan
    left_keys: tuple[str, ...]
    right_keys: tuple[str, ...]
    output: tuple[JoinOutputCol, ...]
    left_time: str
    right_time: str
    within_micros: int
    lateness_micros: int

    def available_columns(self) -> list[str]:
        return [o.alias for o in self.output]

    def available_schema(self) -> SchemaRef | None:
        return _join_output_schema(self.left, self.right, self.output)


@dataclass(frozen=True, slots=True)
class AsofJoin(LogicalPlan):
    """ASOF (nearest-match) join — DataFrame ``join_asof`` / SQL ``ASOF JOIN``.

    Each left row is matched to the right row whose `on` key is nearest in
    `direction` (``"backward"``: largest right.on ≤ left.on; ``"forward"``: smallest
    ≥), within the same `by` group (exact equality). Left-style: every left row is
    emitted, with null right columns when unmatched. A pipeline breaker.
    """

    left: LogicalPlan
    right: LogicalPlan
    left_on: str
    right_on: str
    left_by: tuple[str, ...]
    right_by: tuple[str, ...]
    direction: str  # "backward" | "forward"
    output: tuple[JoinOutputCol, ...]

    def __post_init__(self) -> None:
        left_cols = set(self.left.available_columns())
        right_cols = set(self.right.available_columns())
        if self.left_on not in left_cols:
            raise PlanError(f"asof_join left_on {self.left_on!r} not in left columns")
        if self.right_on not in right_cols:
            raise PlanError(f"asof_join right_on {self.right_on!r} not in right columns")
        if len(self.left_by) != len(self.right_by):
            raise PlanError("asof_join requires the same number of left/right `by` keys")
        if self.direction not in ASOF_DIRECTIONS:
            allowed = sorted(ASOF_DIRECTIONS)
            raise PlanError(f"asof_join direction must be one of {allowed}, got {self.direction!r}")

    def to_ir(self) -> dict[str, Any]:
        return {
            "op": Op.ASOF_JOIN,
            "left": self.left.to_ir(),
            "right": self.right.to_ir(),
            "left_on": self.left_on,
            "right_on": self.right_on,
            "left_by": list(self.left_by),
            "right_by": list(self.right_by),
            "backward": self.direction == "backward",
            "output": [{"side": o.side, "name": o.name, "alias": o.alias} for o in self.output],
        }

    def available_columns(self) -> list[str]:
        return [o.alias for o in self.output]

    def available_schema(self) -> SchemaRef | None:
        return _join_output_schema(self.left, self.right, self.output)
