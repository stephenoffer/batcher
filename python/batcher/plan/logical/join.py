"""Join logical nodes: `JoinOutputCol` and `Join`.

Equi-join of two relations — a pipeline breaker with two inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from batcher._internal.errors import PlanError
from batcher.plan.ir_tags import Op
from batcher.plan.logical.base import LogicalPlan

__all__ = ["AsofJoin", "Join", "JoinOutputCol", "WatermarkStreamJoin"]


@dataclass(frozen=True, slots=True)
class JoinOutputCol:
    """One output column of a join: which side, the source name, the output name."""

    side: str  # "left" | "right"
    name: str
    alias: str


# Physical join algorithms the engine understands. A planner hint, not a semantic
# change — every strategy yields the same relation (see `bc_ir::JoinStrategy`).
JOIN_STRATEGIES = frozenset({"hash", "broadcast", "sort_merge"})


@dataclass(frozen=True, slots=True)
class Join(LogicalPlan):
    """Equi-join of two relations. A pipeline breaker with two inputs."""

    left: LogicalPlan
    right: LogicalPlan
    left_keys: tuple[str, ...]
    right_keys: tuple[str, ...]
    join_type: str  # inner|left|right|full|semi|anti
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
        if self.direction not in ("backward", "forward"):
            raise PlanError(f"asof_join direction must be backward|forward, got {self.direction!r}")

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
