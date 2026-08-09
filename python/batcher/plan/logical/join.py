"""Join logical nodes: `JoinOutputCol`, `Join`, `AsofJoin` and `RangeJoin`.

Equi-join, nearest-match and inequality joins of two relations — each a pipeline
breaker with two inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pyarrow as pa

from batcher._internal.errors import PlanError
from batcher.plan.ir_tags import ORDERING_COMPARISONS, Op
from batcher.plan.logical.base import LogicalPlan, _reject_duplicate_aliases
from batcher.plan.schema import SchemaRef

__all__ = [
    "AsofJoin",
    "Join",
    "JoinOutputCol",
    "RangeCondition",
    "RangeJoin",
    "WatermarkStreamJoin",
]


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


def _validate_key_types(
    left: LogicalPlan,
    right: LogicalPlan,
    left_keys: tuple[str, ...],
    right_keys: tuple[str, ...],
) -> None:
    """Reject a join whose key columns have mismatched types.

    The engine's row-encoder compares keys byte-for-byte and requires each paired
    key to have the *same* Arrow type (it does not coerce ``Int64`` against
    ``Float64`` or ``Utf8``). Without this check a mismatch surfaces only at
    execution as an opaque ``RowConverter column schema mismatch`` — so we validate
    at build time when both sides' schemas are known, and stay silent otherwise.
    """
    pairs = _paired_key_types(left, right, left_keys, right_keys)
    for lk, rk, lt, rt in pairs:
        if not lt.equals(rt):
            raise PlanError(
                f"join key type mismatch: left {lk!r} is {lt} but right {rk!r} is {rt}; "
                f"cast one side so the keys share a type"
            )


def _paired_key_types(
    left: LogicalPlan,
    right: LogicalPlan,
    left_keys: tuple[str, ...],
    right_keys: tuple[str, ...],
) -> list[tuple[str, str, pa.DataType, pa.DataType]]:
    """Resolve each key pair to its (left, right) Arrow types, skipping the unknown.

    Returns only the pairs both of whose types are known. Any failure to introspect
    a side's schema yields no pairs — the validation stays silent rather than raise a
    non-``PlanError`` (a plan may legitimately have an un-inferable schema)."""
    try:
        left_schema = left.available_schema()
        right_schema = right.available_schema()
    except Exception:
        return []
    if left_schema is None or right_schema is None:
        return []
    out: list[tuple[str, str, pa.DataType, pa.DataType]] = []
    for lk, rk in zip(left_keys, right_keys, strict=True):
        if left_schema.has(lk) and right_schema.has(rk):
            out.append((lk, rk, left_schema.field(lk).type, right_schema.field(rk).type))
    return out


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
        _validate_key_types(self.left, self.right, self.left_keys, self.right_keys)
        # A join must disambiguate colliding names (via `suffix`), never silently emit
        # two columns of the same name — one would be lost when the result is
        # materialized to a name-keyed structure.
        _reject_duplicate_aliases([o.alias for o in self.output], what="join output")

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
    """A watermark-bounded stream-stream interval join (Spark stream-stream join).

    Joins two streams on equality keys *and* an event-time interval
    (``|left_time - right_time| <= within``), which is what lets buffered state be
    evicted once the watermark guarantees no future match — keeping memory bounded.
    A streaming-only node executed by the driver (over bounded sources a plain `join`
    is used), so it is never lowered to the Rust IR.

    `how` is ``"inner"`` (only matched pairs), or ``"left"`` / ``"right"`` / ``"full"``,
    where a row that reaches the end of its watermark window unmatched is emitted once,
    padded with nulls. The interval bound is what makes an outer join expressible at all
    on an unbounded stream: without it there is no moment at which "no match will ever
    arrive" becomes true.
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
    how: str = "inner"

    @property
    def emits_unmatched_left(self) -> bool:
        """Whether an unmatched left row is emitted null-padded when its window closes."""
        return self.how in ("left", "full")

    @property
    def emits_unmatched_right(self) -> bool:
        """Whether an unmatched right row is emitted null-padded when its window closes."""
        return self.how in ("right", "full")

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


# The inequalities a `RangeJoin` condition may carry. `eq` is a hash join and `ne`
# admits no ordering structure, so neither appears here — both are left to the paths
# that already handle them.


@dataclass(frozen=True, slots=True)
class RangeCondition:
    """One inequality of a `RangeJoin`, oriented ``left_key OP right_key``."""

    left_key: str
    right_key: str
    op: str  # one of ORDERING_COMPARISONS


@dataclass(frozen=True, slots=True)
class RangeJoin(LogicalPlan):
    """Inequality join on one or two range conditions. A pipeline breaker.

    ``A JOIN B ON a.x < b.y`` (and interval containment, and band joins) otherwise
    lowers to a cartesian `Join` with the predicate as a `Filter` above it, so the
    intermediate is ``|A| x |B|`` rows however few survive. This node is executed by an
    output-sensitive algorithm instead — a sorted-suffix scan for one inequality, IEJoin
    for two — so the cost tracks the *result* size, not the product of the input sizes.

    Anything else in the original predicate stays in a `Filter` above this node, which
    is what makes the rewrite a restriction of the cartesian plan rather than a
    reinterpretation of it.
    """

    left: LogicalPlan
    right: LogicalPlan
    conditions: tuple[RangeCondition, ...]
    join_type: str  # one of JOIN_TYPES
    output: tuple[JoinOutputCol, ...]

    def __post_init__(self) -> None:
        if not 1 <= len(self.conditions) <= 2:
            raise PlanError(f"range join takes one or two conditions, got {len(self.conditions)}")
        if self.join_type not in JOIN_TYPES:
            raise PlanError(f"unknown join type {self.join_type!r}; expected {sorted(JOIN_TYPES)}")
        left_cols = set(self.left.available_columns())
        right_cols = set(self.right.available_columns())
        for c in self.conditions:
            if c.op not in ORDERING_COMPARISONS:
                raise PlanError(
                    f"unknown range op {c.op!r}; expected {sorted(ORDERING_COMPARISONS)}"
                )
            if c.left_key not in left_cols:
                raise PlanError(f"range join left key {c.left_key!r} not in left columns")
            if c.right_key not in right_cols:
                raise PlanError(f"range join right key {c.right_key!r} not in right columns")
        # The engine encodes both sides of a condition with one row converter, so a
        # mismatched pair would surface at execution as an opaque encoder error.
        _validate_key_types(
            self.left,
            self.right,
            tuple(c.left_key for c in self.conditions),
            tuple(c.right_key for c in self.conditions),
        )
        _reject_duplicate_aliases([o.alias for o in self.output], what="range join output")

    def to_ir(self) -> dict[str, Any]:
        return {
            "op": Op.RANGE_JOIN,
            "left": self.left.to_ir(),
            "right": self.right.to_ir(),
            "conditions": [
                {"left_key": c.left_key, "right_key": c.right_key, "op": c.op}
                for c in self.conditions
            ],
            "join_type": self.join_type,
            "output": [{"side": o.side, "name": o.name, "alias": o.alias} for o in self.output],
        }

    def available_columns(self) -> list[str]:
        return [o.alias for o in self.output]

    def available_schema(self) -> SchemaRef | None:
        return _join_output_schema(self.left, self.right, self.output)
