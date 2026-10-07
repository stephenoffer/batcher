"""Row-wise and set relational logical nodes.

`Scan`, `Filter`, `Projection`/`Project`, `Limit`, `Distinct`, `Sample`, and `Union`. These
are the non-grouping operators; grouping/ordering, windowing, the row-reshaping nodes, and
the opaque Python `MapBatches` stage live in sibling modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pyarrow as pa

from batcher._internal.errors import PlanError
from batcher.plan.expr_ir import Expr
from batcher.plan.ir_specs import sort_keys_ir
from batcher.plan.ir_tags import Op
from batcher.plan.logical._setops import validate_branch_types
from batcher.plan.logical.base import (
    LogicalPlan,
    SortKeySpec,
    _reject_duplicate_aliases,
    _validate_projection_refs,
    _validate_refs,
    available_column_set,
)
from batcher.plan.schema import SchemaRef
from batcher.plan.types import infer_type, promote, widen

__all__ = [
    "Distinct",
    "Filter",
    "Limit",
    "Project",
    "Projection",
    "Sample",
    "Scan",
    "StreamingSessionWindow",
    "Union",
    "WatermarkDedup",
]


@dataclass(frozen=True, slots=True)
class Scan(LogicalPlan):
    """Read an input relation, identified by index into the supplied sources."""

    source_id: int
    schema: SchemaRef
    #: A stable name for *which relation* this reads, from `plan.source_stats.source_stats_key`
    #: — `""` for a scan over an intermediate that has no cross-run identity.
    #:
    #: `source_id` cannot serve: it is an index into *this plan's* own source list, so the
    #: first source of every query is `0`. That is why `kyber.signature` rendered every scan
    #: as the bare token `["scan"]`, and why two filters of the same shape over different
    #: tables shared one learned entry — the "scan-collision defect" that module names.
    #:
    #: Excluded from equality (`compare=False`) deliberately. Plan nodes are compared to
    #: decide whether a rewrite changed anything, and that question is about *shape*; making
    #: two structurally identical scans unequal because they read different files would
    #: perturb rule fixpoints to fix a problem that is not about rewriting. Identity is what
    #: `signature` and `content_key` ask for, and both read the field directly.
    source_key: str = field(default="", compare=False)

    def to_ir(self) -> dict[str, Any]:
        # Deliberately not on the wire. The engine is handed the bound sources positionally,
        # so `source_id` is all it needs; `source_key` exists for the planner's own keying and
        # a second copy of it in the IR would be a second, driftable source of truth.
        return {"op": Op.SCAN, "source_id": self.source_id}

    def identity_suffix(self) -> str:
        """This scan's schema — the part of its identity `to_ir()` deliberately omits.

        The engine reads types off the Arrow batches it is handed, so the schema is not on
        the wire and must not be: a second copy of the types would be a second, driftable
        source of truth. But that leaves every scan of source *n* with identical IR
        regardless of what source *n* actually is, which is a collision in
        `content_key` — and `content_key` is what `kyber.plan_cache` memoizes optimized
        plans on. See `LogicalPlan.content_key`.
        """
        return str(self.schema.arrow)

    def available_columns(self) -> list[str]:
        return self.schema.names

    def available_schema(self) -> SchemaRef | None:
        # The FFI boundary widens narrow numerics on input, so the schema the
        # engine actually produces from a scan is the widened source schema.
        fields = [pa.field(f.name, widen(f.type)) for f in self.schema.arrow]
        return SchemaRef.from_arrow(pa.schema(fields))


@dataclass(frozen=True, slots=True)
class Filter(LogicalPlan):
    """Keep rows where `predicate` is true. Preserves the input schema."""

    input: LogicalPlan
    predicate: Expr

    def __post_init__(self) -> None:
        # Validate against the INPUT's columns (predicate runs before projection).
        _validate_refs(self.predicate, available_column_set(self.input), what="filter")

    def to_ir(self) -> dict[str, Any]:
        return {
            "op": Op.FILTER,
            "input": self.input.to_ir(),
            "predicate": self.predicate.to_ir(),
        }

    def available_columns(self) -> list[str]:
        return self.input.available_columns()

    def available_schema(self) -> SchemaRef | None:
        return self.input.available_schema()


@dataclass(frozen=True, slots=True)
class Projection:
    """One output column of a `Project`: an expression bound to a name."""

    alias: str
    expr: Expr


@dataclass(frozen=True, slots=True)
class Project(LogicalPlan):
    """Produce a relation with exactly the listed output columns."""

    input: LogicalPlan
    items: tuple[Projection, ...]

    def __post_init__(self) -> None:
        available = available_column_set(self.input)
        aliases = []
        for item in self.items:
            _validate_projection_refs(item.expr, available, item.alias)
            aliases.append(item.alias)
        _reject_duplicate_aliases(aliases, what="select/with_columns")

    def to_ir(self) -> dict[str, Any]:
        return {
            "op": Op.PROJECT,
            "input": self.input.to_ir(),
            "exprs": [{"expr": item.expr.to_ir(), "alias": item.alias} for item in self.items],
        }

    def available_columns(self) -> list[str]:
        return [item.alias for item in self.items]

    def available_schema(self) -> SchemaRef | None:
        inp = self.input.available_schema()
        if inp is None:
            return None
        # One uncertain column falls the whole plan back — see `from_typed_fields`.
        return SchemaRef.from_typed_fields(
            (item.alias, infer_type(item.expr, inp)) for item in self.items
        )


@dataclass(frozen=True, slots=True)
class Limit(LogicalPlan):
    """Keep at most `n` rows after skipping `offset`."""

    input: LogicalPlan
    n: int
    offset: int = 0

    def to_ir(self) -> dict[str, Any]:
        return {
            "op": Op.LIMIT,
            "input": self.input.to_ir(),
            "n": self.n,
            "offset": self.offset,
        }

    def available_columns(self) -> list[str]:
        return self.input.available_columns()

    def available_schema(self) -> SchemaRef | None:
        return self.input.available_schema()


@dataclass(frozen=True, slots=True)
class Distinct(LogicalPlan):
    """Deduplicate rows: over every column, or over a key subset keeping one whole row.

    With no `keys` this is SQL `DISTINCT` — rows agreeing on all columns collapse, and
    there is nothing to choose between them. With `keys` it is `DISTINCT ON`: the named
    columns decide which rows collapse and the survivor still carries every other column,
    chosen by `order` (the minimum under it) or arbitrarily when `order` is empty.

    Both forms are one mergeable reduction in the engine, so the single-node, parallel and
    distributed paths schedule the same operator rather than implementing it twice.
    """

    input: LogicalPlan
    keys: tuple[str, ...] = ()
    order: tuple[SortKeySpec, ...] = ()
    limit: int | None = None

    def __post_init__(self) -> None:
        available = available_column_set(self.input)
        # A dedup key is a column name, so an unknown one is caught here rather than
        # surfacing from the engine as an index-of failure after the whole scan.
        unknown = [k for k in self.keys if k not in available]
        if unknown:
            raise PlanError(f"distinct(): unknown key column(s) {sorted(unknown)}")
        for key in self.order:
            _validate_refs(key.expr, available, what="distinct order key")
        if self.order and not self.keys:
            raise PlanError(
                "distinct() over every column has no payload to order: an ordering only "
                "chooses between rows that differ, and rows that agree on all columns do not"
            )
        from batcher.plan.logical.base import validate_dedup_keys

        validate_dedup_keys(self.input, self.keys, operation="distinct()")
        if self.limit is not None:
            if self.limit < 0:
                raise PlanError(f"distinct(): limit must not be negative, got {self.limit}")
            # The engine's early exit keeps the first `limit` distinct rows in input order,
            # which only has a meaning when every surviving row is interchangeable. A keyed
            # dedup chooses *which* row survives per key, so a later row can replace an
            # earlier survivor and no prefix of the input settles the answer.
            if self.keys:
                raise PlanError(
                    "distinct(): a limit fuses only into a whole-column DISTINCT, not "
                    "DISTINCT ON — a keyed dedup's survivor can be replaced by a later row"
                )

    def shape_ir(self) -> dict[str, Any]:
        """Every IR field but the input — see `Sort.shape_ir` for why this seam exists.

        The distributed dedup re-roots this node on the bucket its reducer holds; going
        through the shape keeps the tag and the field list stated once.
        """
        ir: dict[str, Any] = {
            "op": Op.DISTINCT,
            "keys": list(self.keys),
            "order": sort_keys_ir(self.order),
        }
        # Omitted when unset so the wire shape is byte-identical to what it was before the
        # limit existed; `bc_ir::RelOp::Distinct::limit` is `#[serde(default)]`.
        if self.limit is not None:
            ir["limit"] = self.limit
        return ir

    def to_ir(self) -> dict[str, Any]:
        return {**self.shape_ir(), "input": self.input.to_ir()}

    def as_aggregate(self):
        """This whole-row `Distinct` as the equivalent `Aggregate` — group by every column.

        DISTINCT is a group-by over all columns with no aggregate functions, which is
        what lets it reuse the mergeable aggregate wholesale: identical rows fold into
        the same group, so the same `partial → combine → finalize` serves the streaming
        fold, the distributed shuffle, and the single-node path with no distinct-specific
        state anywhere.

        The derivation lives on the node because all three callers need it and they sit
        in mutually-independent subsystems (`core` twice, `dist` once). Those subsystems
        may not import one another, so a shared helper in any of them would have to be
        copy-pasted — which is exactly what had happened. `plan` is neutral, so this is
        the one place all three can reach.

        Only the whole-row form has this equivalence. A keyed dedup is *not* a group-by:
        the surviving row carries columns the grouping does not determine, and folding
        them with per-column aggregates would build a row that was never in the input.

        Returns:
            An `Aggregate` over the same input, grouping by every available column.

        Raises:
            PlanError: If this node dedups on a key subset, or carries a fused limit.
        """
        from batcher.plan.expr_ir import Col
        from batcher.plan.logical.aggregate import Aggregate

        if self.keys:
            raise PlanError(
                "a keyed distinct is not a group-by: its surviving row carries columns the "
                "key does not determine, so there is no aggregate equivalent"
            )
        # `Aggregate` has nowhere to put the fused limit, so converting would drop the early
        # exit *and* the truncation — the same rows as an unlimited DISTINCT, silently. Raise
        # instead: every caller here reaches a path that runs the `Distinct` operator itself,
        # so a limit arriving here means the fusion rule fired somewhere it should not have.
        if self.limit is not None:
            raise PlanError(
                "a distinct carrying a fused limit has no aggregate equivalent: `Aggregate` "
                "cannot express the early exit, so the conversion would silently drop it"
            )
        keys = tuple(Projection(c, Col(c)) for c in self.input.available_columns())
        return Aggregate(self.input, keys, ())

    def available_columns(self) -> list[str]:
        return self.input.available_columns()

    def available_schema(self) -> SchemaRef | None:
        return self.input.available_schema()


@dataclass(frozen=True, slots=True)
class WatermarkDedup(LogicalPlan):
    """Watermark-bounded streaming deduplication (Spark ``dropDuplicatesWithinWatermark``).

    Keeps the first row per `subset` key seen within the event-time watermark window;
    once the watermark (``max event time - lateness``) passes a key, the key is
    forgotten so a much-later duplicate may re-appear — which is what keeps the
    seen-key state bounded. A *streaming-only* node (over a bounded source, plain
    `distinct` is exact and used instead), executed entirely by the streaming driver,
    so it is never lowered to the Rust IR.
    """

    input: LogicalPlan
    subset: tuple[str, ...]
    event_time: str
    lateness_micros: int

    def available_columns(self) -> list[str]:
        return self.input.available_columns()

    def available_schema(self) -> SchemaRef | None:
        return self.input.available_schema()


@dataclass(frozen=True, slots=True)
class StreamingSessionWindow(LogicalPlan):
    """Gap-based session windows over a stream (Spark ``session_window``).

    A session is a run of events for one key with no gap longer than `gap_micros`
    between consecutive events. Unlike a tumbling or sliding window, its bounds are not
    known in advance: every new event can extend the session it lands in, and two
    sessions can merge when an event arrives between them. That is why a bounded
    `session_window` composes cleanly out of a window function and a group-by, and a
    streaming one cannot — it has to wait.

    What it waits for is the watermark. A session whose last event is at ``t`` can still
    be extended by any event in ``(t, t + gap]``, so it is complete exactly when the
    watermark passes ``t + gap``: the watermark is the engine's promise that no event
    older than it will arrive, and a row that arrives older is dropped as late. Complete
    sessions are aggregated and emitted; the rest stay buffered. **That is the operator's
    memory bound**: rows for open sessions only, which is bounded by the key space times
    the gap rather than by the length of the stream.

    A *streaming-only* node — over a bounded source `session_window` builds the composed
    window + group-by plan instead, because there is nothing to wait for. Executed by the
    streaming driver and never lowered to the Rust IR; `aggs` are re-applied per closed
    batch through the ordinary engine, so the aggregation itself is the same code the
    bounded path runs.
    """

    input: LogicalPlan
    time_col: str
    #: The gap that separates two sessions, in microseconds.
    gap_micros: int
    partition_by: tuple[str, ...]
    #: ``(output_name, aggregate expression)`` pairs, in output order.
    aggs: tuple[tuple[str, Expr], ...]
    #: Allowed lateness from the watermark, in microseconds.
    lateness_micros: int = 0

    def to_ir(self) -> dict[str, Any]:
        raise NotImplementedError(
            "a streaming session window is executed by the streaming driver, not lowered to the IR"
        )

    def available_columns(self) -> list[str]:
        return [*self.partition_by, "session_start", "session_end", *(a for a, _ in self.aggs)]

    def available_schema(self) -> SchemaRef | None:
        return None  # the aggregate output types come from the engine, not from here


@dataclass(frozen=True, slots=True)
class TransformWithState(LogicalPlan):
    """Arbitrary keyed stateful processing over a stream (Spark ``transformWithState``).

    The escape hatch for the shapes the relational operators cannot express: sessionization
    with custom rules, a running fraud score, a state machine per device, "alert when this
    key has been silent for ten minutes". Spark calls the family ``mapGroupsWithState`` /
    ``transformWithState``; the shared idea is that a *user function* owns the state for a
    key, and the engine owns when it is called, checkpointed, and expired.

    `fn` is called once per key per micro-batch with ``(key, rows, state)`` and returns
    ``(rows_out, state_out)``:

    * `key` is the group key's values as a tuple, in `group_keys` order;
    * `rows` is that key's rows *in this micro-batch*, as one Arrow `RecordBatch`;
    * `state` is whatever the previous call returned for this key, or None the first time;
    * `rows_out` is what to emit (a `RecordBatch`, a column dict, or None for nothing);
    * `state_out` is the state to keep, or None to forget the key entirely.

    Per-key, per-micro-batch Python — not per row. That is the same bargain `map_batches`
    strikes: the iteration granularity *is* the user's chosen semantics, and everything
    around it (the scan, the shuffle into groups, the emit) stays in the engine.

    **State must be a flat mapping of scalars**, because it is checkpointed as one Arrow
    `RecordBatch` alongside the keys. Spark requires a state schema for the same reason. A
    state that cannot be expressed that way is a signal to keep the payload elsewhere and
    hold a reference in state.

    `ttl_micros` is what keeps the operator's memory bounded on an unbounded stream: a key
    whose state has not been touched for that long is dropped. ``0`` means never, which is
    correct only for a bounded key space — and is the shape `kyber.streaming
    .retains_unbounded_state` is entitled to complain about.

    A *streaming* node executed by the driver, never lowered to the Rust IR. Its mergeable
    form is a shuffle by `group_keys`: each key's state lives on exactly one worker, so the
    partitions' key sets are disjoint and `combine` is their union. The distributed runner
    does not implement that yet and the conductor refuses `distributed=True` rather than
    running different semantics (the same refusal a watermarked distributed aggregate gets).
    """

    input: LogicalPlan
    #: ``(key, rows, state) -> (rows_out, state_out)``. See the class docstring.
    fn: object
    group_keys: tuple[str, ...]
    #: The output column names. Types come from what `fn` actually returns, exactly as
    #: they do for `MapBatches` — declaring them here would be a second source of truth.
    output_columns: tuple[str, ...]
    #: Microseconds of inactivity after which a key's state is dropped; 0 = never.
    ttl_micros: int = 0

    def to_ir(self) -> dict[str, Any]:
        raise NotImplementedError(
            "transform_with_state is executed by the streaming driver, not lowered to the IR"
        )

    def available_columns(self) -> list[str]:
        return list(self.output_columns)

    def available_schema(self) -> SchemaRef | None:
        return None  # opaque: the types are whatever `fn` returns


@dataclass(frozen=True, slots=True)
class Union(LogicalPlan):
    """Concatenate relations with identical schemas (UNION ALL, or UNION if distinct)."""

    inputs: tuple[LogicalPlan, ...]
    distinct: bool = False

    def __post_init__(self) -> None:
        if self.distinct:
            from batcher.plan.logical.base import validate_dedup_keys

            for branch in self.inputs:
                validate_dedup_keys(branch, (), operation="union(distinct=True)")
        if len(self.inputs) < 1:
            raise PlanError("union requires at least one input")
        cols = self.inputs[0].available_columns()
        for other in self.inputs[1:]:
            if other.available_columns() != cols:
                raise PlanError(
                    f"union inputs must have identical columns: {cols} vs "
                    f"{other.available_columns()}. To match columns by name in any order and "
                    "null-fill a missing one, use bt.concat([a, b], how='diagonal')"
                )
        validate_branch_types([i.available_schema() for i in self.inputs], cols)

    def to_ir(self) -> dict[str, Any]:
        return {
            "op": Op.UNION,
            "inputs": [i.to_ir() for i in self.inputs],
            "distinct": self.distinct,
        }

    def available_columns(self) -> list[str]:
        return self.inputs[0].available_columns()

    def available_schema(self) -> SchemaRef | None:
        schemas = [i.available_schema() for i in self.inputs]
        if any(s is None for s in schemas):
            return None
        base = schemas[0]
        names = base.names  # branches share column names (validated at build)
        out_types: list[pa.DataType] = [base.field(n).type for n in names]
        for s in schemas[1:]:
            for idx, n in enumerate(names):
                common = promote(out_types[idx], s.field(n).type)
                if common is None:  # uncertain engine coercion → fall back
                    return None
                out_types[idx] = common
        return SchemaRef.from_arrow(
            pa.schema([pa.field(n, t) for n, t in zip(names, out_types, strict=True)])
        )


@dataclass(frozen=True, slots=True)
class Sample(LogicalPlan):
    """Randomly keep a `fraction` of rows (DataFrame ``sample``).

    Deterministic and partition-independent: a row is kept iff a stable seeded hash
    of its values falls under `fraction`, so the same rows are sampled single-node or
    distributed. Streaming and stateless; output schema equals the input's.

    The selection unit is therefore the *distinct row*, not the row: identical rows hash
    identically and are all kept or all dropped together. On an input with few distinct
    rows the realized fraction is far from the requested one — two distinct values over
    10,000 rows yield 0 at ``fraction=0.1`` and 5,000 at ``0.5``. Fixing that needs a
    per-row disambiguator in the hash, which costs a shuffle to compute and would change
    which rows every existing query samples, so it is a deliberate open trade rather than
    an oversight. `Dataset.sample` documents the user-facing consequence.
    """

    input: LogicalPlan
    fraction: float
    seed: int
    # Fixed-count mode: keep exactly `n` rows (the n smallest-hash rows, a breaker).
    # None → the streaming fraction path.
    n: int | None = None

    def __post_init__(self) -> None:
        if self.n is None and not 0.0 <= self.fraction <= 1.0:
            raise PlanError(f"sample fraction must be in [0, 1], got {self.fraction}")
        if self.n is not None and self.n < 0:
            raise PlanError(f"sample n must be non-negative, got {self.n}")

    def shape_ir(self) -> dict[str, Any]:
        """Every IR field but the input — see `Sort.shape_ir` for why this seam exists.

        The streaming fixed-count driver re-applies this node to its own running best-`n`,
        and it must carry the plan's **baked seed** rather than build a fresh one: `seed=None`
        mints a seed at plan-build, so a re-derived node would sample a different relation and
        the fold would not converge on the single-node answer.
        """
        ir: dict[str, Any] = {
            "op": Op.SAMPLE,
            "fraction": self.fraction,
            "seed": self.seed,
        }
        if self.n is not None:
            ir["n"] = self.n
        return ir

    def to_ir(self) -> dict[str, Any]:
        return {**self.shape_ir(), "input": self.input.to_ir()}

    def available_columns(self) -> list[str]:
        return self.input.available_columns()

    def available_schema(self) -> SchemaRef | None:
        return self.input.available_schema()
