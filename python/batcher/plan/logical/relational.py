"""Row-wise and set relational logical nodes.

`Scan`, `Filter`, `Projection`/`Project`, `Limit`, `Distinct`, `Sample`, `Union`, and
the opaque `MapBatches`. These are the non-grouping operators; grouping/ordering,
windowing, and the row-reshaping nodes live in sibling modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pyarrow as pa

from batcher._internal.errors import PlanError
from batcher.plan.expr_ir import Expr
from batcher.plan.ir_specs import sort_keys_ir
from batcher.plan.ir_tags import Op
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
    "MapBatches",
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

    def to_ir(self) -> dict[str, Any]:
        ir: dict[str, Any] = {
            "op": Op.DISTINCT,
            "input": self.input.to_ir(),
            "keys": list(self.keys),
            "order": sort_keys_ir(self.order),
        }
        # Omitted when unset so the wire shape is byte-identical to what it was before the
        # limit existed; `bc_ir::RelOp::Distinct::limit` is `#[serde(default)]`.
        if self.limit is not None:
            ir["limit"] = self.limit
        return ir

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
        if len(self.inputs) < 1:
            raise PlanError("union requires at least one input")
        cols = self.inputs[0].available_columns()
        for other in self.inputs[1:]:
            if other.available_columns() != cols:
                raise PlanError(
                    "union inputs must have identical columns: "
                    f"{cols} vs {other.available_columns()}"
                )

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

    def to_ir(self) -> dict[str, Any]:
        ir: dict[str, Any] = {
            "op": Op.SAMPLE,
            "input": self.input.to_ir(),
            "fraction": self.fraction,
            "seed": self.seed,
        }
        if self.n is not None:
            ir["n"] = self.n
        return ir

    def available_columns(self) -> list[str]:
        return self.input.available_columns()

    def available_schema(self) -> SchemaRef | None:
        return self.input.available_schema()


@dataclass(frozen=True, slots=True)
class MapBatches(LogicalPlan):
    """Apply an arbitrary Python function to each Arrow record batch.

    This is the opaque/black-box operator (ML inference, embeddings, custom
    preprocessing). It is executed in Python — never lowered to the Rust IR — so
    compiled relational operators and black-box ML compose in one pipeline. The
    optional `output_columns` declares the result schema for downstream
    validation; if omitted, the input columns are assumed to pass through.

    `input_columns` is the other half of that contract, and it is what lets the optimizer
    see *into* the black box far enough to be useful. Without it the plan must assume the
    `fn` may read any column of its input, so projection pushdown gives up and the scan
    reads the whole table: an embedding stage over one column of a 41-column Parquet file
    read all 41. Declaring the columns the `fn` actually reads turns that into a one-column
    scan, and lets column lineage narrow to the truth instead of "everything derives from
    everything". It is opt-in precisely because getting it wrong is a wrong answer, not a
    slow one — an undeclared column the `fn` secretly reads would be pruned away beneath it.
    """

    input: LogicalPlan
    # Either a callable `RecordBatch -> RecordBatch|Table|dict` (stateless), or a
    # zero-arg *factory*/class that builds such a callable once per worker — the
    # "load the model once, reuse across batches" pattern for GPU inference.
    fn: object
    batch_size: int | None = None
    output_columns: tuple[str, ...] | None = None
    # The columns `fn` reads. None = unknown, so the optimizer must keep every column alive
    # (the safe default). When declared, projection pushdown prunes the scan to these columns
    # (plus whatever the operators *above* still need), and lineage attributes the outputs to
    # these inputs only. Declaring a column the `fn` does not read is merely wasteful;
    # OMITTING one it does read is a correctness bug — the column gets pruned out from under it.
    input_columns: tuple[str, ...] | None = None
    # The columns `fn` passes through UNCHANGED — same name, same value, in every output row.
    # None = unknown, so the optimizer must assume `fn` may rewrite any column and no predicate
    # can ever move below the UDF (the safe default). When a column is declared here, a `Filter`
    # whose predicate reads only preserved columns is pushed *below* the UDF, so the model runs
    # on the rows that survive the filter instead of every row — filtering 60% of the rows
    # before GPU inference saves 60% of the GPU work. This is the mirror of `input_columns`:
    # that field says only what `fn` READS, which cannot justify the pushdown (a column the fn
    # reads it may still overwrite). Preservation is the stronger claim, and it is opt-in for
    # the same reason `input_columns` is — declaring a column the `fn` actually rewrites is a
    # WRONG ANSWER, not a slow one: rows the predicate would drop on the *rewritten* value are
    # dropped on the *input* value instead, silently changing the result.
    preserves_columns: tuple[str, ...] | None = None
    # Concurrent workers for the per-batch call (>1 overlaps GIL-releasing model
    # inference across cores; the GIL serializes pure-Python `fn`s).
    num_workers: int = 1
    # GPUs to reserve per distributed worker/actor (Ray resource). 0 = CPU only.
    num_gpus: float = 0.0
    # Distributed actor-pool size: when set (or when a factory `fn` needs building
    # once per worker), the distributed path runs long-lived actors that each build
    # the model once and stream partitions through it. An `int` fixes the pool size;
    # a `(min, max)` tuple autoscales the pool to the workload within those bounds.
    concurrency: int | tuple[int, int] | None = None
    # The object `fn` receives and returns per batch: "pyarrow" (RecordBatch),
    # "numpy" ({col: ndarray}), "pandas" (DataFrame), or "torch" ({col: tensor}).
    # The Arrow boundary is unchanged — conversion happens around the call only.
    batch_format: str = "pyarrow"
    # Optional GPU model to pin GPU actors/tasks to (a `ray.util.accelerators` name
    # like "NVIDIA_A100"); None lets Ray pick any GPU.
    accelerator_type: str | None = None
    # Custom Ray resources per worker, as `((name, amount), ...)`. `num_gpus` only covers
    # what Ray calls the `GPU` resource (NVIDIA/AMD/Intel/MetaX); a TPU, Trainium
    # (`neuron_cores`), Gaudi (`HPU`), or an operator's own on-prem resource is named
    # instead. A tuple so the node stays hashable/frozen like every other field here.
    resources: tuple[tuple[str, float], ...] = ()
    # Optional estimate of the model's memory footprint in GB. Lets the resource layer
    # budget host RAM per worker (so loading the model into many workers can't OOM the
    # node) and VRAM-pack the GPU fraction; lets Kyber's cost model scale the
    # inference cost by model size. 0.0 = unknown (no budgeting).
    model_memory_gb: float = 0.0
    # Run the per-batch calls across `num_workers` *processes* instead of threads, so a
    # CPU-bound pure-Python `fn` (which the GIL would serialize across threads) uses
    # multiple cores on a single node. Opt-in; the local executor falls back to threads
    # when the `fn` is not process-safe (a factory/class, a GPU `fn`, or one that cannot be
    # serialized to a child). Any `batch_format` is fine — the conversion runs in the child.
    # No effect on the distributed path (Ray actors already isolate).
    multiprocessing: bool = False
    # Dirty-data tolerance: the maximum number of ROWS whose per-row `fn` call may raise
    # before the query fails. 0 (the default) = strict (any error propagates). When > 0, a
    # batch that raises is bisected to isolate the offending rows; a failing single row is
    # dropped (up to this budget) and the rest of the batch proceeds — so a corrupt image /
    # malformed JSON / bad record doesn't kill a long inference job (the guides' universal
    # ``max_errored_blocks`` need). Executed in Python; no IR change.
    max_errored_rows: int = 0
    # Transient-failure resilience for a flaky/external `fn` (an LLM API, a vector-DB upsert, a
    # model that intermittently OOMs) — the ML-inference workload Batcher targets. A batch whose
    # `fn` raises a retryable error is retried up to `max_retries` times with exponential backoff
    # (`retry_backoff_s * 2**attempt`), before the failure falls through to `max_errored_rows`.
    # 0 (the default) = no retry, so a real bug on clean data still fails fast on the first call.
    max_retries: int = 0
    retry_backoff_s: float = 0.5
    # The exception types worth retrying; empty = retry any `Exception` when `max_retries > 0`.
    # A non-retryable bug (a `TypeError` from a schema mismatch) should not burn the retry budget,
    # so restrict retries to the transient errors an external service actually raises.
    retry_on: tuple[type[BaseException], ...] = ()
    # Wall-clock ceiling (seconds) for a single per-batch `fn` call; 0 = no timeout. A call that
    # exceeds it raises `TimeoutError` (retried like any transient error, then charged to the
    # error budget). Guards a query against a hung external call — Python cannot preempt a
    # running call, so the timed-out call's thread is abandoned and its result discarded, not
    # killed. Applies to the thread/sequential paths (where a flaky I/O-bound `fn` runs), not the
    # multiprocessing path (reserved for CPU-bound pure-Python `fn`s). On the async path
    # (`async def fn`) the timeout instead *cancels* the pending coroutine at its next await.
    timeout_s: float = 0.0
    # Max in-flight batches for an async (`async def`) `fn`: an I/O-bound inference/enrichment
    # `fn` awaits a remote service, so many batches' awaits overlap on ONE event loop bounded by
    # this semaphore — the LLM-API concurrency pattern, without a thread per request. 0 = an
    # adaptive default. Ignored for a synchronous `fn` (which uses the thread/process paths).
    max_concurrency: int = 0

    def to_ir(self) -> dict[str, Any]:
        raise NotImplementedError("map_batches is executed in Python, not lowered to the engine IR")

    def available_columns(self) -> list[str]:
        if self.output_columns is not None:
            return list(self.output_columns)
        return self.input.available_columns()
