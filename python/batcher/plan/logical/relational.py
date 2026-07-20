"""Row-wise and set relational logical nodes.

`Scan`, `Filter`, `Projection`/`Project`, `Limit`, `Distinct`, `Sample`, `Union`, and
the opaque `MapBatches`. These are the non-grouping operators; grouping/ordering,
windowing, and the row-reshaping nodes live in sibling modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pyarrow as pa

from batcher._internal.errors import PlanError
from batcher.plan.expr_ir import Expr
from batcher.plan.ir_tags import Op
from batcher.plan.logical.base import LogicalPlan, _reject_duplicate_aliases, _validate_refs
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
    "Union",
    "WatermarkDedup",
]


@dataclass(frozen=True, slots=True)
class Scan(LogicalPlan):
    """Read an input relation, identified by index into the supplied sources."""

    source_id: int
    schema: SchemaRef

    def to_ir(self) -> dict[str, Any]:
        return {"op": Op.SCAN, "source_id": self.source_id}

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
        _validate_refs(self.predicate, set(self.input.available_columns()), what="filter")

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
        available = set(self.input.available_columns())
        for item in self.items:
            _validate_refs(item.expr, available, what=f"projection {item.alias!r}")
        _reject_duplicate_aliases([item.alias for item in self.items], what="select/with_columns")

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
        fields: list[pa.Field] = []
        for item in self.items:
            t = infer_type(item.expr, inp)
            if t is None:  # one uncertain column → fall back for the whole plan
                return None
            fields.append(pa.field(item.alias, t))
        return SchemaRef.from_arrow(pa.schema(fields))


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
    """Deduplicate rows over all columns."""

    input: LogicalPlan

    def to_ir(self) -> dict[str, Any]:
        return {"op": Op.DISTINCT, "input": self.input.to_ir()}

    def as_aggregate(self):
        """This `Distinct` as the equivalent `Aggregate` — group by every column.

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

        Returns:
            An `Aggregate` over the same input, grouping by every available column.
        """
        from batcher.plan.expr_ir import Col
        from batcher.plan.logical.aggregate import Aggregate

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
    # when the `fn` is not process-safe (a factory/class, a GPU `fn`, or a non-pyarrow
    # `batch_format`). No effect on the distributed path (Ray actors already isolate).
    multiprocessing: bool = False
    # Dirty-data tolerance: the maximum number of ROWS whose per-row `fn` call may raise
    # before the query fails. 0 (the default) = strict (any error propagates). When > 0, a
    # batch that raises is bisected to isolate the offending rows; a failing single row is
    # dropped (up to this budget) and the rest of the batch proceeds — so a corrupt image /
    # malformed JSON / bad record doesn't kill a long inference job (the guides' universal
    # ``max_errored_blocks`` need). Executed in Python; no IR change.
    max_errored_rows: int = 0

    def to_ir(self) -> dict[str, Any]:
        raise NotImplementedError("map_batches is executed in Python, not lowered to the engine IR")

    def available_columns(self) -> list[str]:
        if self.output_columns is not None:
            return list(self.output_columns)
        return self.input.available_columns()
