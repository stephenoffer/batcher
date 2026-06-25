"""Row-wise and set relational logical nodes.

`Scan`, `Filter`, `Projection`/`Project`, `Limit`, `Distinct`, `Union`, and the
opaque `MapBatches`. These are the non-grouping operators; grouping/ordering and
windowing live in sibling modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from batcher._internal.errors import PlanError
from batcher.plan.expr_ir import Expr
from batcher.plan.ir_tags import Op
from batcher.plan.logical.base import LogicalPlan, _validate_refs
from batcher.plan.schema import SchemaRef

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
    "Unnest",
    "Unpivot",
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

    def to_ir(self) -> dict[str, Any]:
        return {
            "op": Op.PROJECT,
            "input": self.input.to_ir(),
            "exprs": [{"expr": item.expr.to_ir(), "alias": item.alias} for item in self.items],
        }

    def available_columns(self) -> list[str]:
        return [item.alias for item in self.items]


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


@dataclass(frozen=True, slots=True)
class Distinct(LogicalPlan):
    """Deduplicate rows over all columns."""

    input: LogicalPlan

    def to_ir(self) -> dict[str, Any]:
        return {"op": Op.DISTINCT, "input": self.input.to_ir()}

    def available_columns(self) -> list[str]:
        return self.input.available_columns()


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


@dataclass(frozen=True, slots=True)
class Unnest(LogicalPlan):
    """Explode a list/array column into one row per element (SQL ``UNNEST`` /
    DataFrame ``explode``).

    The named `column` is replaced in place by its element values bound to `alias`;
    every other column repeats once per element. Null and empty lists produce no
    output rows (DuckDB ``UNNEST`` semantics). Streaming and stateless.
    """

    input: LogicalPlan
    column: str
    alias: str

    def __post_init__(self) -> None:
        available = self.input.available_columns()
        if self.column not in available:
            raise PlanError(
                f"unnest column {self.column!r} not found in input columns: {available}"
            )

    def to_ir(self) -> dict[str, Any]:
        return {
            "op": Op.UNNEST,
            "input": self.input.to_ir(),
            "column": self.column,
            "alias": self.alias,
        }

    def available_columns(self) -> list[str]:
        return [self.alias if c == self.column else c for c in self.input.available_columns()]


@dataclass(frozen=True, slots=True)
class RowId(LogicalPlan):
    """Append a 0-based (plus `offset`) sequential row-index column (Polars
    ``with_row_index``).

    The index numbers rows in arrival order across the whole input via one sequential
    counter, so it is identical on the single-node and parallel paths for an
    order-preserving pipeline. Streaming.
    """

    input: LogicalPlan
    alias: str
    offset: int = 0

    def __post_init__(self) -> None:
        if self.alias in self.input.available_columns():
            raise PlanError(f"with_row_index name {self.alias!r} collides with an existing column")

    def to_ir(self) -> dict[str, Any]:
        return {
            "op": Op.ROW_ID,
            "input": self.input.to_ir(),
            "alias": self.alias,
            "offset": self.offset,
        }

    def available_columns(self) -> list[str]:
        return [self.alias, *self.input.available_columns()]


@dataclass(frozen=True, slots=True)
class Unpivot(LogicalPlan):
    """Reshape wide → long (SQL ``UNPIVOT`` / pandas ``melt`` / Polars ``unpivot``).

    Each input row becomes one row per `on` column: the `index` columns repeat, the
    `variable_name` column holds the melted column's name, and `value_name` holds its
    value. The `on` columns must share a type. Streaming and stateless.
    """

    input: LogicalPlan
    index: tuple[str, ...]
    on: tuple[str, ...]
    variable_name: str
    value_name: str

    def __post_init__(self) -> None:
        available = self.input.available_columns()
        missing = [c for c in (*self.index, *self.on) if c not in available]
        if missing:
            raise PlanError(f"unpivot columns {missing} not found in input columns: {available}")
        if not self.on:
            raise PlanError("unpivot requires at least one column in `on`")

    def to_ir(self) -> dict[str, Any]:
        return {
            "op": Op.UNPIVOT,
            "input": self.input.to_ir(),
            "index": list(self.index),
            "on": list(self.on),
            "variable_name": self.variable_name,
            "value_name": self.value_name,
        }

    def available_columns(self) -> list[str]:
        return [*self.index, self.variable_name, self.value_name]


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


@dataclass(frozen=True, slots=True)
class MapBatches(LogicalPlan):
    """Apply an arbitrary Python function to each Arrow record batch.

    This is the opaque/black-box operator (ML inference, embeddings, custom
    preprocessing). It is executed in Python — never lowered to the Rust IR — so
    compiled relational operators and black-box ML compose in one pipeline. The
    optional `output_columns` declares the result schema for downstream
    validation; if omitted, the input columns are assumed to pass through.
    """

    input: LogicalPlan
    # Either a callable `RecordBatch -> RecordBatch|Table|dict` (stateless), or a
    # zero-arg *factory*/class that builds such a callable once per worker — the
    # "load the model once, reuse across batches" pattern for GPU inference.
    fn: object
    batch_size: int | None = None
    output_columns: tuple[str, ...] | None = None
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

    def to_ir(self) -> dict[str, Any]:
        raise NotImplementedError("map_batches is executed in Python, not lowered to the engine IR")

    def available_columns(self) -> list[str]:
        if self.output_columns is not None:
            return list(self.output_columns)
        return self.input.available_columns()
