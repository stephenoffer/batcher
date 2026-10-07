"""The opaque Python stage: `MapBatches`, behind `map_batches`, `map`, `flat_map` and `filter(fn)`.

Its own module because it is the one node whose output the plan cannot derive: every other
node computes its schema from its input and its expressions, while this one runs user code.
What it *can* know is what the user declared -- the output names, and optionally a full
`pyarrow.Schema` -- and that declaration is what keeps `Dataset.schema` from running the
callback to find out.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pyarrow as pa

from batcher.plan.logical.base import LogicalPlan
from batcher.plan.schema import SchemaRef

__all__ = ["MapBatches"]


@dataclass(frozen=True, slots=True)
class MapBatches(LogicalPlan):
    """Apply an arbitrary Python function to each Arrow record batch.

    This is the opaque/black-box operator (ML inference, embeddings, custom
    preprocessing). It is executed in Python — never lowered to the Rust IR — so
    compiled relational operators and black-box ML compose in one pipeline. The
    optional `output_columns` declares the result columns for downstream
    validation; if omitted, the input columns are assumed to pass through. A declared
    `output_schema` goes one step further and fixes their types as well, which is what lets
    `available_schema()` answer without running the `fn`, and what every output batch is cast
    to (so a stage that saw no rows still returns its declared types rather than `null`).

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
    # The full declared output schema (names, types, nullability), when the caller passed
    # `output_columns` as a `pyarrow.Schema`. `output_columns` then holds its names, so every
    # name-only consumer is unchanged. None = only names (or nothing) are known, and the
    # stage's types are learned by running it.
    output_schema: pa.Schema | None = None
    # Opt-in quarantine for `max_errored_rows`: a row the error budget would DROP is emitted
    # instead, its output columns null (except preserved columns, which carry their input
    # value) and this extra string column holding ``"<ExcType>: <message>"``. Every other row
    # carries null here. The row is still charged against `max_errored_rows`. None = drop.
    error_column: str | None = None
    # Whether `fn` is the per-row adapter behind `map`/`flat_map`, which calls the user's
    # function once per row and so never on an empty batch. Read by `Dataset.schema`'s probe:
    # a zero-row probe cannot learn such a stage's output, so it hands the stage one row.
    per_row: bool = False

    def to_ir(self) -> dict[str, Any]:
        raise NotImplementedError("map_batches is executed in Python, not lowered to the engine IR")

    def available_columns(self) -> list[str]:
        names = list(self.output_columns) if self.output_columns is not None else None
        if names is None:
            names = self.input.available_columns()
        return names if self.error_column is None else [*names, self.error_column]

    def declared_output(self) -> pa.Schema | None:
        """The schema every output batch is conformed to, or `None` when only `fn` knows it.

        The declared `output_schema` with each type normalized the way the engine boundary
        normalizes it (`plan.types.widen`: a narrow integer or float widens, a dictionary
        decodes), plus the `error_column` when one is kept. Normalized because every operator
        above this stage reads its input as already normalized; a stage that kept a declared
        ``int32`` would report ``int32`` through a `select` the engine returns as ``int64``.
        A quarantined row nulls every column the stage does not preserve, so with an
        `error_column` each declared field is also relaxed to nullable.
        """
        from batcher.plan.types import widen

        schema = self.output_schema
        if schema is None:
            return None
        relax = self.error_column is not None
        fields = [f.with_type(widen(f.type)).with_nullable(f.nullable or relax) for f in schema]
        if relax:
            fields.append(pa.field(self.error_column, pa.string()))
        return pa.schema(fields, schema.metadata)

    def available_schema(self) -> SchemaRef | None:
        """The declared output schema, or the input's when the stage provably keeps it.

        Two cases are known without running the `fn`. A declared `output_schema` is the
        contract every output batch is cast to (`declared_output`). And a stage
        that declares no output columns but preserves *every* input column -- the callable
        form of `filter`, which only drops rows -- returns exactly its input's schema.
        Anything else is the `fn`'s choice.
        """
        declared = self.declared_output()
        if declared is not None:
            return SchemaRef(declared)
        if self.output_columns is not None or not self._keeps_input_schema():
            return None
        inferred = self.input.available_schema()
        if inferred is None or self.error_column is None:
            return inferred
        error = pa.field(self.error_column, pa.string())
        return SchemaRef(pa.schema([*inferred.arrow, error], inferred.arrow.metadata))

    def _keeps_input_schema(self) -> bool:
        """Whether every input column is declared preserved (the row-dropping `filter` form)."""
        if not self.preserves_columns:
            return False
        try:
            inputs = self.input.available_columns()
        except Exception:  # an un-inferable input names nothing, so nothing is provably kept
            return False
        return set(inputs) <= set(self.preserves_columns)
