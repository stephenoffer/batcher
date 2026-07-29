"""Measuring the Python-UDF stages of a pipeline the engine cannot see into.

The Rust engine reports `ExecMetrics` for every operator it runs, which is where the whole
profile comes from — but a `map_batches` stage never reaches it. That left the one pipeline
shape the ML surface exists for (``read -> decode -> infer -> write``) with no per-stage
numbers at all: `stats()` refused outright, so "which stage is the bottleneck", the first
question any tuning guide asks, had no answer for a batch-inference job.

This module is the missing reporter. `StageRecorder` is a mutable sink `core` fills while it
orchestrates the UDF stages, emitting rows/wall-time/bytes per stage in **exactly the shape
the engine's `ExecMetrics` dicts use**, so the profile builder joins Python stages and
engine operators with one code path instead of two.

Stages are keyed by the position of their node in a **pre-order walk of the logical plan**,
which is the same numbering the planned-only profile assigns, so the measured stage lands on
the row the plan tree already shows. Numbering lives here, next to the recorder, because two
independently-derived numberings that drift put one stage's measurement on another stage's
row — a silent misattribution, not an error.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pyarrow as pa

    from batcher.plan.logical import LogicalPlan

__all__ = ["StageRecorder", "logical_op_ids", "logical_preorder", "metered", "stage_kind"]


def logical_preorder(plan: LogicalPlan) -> list[tuple[int, LogicalPlan]]:
    """``(depth, node)`` for every node of `plan`, pre-order (root first, then each subtree).

    The one definition of "which operator is number N" for a plan that has no engine IR,
    shaped like `walk_ir` so the two numbering schemes stay interchangeable.
    """
    from batcher.plan.visitor import children

    out: list[tuple[int, LogicalPlan]] = []
    stack: list[tuple[int, LogicalPlan]] = [(0, plan)]
    while stack:
        depth, node = stack.pop()
        out.append((depth, node))
        # Reversed, so the leftmost child is visited first once popped.
        stack.extend((depth + 1, c) for c in reversed(list(children(node))))
    return out


def logical_op_ids(plan: LogicalPlan) -> dict[int, int]:
    """``id(node) -> op_id`` for every node of `plan`, in `logical_preorder` order.

    Keyed by object identity rather than by value: two sibling `map_batches` stages running
    the same function are equal as dataclasses but are different stages, and a value key
    would merge their measurements into one row.
    """
    return {id(node): i for i, (_depth, node) in enumerate(logical_preorder(plan))}


@dataclass
class StageRecorder:
    """Per-stage measurements for one run of a UDF pipeline.

    Accumulates rather than overwrites: a stage is applied once per morsel on the streaming
    path and once per partition on the materializing path, so the stage's totals are the sum
    of its calls (rows, bytes, wall time) — the same convention the engine uses for an
    operator that ran over many morsels.

    Thread-safe because the streaming path runs each stage on its own thread; the lock is
    taken once per morsel, not per row, so it is off the hot path.
    """

    _by_op: dict[int, dict[str, Any]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(
        self,
        op_id: int,
        *,
        kind: str,
        rows_in: int,
        rows_out: int,
        elapsed_ns: int,
        result_bytes: int,
        backend: str = "",
    ) -> None:
        """Fold one application of a stage into its running totals."""
        with self._lock:
            cur = self._by_op.get(op_id)
            if cur is None:
                self._by_op[op_id] = {
                    "op_id": op_id,
                    "kind": kind,
                    "rows_in": rows_in,
                    "rows_out": rows_out,
                    "elapsed_ns": elapsed_ns,
                    "result_bytes": result_bytes,
                    "threads": 1,
                    "backend": backend,
                }
                return
            cur["rows_in"] += rows_in
            cur["rows_out"] += rows_out
            cur["elapsed_ns"] += elapsed_ns
            cur["result_bytes"] += result_bytes

    def metric_ops(self) -> list[dict[str, Any]]:
        """The recorded stages as `ExecMetrics`-shaped dicts, ordered by `op_id`.

        Returns:
            One dict per measured stage, ready to hand to `build_op_profiles`.
        """
        with self._lock:
            return [dict(self._by_op[k]) for k in sorted(self._by_op)]

    def __bool__(self) -> bool:
        """Whether anything was measured — `False` for a run that recorded no stage."""
        with self._lock:
            return bool(self._by_op)


def metered(
    gen: Iterator[pa.RecordBatch],
    recorder: StageRecorder,
    op_id: int,
    kind: str,
    backend: str = "",
) -> Iterator[pa.RecordBatch]:
    """Fold a streaming stage's produced morsels into `recorder` as they are yielded.

    The clock brackets `next(gen)` — the wall time this stage held while producing one
    morsel. On a pipelined path that is *residency*, not pure compute: when the stage's input
    queue is empty it includes the wait for upstream, so a stage fed by a slower one reads
    high. That is the honest reading of a pipeline (the stage really was occupied), and it is
    why the bottleneck call compares stages rather than trusting any single number.

    Rows *in* are not observable at this seam — a stage re-chunks its input internally — so
    the stage reports its output rows and `rows_in` is read off the node below it in the
    tree, which in a linear chain is exactly its input.

    Args:
        gen: The stage's output morsels.
        recorder: The sink to fold each morsel's measurement into.
        op_id: The stage's position in the logical plan (see `logical_op_ids`).
        kind: The operator name to show for this stage.
        backend: Where the stage ran (``"gpu"`` or empty), so a reader can tell a device
            forward from the CPU work feeding it.

    Yields:
        Each morsel of `gen`, unchanged and in order.
    """
    while True:
        started = time.perf_counter_ns()
        try:
            batch = next(gen)
        except StopIteration:
            return
        elapsed = time.perf_counter_ns() - started
        recorder.record(
            op_id,
            kind=kind,
            rows_in=0,
            rows_out=batch.num_rows,
            elapsed_ns=elapsed,
            result_bytes=batch.nbytes,
            backend=backend,
        )
        yield batch


def stage_kind(fn: object) -> str:
    """``"MapRows"`` for a per-row adapter, else ``"MapBatches"``.

    `map`/`flat_map` lower to `map_batches` over a row loop, so the engine and the plan tree
    see one operator for both. That erases the distinction the field guides put at the top of
    their list: a per-row Python call is 10-100x a vectorized batch call, and a profile that
    calls both "map_batches" cannot say which one the run paid for. The adapters mark
    themselves (`batcher_row_adapter`); nothing else needs to know.

    Args:
        fn: The stage's callable.

    Returns:
        The operator name to report for this stage.
    """
    return "MapRows" if getattr(fn, "batcher_row_adapter", False) else "MapBatches"
