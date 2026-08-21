"""How much of a streaming query's state to persist on any one micro-batch.

Two policies, both about *volume* rather than about driving the loop, which is why they sit
beside the engine rather than inside it:

* a **changelog** entry instead of a whole snapshot, when the operator can replay one and the
  delta is genuinely smaller — so a checkpoint's cost stops growing with the state it
  protects;
* a **multi-part** snapshot when the operator holds part of its state on disk, so a spilled
  fold is checkpointed in full without coming back into memory to do it.

Layer: `core`. It decides how to keep running under a budget it is handed; the budget itself
is `memory.streaming_state_max_bytes`, read from config, which is also what Carbonite's
policies are written against.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    import pyarrow as pa

__all__ = ["worth_a_delta", "write_state"]


def worth_a_delta(delta: pa.RecordBatch, state: pa.RecordBatch | None) -> bool:
    """Whether recording `delta` is meaningfully cheaper than rewriting the whole `state`.

    The guard that makes incremental checkpointing safe to switch on by default. A stream
    whose every micro-batch touches every group produces a delta the size of the state, so
    the chain would write `interval` state-sized files *and then* a snapshot — strictly more
    I/O than the whole-snapshot scheme it replaced. Requiring the delta to be at least twice
    as small bounds an interval's total cost at roughly two snapshots, against the `interval`
    snapshots it would otherwise pay.

    `retained_bytes` rather than `nbytes` for the same reason `RunningAggregate.nbytes` uses
    it: a bare `nbytes` raises on Arrow's view layouts, which would turn a checkpoint policy
    question into a failed query.

    Args:
        delta: This micro-batch's changelog entry.
        state: The whole running state, or None when the runner cannot report it.

    Returns:
        True when the delta should be written in place of a snapshot.
    """
    if state is None:
        return False
    from batcher.plan.types import retained_bytes

    return retained_bytes(delta) * 2 <= retained_bytes(state)


def write_state(engine: Any, snapshot: Callable[[], pa.RecordBatch | None]) -> None:
    """Persist this epoch's state — as a changelog delta when that is genuinely cheaper.

    The default was to rewrite the whole running state every micro-batch, which makes
    the checkpoint's cost grow with the state it protects. An aggregate with no
    watermark never evicts, so a query that has accumulated ten million groups wrote ten
    million rows on every trigger, forever, and the fsyncs (or the object-store PUT) sat
    on the critical path of every epoch.

    A delta is written only when **both** conditions hold, and both matter:

    * the runner offered one. `snapshot_delta` is opt-in precisely because a delta chain
      cannot express a *removal* — replaying one against a fold that evicts would
      resurrect a closed window, which is a wrong number rather than an error. Only
      `AggregateProcessor`, whose state is monotone, offers it.
    * it is at least twice as small as the whole state. Without this rule a stream whose
      every batch touches every group would write a delta the size of the state and
      then a snapshot as well, ending up *worse* off than before. With it, the interval
      costs at most one extra snapshot's worth of bytes, so the change can only help.

    A full snapshot is also forced every `checkpoint_delta_interval` deltas, because the
    chain is what recovery replays and an unbounded one turns a restart into a rerun.
    """
    from batcher.config import active_config

    interval = active_config().streaming.checkpoint_delta_interval
    delta_of = getattr(engine._runner, "snapshot_delta", None)
    if interval and delta_of is not None and engine._deltas_written < interval:
        delta = delta_of()
        if delta is not None and worth_a_delta(delta, engine._runner.snapshot_state()):
            engine._checkpoint.snapshot_state_delta(engine._batches, delta)
            engine._deltas_written += 1
            return
    # Prefer the multi-part snapshot where the runner has one: a fold that has spilled
    # holds part of its state on disk, and `snapshot_state` reports only the resident
    # half. The store writes the parts into one file, streaming, so a state larger than
    # the memory cap does not come back into memory to be checkpointed.
    parts = getattr(engine._runner, "snapshot_state_parts", None)
    streamed = parts() if parts is not None else None
    engine._checkpoint.snapshot_state(
        engine._batches, streamed if streamed is not None else snapshot()
    )
    engine._deltas_written = 0
