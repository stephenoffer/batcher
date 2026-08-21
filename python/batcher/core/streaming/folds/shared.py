"""Pieces both streaming folds need: the empty-aggregate identity and the state guard.

Split out of `folds.py` when the windowed fold grew a disk tier and a changelog. These are
the parts neither fold owns — the identity row a keyless aggregate owes an empty stream, and
the one budget check both are held to — so they sit below both rather than inside either.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import pyarrow as pa

from batcher._internal.native import engine
from batcher.config import active_config
from batcher.io.source import Source, iter_source
from batcher.plan.logical import Aggregate

__all__ = ["check_agg_state_bounded", "empty_global_aggregate", "streaming_state_budget"]


def empty_global_aggregate(agg: Aggregate, schema: pa.Schema) -> pa.RecordBatch | None:
    """The one row a *keyless* aggregate owes an input that had none.

    A global `count`/`sum` over zero rows still yields exactly one row — `0`, `NULL` — which
    is what SQL, DuckDB, and `collect()` all produce. The incremental fold cannot produce
    it: it skips empty batches, so with nothing to finalize it yields nothing at all and the
    stream silently disagrees with the oracle. Asking the engine for the empty-input result
    through the ordinary plan path means the identity element falls out of the mergeable
    algebra rather than being special-cased per aggregate function.

    Takes a schema rather than a source, so both the `iter_batches` driver and the
    micro-batch processor — which has a plan and no source — can reach the same answer.

    Args:
        agg: The keyless aggregate.
        schema: The input schema, to type the empty batch fed through the plan.

    Returns:
        The one-row result, or `None` if the engine produced none.
    """
    nat = engine()
    empty = pa.RecordBatch.from_pylist([], schema=schema)
    out = nat.execute_plan(json.dumps(agg.to_ir()), [[empty]], active_config().engine_config_json())
    return next((b for b in out if b.num_rows), None)


def _rebatch(result: pa.RecordBatch, batch_size: int | None) -> Iterator[pa.RecordBatch]:
    """Yield `result` whole, or sliced into `batch_size`-row chunks."""
    if batch_size is None:
        yield result
    else:
        for off in range(0, result.num_rows, batch_size):
            yield result.slice(off, batch_size)


def _read(source: Source, projection: list[str] | None) -> Iterator[pa.RecordBatch]:
    """Read `source` through the projection Kyber decided for this plan.

    Every driver in this module used to call ``source.iter_batches(None)`` — decoding *every*
    column of every message regardless of what the plan touched, while `_iter_streaming` on
    the neighbouring path already read through the pushdown. On a wide topic that is the
    dominant cost of a streaming aggregate: a `group_by("user").sum("cents")` over a
    forty-column event decoded thirty-eight columns it then discarded, per micro-batch,
    forever.

    Core does not compute the projection — the conductor asks Kyber for it and passes it in,
    keeping the decision in Kyber's lane. `iter_source` degrades safely, so a source that
    cannot narrow its read simply returns everything and the plan is unaffected.

    Args:
        source: The stream to read.
        projection: Columns the plan needs, or ``None`` to read everything.

    Returns:
        An iterator of the source's record batches, narrowed where the source can.
    """
    return iter_source(source, projection, None)


def streaming_state_budget() -> int:
    """The byte envelope a streaming operator's retained state must stay inside."""
    return active_config().memory.streaming_state_budget_bytes()


def check_agg_state_bounded(
    fold, cap: int, cause: str, *, label: str, extra_bytes: int = 0
) -> None:
    """Raise an actionable `ResourceError` when a running aggregate has outgrown `cap`.

    Streaming state is only bounded by something that *releases* it, and the two folds here
    release differently — the windowed one by an advancing watermark, the plain one not at
    all — so the diagnosis differs while the check does not. Sharing the check is what keeps
    them from drifting into one operator that guards its state and one that does not, which
    is exactly what had happened.

    `extra_bytes` is what an operator retains *beside* the fold. It exists because `update`
    output mode keeps a full copy of the last emitted result to diff against, exactly as large
    as the fold's own state — and nothing counted it. A query configured with a one-gigabyte
    cap therefore held two, the cap under-reported by 2x, and the progress record's
    `memory_used_bytes` reported half of what the operator was actually using. The failure is
    an OOM at twice the configured budget, which is precisely the kind the budget exists to
    turn into an error instead.

    Args:
        fold: The running aggregate to measure.
        cap: The byte budget.
        cause: Why the state grew, and what the user can do about it.
        label: The operator's name, quoted back in the message.
        extra_bytes: Bytes the operator retains outside the fold, counted against the cap.

    Raises:
        ResourceError: When the retained state exceeds `cap`.
    """
    size = fold.nbytes() + extra_bytes
    if size > cap:
        from batcher._internal.errors import ResourceError

        raise ResourceError(f"{label} state reached {size} bytes (cap {cap}): {cause}.")
