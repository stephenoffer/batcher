"""A changelog for the operator that *removes* state, and the one integer that makes it work.

A changelog records what a micro-batch folded in and has no way to say what was taken out,
which is why the evicting operators kept whole snapshots: replaying a chain would resurrect
the windows eviction had already emitted, and re-emitting them under a new batch id is a
duplicate no sink's by-batch-id idempotency absorbs.

A windowed aggregate qualifies anyway, because its removal is not arbitrary. Eviction drops
every window whose start is at or below a threshold, on a totally ordered axis — so what it
removes is always a **prefix**, and a prefix is fully described by its upper bound. That one
integer travels in each entry's metadata; replay combines the partials and re-applies it.

`test_a_chain_does_not_resurrect_windows_already_emitted` is the load-bearing test, and
`test_without_the_eviction_bound_the_chain_does_resurrect_them` is its negative control — the
second is what stops the first from passing for the wrong reason.
"""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

import batcher as bt
from batcher.core.streaming.folds import _EVICTED_META, _window_key, _WindowedAggFold

pytestmark = pytest.mark.integration

_BASE = dt.datetime(2024, 1, 1)
_SCHEMA = pa.schema([("ts", pa.timestamp("us")), ("k", pa.string()), ("v", pa.int64())])


def _at(second: int, keys: int = 5) -> pa.RecordBatch:
    """One micro-batch whose rows all carry event time `second`."""
    return pa.record_batch(
        {
            "ts": pa.array([_BASE + dt.timedelta(seconds=second)] * keys, type=pa.timestamp("us")),
            "k": pa.array([f"k{i}" for i in range(keys)]),
            "v": pa.array([1] * keys, type=pa.int64()),
        },
        schema=_SCHEMA,
    )


def _fold() -> _WindowedAggFold:
    """A ten-second windowed sum with five seconds of allowed lateness."""
    plan = (
        bt.from_batches(lambda: iter([_at(0)]), _SCHEMA, bounded=False)
        .with_watermark("ts", "5 seconds")
        .group_by(w=bt.window(bt.col("ts"), "10 seconds"), k=bt.col("k"))
        .agg(total=bt.col("v").sum())
    )._plan
    return _WindowedAggFold(plan, _window_key(plan))


def _windows(batches) -> list[str]:
    """The distinct window starts appearing in `batches`, as clock strings."""
    seen: set[str] = set()
    for batch in batches:
        seen |= {str(value)[-8:] for value in batch.column("w").to_pylist()}
    return sorted(seen)


def _drive(pushes: int = 10, base_after: int = 4):
    """Run a fold, snapshotting after `base_after` pushes and recording deltas afterwards.

    Returns:
        ``(base, deltas, emitted_after_the_base)`` — the shape a crash mid-chain leaves.
    """
    fold = _fold()
    for step in range(base_after):
        fold.push(_at(step * 4))
    base = fold.state()
    deltas: list[pa.RecordBatch] = []
    emitted: list[pa.RecordBatch] = []
    for step in range(base_after, pushes):
        emitted.extend(fold.push(_at(step * 4)))
        entry = fold.take_delta()
        if entry is not None:
            deltas.append(entry)
    return base, deltas, emitted


def _strip_bound(batch: pa.RecordBatch) -> pa.RecordBatch:
    """The same batch without its eviction bound — the negative control."""
    metadata = dict(batch.schema.metadata or {})
    metadata.pop(_EVICTED_META, None)
    return batch.replace_schema_metadata(metadata)


# --- the invariant ---------------------------------------------------------


def test_a_chain_does_not_resurrect_windows_already_emitted():
    """Replaying partials rebuilds the *pre*-eviction state; the bound turns it back into
    the open set. Without that, every window emitted between the base snapshot and the crash
    comes back and is emitted a second time under a different batch id."""
    base, deltas, emitted_before = _drive()
    assert deltas and _windows(emitted_before), "the run emitted nothing, so this is vacuous"

    revived = _fold()
    revived.restore_parts([base, *deltas])
    emitted_after = revived.push(_at(40))

    resurrected = set(_windows(emitted_before)) & set(_windows(emitted_after))
    assert not resurrected, f"already-emitted windows came back: {sorted(resurrected)}"


def test_without_the_eviction_bound_the_chain_does_resurrect_them():
    """The negative control. If this ever stops failing to resurrect, the bound has stopped
    doing anything and the test above is passing for the wrong reason."""
    base, deltas, emitted_before = _drive()

    revived = _fold()
    revived.restore_parts([_strip_bound(base), *(_strip_bound(d) for d in deltas)])
    emitted_after = revived.push(_at(40))

    assert set(_windows(emitted_before)) & set(_windows(emitted_after)), (
        "stripping the bound no longer resurrects anything, so it is not load-bearing"
    )


def test_a_chain_rebuilds_the_same_open_state_as_the_live_fold():
    """What survives the replay must be exactly what the live fold still held."""
    fold = _fold()
    for step in range(4):
        fold.push(_at(step * 4))
    base = fold.state()
    deltas = []
    for step in range(4, 10):
        fold.push(_at(step * 4))
        entry = fold.take_delta()
        if entry is not None:
            deltas.append(entry)

    revived = _fold()
    revived.restore_parts([base, *deltas])

    def flushed(target) -> list[tuple]:
        result = target.flush()
        if result is None:
            return []
        return sorted(
            zip(
                result.column("w").to_pylist(),
                result.column("k").to_pylist(),
                result.column("total").to_pylist(),
                strict=True,
            )
        )

    assert flushed(revived) == flushed(fold)


def test_the_bound_is_a_no_op_on_a_whole_snapshot():
    """A snapshot's state is already post-eviction, which is why restore applies the bound
    unconditionally instead of branching on which kind of checkpoint it is reading."""
    fold = _fold()
    for step in range(8):
        fold.push(_at(step * 4))
    snapshot = fold.state()
    assert (snapshot.schema.metadata or {}).get(_EVICTED_META), "no bound was recorded"

    revived = _fold()
    revived.restore_parts([snapshot])
    assert revived._fold.state().num_rows == snapshot.num_rows


def test_reading_a_delta_consumes_it():
    """The engine commits epochs that never push — an end-of-drain marker, an idle trigger —
    and an entry left in place is written again under a new batch id and replayed twice."""
    fold = _fold()
    fold.push(_at(0))
    assert fold.take_delta() is not None
    assert fold.take_delta() is None


def test_an_epoch_that_folded_nothing_has_no_entry():
    fold = _fold()
    fold.push(_at(0).slice(0, 0))
    assert fold.take_delta() is None
