"""Spillable windowed state: the query that used to die at the cap now finishes.

A watermarked windowed aggregate bounds its state by evicting closed windows. That is the
right bound and not always a small one — the open set is `allowed_lateness / hop` windows
wide with a row per group key — so a high-cardinality stream reaches
`memory.streaming_state_max_bytes` while behaving exactly as designed, and what happened
there was a `ResourceError`.

The disk tier under that bound exploits the one thing this operator has that a general state
store does not: the watermark only moves forward, so windows are evicted in increasing order
and a spilled window is read back exactly once. Correctness rests on the same property as the
rest of the streaming state — the runs hold *partial* state and `combine` is associative and
commutative (invariant #7), so where a window's rows happened to sit cannot change what they
fold to.

The load-bearing test is `test_a_spilled_run_matches_the_unspilled_answer_exactly`, and it is
paired with `test_the_same_query_without_the_spill_tier_raises` so that the first cannot pass
vacuously: without the second, a cap that never actually triggered a spill would look like a
clean pass.
"""

from __future__ import annotations

import dataclasses
import datetime as dt

import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import ResourceError
from batcher.config import active_config, config_context
from batcher.core.streaming import folds

pytestmark = pytest.mark.integration

_BASE = dt.datetime(2024, 1, 1)
_SCHEMA = pa.schema([("ts", pa.timestamp("us")), ("k", pa.string()), ("v", pa.int64())])

#: Peak resident state for the stream below is ~2.8 kB, so this cap forces several spills
#: while leaving the newest windows resident. Pinned as a constant because the point of the
#: pairing above is that it is *below* the peak; a cap above it tests nothing.
_TIGHT_CAP = 1200


def _batches(windows: int = 40, keys: int = 25):
    """A stream advancing event time two seconds per batch, `keys` groups per batch."""
    for window in range(windows):
        yield pa.record_batch(
            {
                "ts": pa.array(
                    [_BASE + dt.timedelta(seconds=window * 2)] * keys, type=pa.timestamp("us")
                ),
                "k": pa.array([f"key-{i:04d}" for i in range(keys)]),
                "v": pa.array(list(range(keys)), type=pa.int64()),
            },
            schema=_SCHEMA,
        )


def _batch(window: int, keys: int = 25) -> pa.RecordBatch:
    """One micro-batch of the stream above, at `window`'s event time."""
    return pa.record_batch(
        {
            "ts": pa.array(
                [_BASE + dt.timedelta(seconds=window * 2)] * keys, type=pa.timestamp("us")
            ),
            "k": pa.array([f"key-{i:04d}" for i in range(keys)]),
            "v": pa.array(list(range(keys)), type=pa.int64()),
        },
        schema=_SCHEMA,
    )


def _at_second(second: int, keys: int = 25) -> pa.RecordBatch:
    """One micro-batch whose rows all carry event time `second`."""
    return pa.record_batch(
        {
            "ts": pa.array([_BASE + dt.timedelta(seconds=second)] * keys, type=pa.timestamp("us")),
            "k": pa.array([f"key-{i:04d}" for i in range(keys)]),
            "v": pa.array(list(range(keys)), type=pa.int64()),
        },
        schema=_SCHEMA,
    )


def _window_batch(micros: int) -> pa.RecordBatch:
    """A one-row spill payload whose window start is `micros`, for store-level tests."""
    return pa.record_batch(
        {
            "w": pa.array([_BASE + dt.timedelta(microseconds=micros)], type=pa.timestamp("us")),
            "v": pa.array([micros], type=pa.int64()),
        }
    )


def _windowed():
    """The watermarked windowed aggregate under test, over an unbounded source."""
    return (
        bt.from_batches(_batches, _SCHEMA, bounded=False)
        .with_watermark("ts", "30 seconds")
        .group_by(w=bt.window(bt.col("ts"), "10 seconds"), k=bt.col("k"))
        .agg(total=bt.col("v").sum())
    )


def _capped(cap: int):
    """A config context with the streaming state cap set to `cap` bytes."""
    config = active_config()
    return config_context(
        config.replace(memory=dataclasses.replace(config.memory, streaming_state_max_bytes=cap))
    )


def _run(cap: int) -> list[tuple]:
    """Drive the windowed aggregate under `cap` and return its rows, sorted."""
    with _capped(cap):
        table = pa.Table.from_batches(list(_windowed().iter_batches())).to_pydict()
    return sorted(zip(table["w"], table["k"], table["total"], strict=True))


@pytest.fixture
def spill_probe(monkeypatch):
    """Record what the fold actually spilled, so a test cannot pass without spilling."""
    seen = {"passes": 0, "runs": 0, "rows": 0}
    original = folds._WindowedAggFold._spill_cold_windows

    def observed(self):
        original(self)
        if self._spill is not None:
            seen["passes"] += 1
            seen["runs"] = max(seen["runs"], len(self._spill))
            seen["rows"] = max(seen["rows"], self._spill.rows())

    monkeypatch.setattr(folds._WindowedAggFold, "_spill_cold_windows", observed)
    return seen


# --- the invariant ---------------------------------------------------------


def test_a_spilled_run_matches_the_unspilled_answer_exactly(spill_probe):
    """Moving state to disk must not change a single row of the result."""
    spilled = _run(_TIGHT_CAP)
    assert spill_probe["passes"] > 0, "the cap did not force a spill, so this proves nothing"
    assert spill_probe["rows"] > 0

    resident = _run(1 << 30)
    assert spilled == resident


def test_the_same_query_without_the_spill_tier_raises(monkeypatch):
    """The 'before' this feature is measured against: at this cap the query used to die.

    Pinned so the equality test above cannot quietly become vacuous — if the cap ever rises
    above the query's peak state, this stops failing and says so.
    """
    monkeypatch.setattr(folds._WindowedAggFold, "_spill_cold_windows", lambda self: None)
    with (
        pytest.raises(ResourceError, match="windowed streaming aggregate state"),
        _capped(_TIGHT_CAP),
    ):
        list(_windowed().iter_batches())


def test_spilling_keeps_resident_state_under_the_cap(spill_probe):
    """The bound the tier exists to hold. Checked at every bound check, not just at the end."""
    peak = {"bytes": 0}
    original = folds._WindowedAggFold._check_state_bounded

    def observed(self):
        original(self)
        peak["bytes"] = max(peak["bytes"], self._fold.nbytes())

    folds._WindowedAggFold._check_state_bounded = observed
    try:
        _run(_TIGHT_CAP)
    finally:
        folds._WindowedAggFold._check_state_bounded = original
    assert 0 < peak["bytes"] <= _TIGHT_CAP


def test_state_that_cannot_be_spilled_still_raises():
    """A cap the *newest* windows alone exceed is a state too large, not one in the wrong
    place — no amount of disk fixes it, and saying so is more useful than thrashing."""
    with pytest.raises(ResourceError, match="spilling the older ones to disk"), _capped(1):
        list(_windowed().iter_batches())


# --- what the operator sees ------------------------------------------------


def test_rows_on_disk_are_still_reported_as_state(spill_probe):
    """Reporting only the resident half would make a spilling query look like it had *shed*
    state rather than moved it — backwards for the operator watching that number."""
    totals = []
    original = folds._WindowedAggFold.metrics

    def observed(self):
        result = original(self)
        totals.append(
            (result.num_rows_total, self._fold.state().num_rows if self._fold.state() else 0)
        )
        return result

    folds._WindowedAggFold.metrics = observed
    try:
        with _capped(_TIGHT_CAP):
            ds = _windowed()
            query = ds.write.for_each_batch(lambda _t, _b: None, trigger=bt.Trigger.available_now())
            query.await_termination()
    finally:
        folds._WindowedAggFold.metrics = original
    assert any(total > resident for total, resident in totals), (
        f"state totals never exceeded the resident rows, so the spilled half went unreported: "
        f"{totals[:5]}"
    )


# --- the checkpoint must carry the spilled half too ------------------------


def _fold_for(cap: int):
    """A `_WindowedAggFold` for the stream above, built under `cap`."""
    from batcher.core.streaming.folds import _window_key

    plan = _windowed()._plan
    return folds._WindowedAggFold(plan, _window_key(plan)), plan


def _flushed(fold) -> list[tuple]:
    """A fold's end-of-stream flush as sorted ``(window, key, total)`` triples."""
    result = fold.flush()
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


def test_a_snapshot_of_a_spilled_fold_carries_every_window(tmp_path):
    """The failure this exists to prevent: a snapshot built from the resident state alone
    persists only what happened to be in memory and silently drops the spilled windows —
    on exactly the queries large enough to have spilled."""
    from batcher.io.formats.streaming.checkpoint.store import CheckpointStore

    with _capped(_TIGHT_CAP):
        fold, plan = _fold_for(_TIGHT_CAP)
        for window in range(20):
            fold.push(_batch(window))
        assert fold._spill is not None and fold._spill.rows() > 0, "no spill, so this is vacuous"

        store = CheckpointStore(str(tmp_path / "ckpt"))
        store.state.snapshot(0, fold.state_parts())
        parts = store.state.restore_chain(0)
        assert len(parts) > 1, "the snapshot collapsed to one part, so the spill was dropped"

        from batcher.core.streaming.folds import _window_key

        revived = folds._WindowedAggFold(plan, _window_key(plan))
        revived.restore_parts(parts)
        assert revived._wm == fold._wm, "the watermark did not survive the multi-part snapshot"
        assert _flushed(revived) == _flushed(fold)


def test_reading_the_runs_for_a_snapshot_does_not_consume_them(tmp_path):
    """A checkpoint must leave the fold exactly as it found it."""
    with _capped(_TIGHT_CAP):
        fold, _plan = _fold_for(_TIGHT_CAP)
        for window in range(20):
            fold.push(_batch(window))
        before = fold._spill.rows()
        assert before > 0
        list(fold.state_parts())
        assert fold._spill.rows() == before


def test_a_fold_that_never_spilled_snapshots_exactly_one_part():
    """The common path must not pay for the spill tier existing."""
    with _capped(1 << 30):
        fold, _plan = _fold_for(1 << 30)
        for window in range(6):
            fold.push(_batch(window))
        assert fold._spill is None
        assert len(list(fold.state_parts())) == 1


def test_a_stopped_query_leaves_no_spill_scratch_behind(tmp_path):
    """Same lifetime problem the sink and checkpoint store already have in the engine's
    teardown: a driver that starts and stops queries otherwise leaves one directory of
    spilled state behind per query, and the only symptom is scratch filling a disk."""
    import os

    scratch = tmp_path / "scratch"
    scratch.mkdir()
    with _capped(_TIGHT_CAP):
        fold, _plan = _fold_for(_TIGHT_CAP)
        for window in range(20):
            fold.push(_batch(window))
        assert fold._spill is not None
        root = fold._spill._root
        assert os.path.isdir(root) and os.listdir(root)
        fold.close()
        assert not os.path.exists(root)
        fold.close()  # idempotent


def test_a_flush_drains_the_spilled_windows_rather_than_dropping_them():
    """A flush that ignored the disk tier would drop every window memory pressure had moved
    there — silently, and only on the streams large enough to have spilled."""
    with _capped(_TIGHT_CAP):
        fold, _plan = _fold_for(_TIGHT_CAP)
        for window in range(20):
            fold.push(_batch(window))
        assert fold._spill is not None and fold._spill.rows() > 0
        flushed = _flushed(fold)
    with _capped(1 << 30):
        resident, _plan = _fold_for(1 << 30)
        for window in range(20):
            resident.push(_batch(window))
        assert resident._spill is None
        assert flushed == _flushed(resident)


def test_a_row_for_an_already_spilled_window_still_reaches_it():
    """The correctness claim the whole tier rests on, and the bug it caught.

    Runs hold *partial* state, so a row arriving out of order for a window already on disk
    folds into memory and meets its spilled half when the window closes — `combine` is
    associative and commutative, so the split cannot change the total.

    It found a real defect. That row puts an early window back in memory, and the next spill
    writes it as the newest run with the *lowest* range; `drain_all` was taking its threshold
    from the last run's high on the assumption that runs are written in increasing window
    order, so the threshold collapsed and the flush silently skipped every run above it —
    two windows out of four, with nothing raised.
    """

    def drive(cap: int) -> list[tuple]:
        with _capped(cap):
            fold, _plan = _fold_for(cap)
            emitted: list[pa.RecordBatch] = []
            for window in range(20):
                emitted.extend(fold.push(_batch(window)))
            # Out of order but not late: the watermark allows 30s and this is well inside it.
            emitted.extend(fold.push(_at_second(4)))
            final = fold.flush()
            if final is not None:
                emitted.append(final)
        rows: list[tuple] = []
        for batch in emitted:
            rows.extend(
                zip(
                    batch.column("w").to_pylist(),
                    batch.column("k").to_pylist(),
                    batch.column("total").to_pylist(),
                    strict=True,
                )
            )
        return sorted(rows)

    spilled = drive(_TIGHT_CAP)
    resident = drive(1 << 30)
    assert spilled == resident
    windows = {window for window, _key, _total in spilled}
    assert len(windows) == 4, f"a window went missing from the spilled run: {sorted(windows)}"


def test_draining_every_run_does_not_depend_on_write_order():
    """`drain_all` must not infer a threshold from the last run — see the test above."""
    from batcher.core.streaming.spill import SpilledWindows

    spill = SpilledWindows("w")
    try:
        spill.spill(_window_batch(500))
        spill.spill(_window_batch(900))
        spill.spill(_window_batch(100))  # out of order, as a re-spill produces
        assert sorted(b.column("v")[0].as_py() for b in spill.drain_all()) == [100, 500, 900]
        assert len(spill) == 0
    finally:
        spill.close()
