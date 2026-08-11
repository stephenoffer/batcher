"""The watermark is a minimum over partitions, not a maximum over rows.

Every test here is a statement about data that is *lost* when it is a maximum: a fast
Kafka partition drags the frontier forward, and the slow partition's rows are then ruled
late and silently dropped. The unit under test is `plan.streaming.WatermarkTracker`, which
is where that rule is stated once for the windowed aggregate, the dedup, the interval join
and the session window.
"""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

from batcher.plan.streaming import WatermarkTracker

pytestmark = pytest.mark.unit

MINUTE = 60_000_000


def _batch(rows: list[tuple[int, dt.datetime]], *, partitioned: bool = True) -> pa.RecordBatch:
    """One broker-shaped batch: `(partition, ts)` pairs on topic ``t``."""
    columns = {"ts": pa.array([ts for _, ts in rows], type=pa.timestamp("us"))}
    if partitioned:
        columns["partition"] = pa.array([p for p, _ in rows], type=pa.int64())
        columns["topic"] = pa.array(["t"] * len(rows), type=pa.string())
    return pa.record_batch(columns)


def _at(minute: int) -> dt.datetime:
    return dt.datetime(2024, 1, 1, 10, 0) + dt.timedelta(minutes=minute)


def _micros(minute: int) -> int:
    """`_at(minute)` as epoch microseconds — the unit every watermark is expressed in.

    Computed against the epoch directly rather than through `datetime.timestamp()`, which
    reads a naive datetime as *local* time while Arrow reads it as UTC. That eight-hour
    disagreement is not a rounding detail; it is the difference between a test that checks
    the watermark and one that checks the machine's timezone.
    """
    return (_at(minute) - dt.datetime(1970, 1, 1)) // dt.timedelta(microseconds=1)


class _Clock:
    """A processing clock the test moves by hand, so idleness is not a `sleep`."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _tracker(**kwargs) -> WatermarkTracker:
    kwargs.setdefault("idle_timeout_seconds", 0)  # idleness off unless a test asks for it
    return WatermarkTracker(kwargs.pop("lateness", 0), **kwargs)


def test_the_slowest_partition_sets_the_watermark() -> None:
    """The whole bug in one assertion: partition 0 at 10:00 does not speak for partition 1."""
    tracker = _tracker()
    tracker.observe(_batch([(0, _at(60)), (1, _at(5))]), "ts", ("topic", "partition"))
    assert tracker.watermark == _micros(5)


def test_a_fast_partition_alone_cannot_advance_past_a_slow_one() -> None:
    """Batch after batch from the fast partition leaves the frontier where the slow one is."""
    tracker = _tracker()
    tracker.observe(_batch([(0, _at(10)), (1, _at(1))]), "ts", ("topic", "partition"))
    frozen = tracker.watermark
    for minute in (20, 30, 40):
        tracker.observe(_batch([(0, _at(minute))]), "ts", ("topic", "partition"))
    assert tracker.watermark == frozen
    # ... and moves the moment the slow partition does.
    tracker.observe(_batch([(1, _at(15))]), "ts", ("topic", "partition"))
    assert tracker.watermark == _micros(15)


def test_lateness_is_subtracted_from_the_minimum() -> None:
    tracker = _tracker(lateness=10 * MINUTE)
    tracker.observe(_batch([(0, _at(60)), (1, _at(30))]), "ts", ("topic", "partition"))
    assert tracker.watermark == _micros(20)


def test_the_watermark_never_rewinds() -> None:
    """A partition that goes backwards must not re-admit rows an earlier pass ruled late."""
    tracker = _tracker()
    tracker.observe(_batch([(0, _at(30)), (1, _at(30))]), "ts", ("topic", "partition"))
    high = tracker.watermark
    tracker.observe(_batch([(0, _at(5)), (1, _at(5))]), "ts", ("topic", "partition"))
    assert tracker.watermark == high


def test_an_unpartitioned_stream_is_the_maximum_it_always_was() -> None:
    """No partition columns means one partition, so min == max and nothing changed."""
    tracker = _tracker()
    tracker.observe(_batch([(0, _at(5)), (0, _at(60))], partitioned=False), "ts", ())
    assert tracker.watermark == _micros(60)


def test_partition_columns_absent_from_the_batch_degrade_to_one_partition() -> None:
    """A pipeline that projected `partition` away must still produce a watermark."""
    tracker = _tracker()
    tracker.observe(_batch([(0, _at(5)), (0, _at(60))], partitioned=False), "ts", ("partition",))
    assert tracker.watermark == _micros(60)


def test_null_event_times_do_not_advance_anything() -> None:
    batch = pa.record_batch(
        {
            "ts": pa.array([None, None], type=pa.timestamp("us")),
            "partition": pa.array([0, 1], type=pa.int64()),
        }
    )
    tracker = _tracker()
    tracker.observe(batch, "ts", ("partition",))
    assert tracker.watermark is None


def test_an_empty_batch_is_a_no_op() -> None:
    tracker = _tracker()
    tracker.observe(_batch([]), "ts", ("topic", "partition"))
    assert tracker.watermark is None


def test_a_nanosecond_column_is_not_scaled_by_a_thousand() -> None:
    """Reading raw ticks of a `timestamp[ns]` column would put the frontier in year 1970."""
    batch = pa.record_batch(
        {
            "ts": pa.array([_at(30), _at(60)], type=pa.timestamp("ns")),
            "partition": pa.array([0, 0], type=pa.int64()),
        }
    )
    tracker = _tracker()
    tracker.observe(batch, "ts", ("partition",))
    assert tracker.watermark == _micros(60)


class TestIdleness:
    """A minimum stalls on a silent partition; idleness is the documented release valve."""

    def test_a_silent_partition_pins_the_watermark_when_idleness_is_off(self) -> None:
        clock = _Clock()
        tracker = WatermarkTracker(0, idle_timeout_seconds=0, clock=clock)
        tracker.observe(_batch([(0, _at(10)), (1, _at(1))]), "ts", ("partition",))
        pinned = tracker.watermark
        clock.now = 10_000.0
        tracker.observe(_batch([(0, _at(90))]), "ts", ("partition",))
        assert tracker.watermark == pinned

    def test_an_idle_partition_stops_holding_the_minimum_back(self) -> None:
        clock = _Clock()
        tracker = WatermarkTracker(0, idle_timeout_seconds=30.0, clock=clock)
        tracker.observe(_batch([(0, _at(10)), (1, _at(1))]), "ts", ("partition",))
        assert tracker.watermark == _micros(1)
        clock.now = 31.0
        tracker.observe(_batch([(0, _at(90))]), "ts", ("partition",))
        assert tracker.watermark == _micros(90)

    def test_idleness_is_re_evaluated_on_read_not_only_on_a_batch(self) -> None:
        """A partition expiring between batches releases the minimum without waiting.

        Partition 0 keeps delivering, so it is the one still active at the read; partition 1
        has been silent since the start and drops out of the minimum at the read itself,
        with no further batch to prompt it.
        """
        clock = _Clock()
        tracker = WatermarkTracker(0, idle_timeout_seconds=30.0, clock=clock)
        tracker.observe(_batch([(0, _at(10)), (1, _at(1))]), "ts", ("partition",))
        clock.now = 25.0
        tracker.observe(_batch([(0, _at(50))]), "ts", ("partition",))
        assert tracker.watermark == _micros(1)  # partition 1 still has a say
        clock.now = 31.0
        assert tracker.watermark == _micros(50)

    def test_every_partition_going_idle_freezes_rather_than_advances(self) -> None:
        """Silence is not progress: with nothing delivering, the frontier does not move."""
        clock = _Clock()
        tracker = WatermarkTracker(0, idle_timeout_seconds=30.0, clock=clock)
        tracker.observe(_batch([(0, _at(10)), (1, _at(1))]), "ts", ("partition",))
        clock.now = 10_000.0
        assert tracker.watermark == _micros(1)


class TestExpectedPartitions:
    """Startup is the other half of the bug: a minimum over a subset is still an over-claim."""

    def test_a_declared_partition_that_has_not_spoken_holds_the_watermark(self) -> None:
        tracker = WatermarkTracker(
            0, idle_timeout_seconds=0, expected_partitions=[("t", 0), ("t", 1)]
        )
        tracker.observe(_batch([(0, _at(60))]), "ts", ("topic", "partition"))
        assert tracker.watermark is None
        tracker.observe(_batch([(1, _at(5))]), "ts", ("topic", "partition"))
        assert tracker.watermark == _micros(5)

    def test_a_declared_partition_is_released_by_idleness(self) -> None:
        """An empty partition must not pin a working query at no watermark forever."""
        clock = _Clock()
        tracker = WatermarkTracker(
            0, idle_timeout_seconds=30.0, expected_partitions=[("t", 0), ("t", 1)], clock=clock
        )
        tracker.observe(_batch([(0, _at(60))]), "ts", ("topic", "partition"))
        assert tracker.watermark is None
        clock.now = 25.0
        tracker.observe(_batch([(0, _at(70))]), "ts", ("topic", "partition"))
        clock.now = 31.0  # partition 1 has now been silent since the query started
        assert tracker.watermark == _micros(70)

    def test_an_unattributable_batch_drops_the_expectation(self) -> None:
        """Declared partitions nothing can ever deliver under must not pin the stream.

        A source declares `(topic, partition)` and a pipeline projects both away. The
        expectation is then unsatisfiable, and holding the watermark at None until every
        declared partition timed out would stall a query that was working before.
        """
        tracker = WatermarkTracker(
            0, idle_timeout_seconds=0, expected_partitions=[("t", 0), ("t", 1)]
        )
        tracker.observe(_batch([(0, _at(60))], partitioned=False), "ts", ("topic", "partition"))
        assert tracker.watermark == _micros(60)


class TestCheckpointRoundTrip:
    """Per-partition state has to survive a restart, or every restart re-runs the bug."""

    def test_the_per_partition_maxima_round_trip(self) -> None:
        tracker = _tracker()
        tracker.observe(_batch([(0, _at(60)), (1, _at(5))]), "ts", ("topic", "partition"))
        resumed = _tracker()
        resumed.restore(tracker.to_json())
        assert resumed.watermark == tracker.watermark
        # The fast partition's maximum survived: it does not get to reset the minimum.
        resumed.observe(_batch([(1, _at(10))]), "ts", ("topic", "partition"))
        assert resumed.watermark == _micros(10)

    def test_restoring_only_the_frontier_would_lose_the_slow_partition(self) -> None:
        """The contrast case: a frontier-only restore lets the next batch set the minimum."""
        resumed = _tracker()
        resumed.restore('{"maxima": {}, "watermark": null}')
        resumed.observe(_batch([(0, _at(90))]), "ts", ("topic", "partition"))
        assert resumed.watermark == _micros(90)

    def test_a_bare_integer_is_the_pre_per_partition_checkpoint_format(self) -> None:
        """A query checkpointed by an older build resumes instead of rewinding event time."""
        resumed = _tracker(lateness=10 * MINUTE)
        resumed.restore(str(_micros(20)))
        assert resumed.watermark == _micros(20)

    def test_a_corrupt_payload_leaves_the_tracker_empty_rather_than_raising(self) -> None:
        tracker = _tracker()
        tracker.restore("not json at all")
        assert tracker.watermark is None
