"""Adaptive ingestion backpressure for a streaming query.

A micro-batch that overruns its trigger interval leaves the next one starting late against a
larger backlog, which overruns by more. The divergence compounds and ends not in a slow query
but in the epoch that no longer fits in memory. Static per-trigger caps bound that, but only
if they were hand-set for the worst trigger the query will ever see.

These tests pin the controller that derives the cap instead, the seam it acts through, and —
most importantly — the cases where it must decline to act, because a rate estimate built from
too little evidence is worse than none.
"""

from __future__ import annotations

import pytest

from batcher.carbonite.policies import PIDRateEstimator, StreamingRateController
from batcher.config import Config, StreamingConfig, config_context
from batcher.io.source import is_rate_limited
from batcher.plan.streaming import RateController, RateLimit, StreamingQueryProgress


def _progress(rows: int, duration_ms: float, behind_ms: float = 0.0) -> StreamingQueryProgress:
    return StreamingQueryProgress(
        batch_id=0,
        num_input_rows=rows,
        num_output_rows=rows,
        duration_ms=duration_ms,
        timestamp=0.0,
        behind_by_ms=behind_ms,
    )


# --- PIDRateEstimator ---------------------------------------------------------------


def test_the_first_measurement_is_the_estimate() -> None:
    """There is nothing to correct against yet, so the measured rate is the answer."""
    est = PIDRateEstimator()
    assert est.compute(rows=1000, processing_seconds=1.0, behind_seconds=0.0) == 1000.0


def test_a_batch_that_falls_behind_lowers_the_rate() -> None:
    """The whole point: a query that overran its cadence must be admitted less next time."""
    est = PIDRateEstimator()
    est.compute(rows=10_000, processing_seconds=1.0, behind_seconds=0.0)
    slowed = est.compute(rows=10_000, processing_seconds=4.0, behind_seconds=3.0)
    assert slowed is not None
    assert slowed < 10_000.0


def test_an_empty_trigger_does_not_throttle_a_healthy_query() -> None:
    """A quiet source is not a slow query.

    An empty batch's "rate" is zero, and folding that in would drive a perfectly healthy
    stream to the floor the first time its topic went quiet — and keep it there, because the
    throttle shrinks the next batch too.
    """
    est = PIDRateEstimator()
    est.compute(rows=10_000, processing_seconds=1.0, behind_seconds=0.0)
    assert est.compute(rows=0, processing_seconds=0.5, behind_seconds=0.0) is None
    assert est.rate == 10_000.0, "the estimate must survive an idle trigger untouched"


def test_a_batch_that_reported_no_duration_is_not_a_measurement() -> None:
    est = PIDRateEstimator()
    assert est.compute(rows=10, processing_seconds=0.0, behind_seconds=0.0) is None


def test_the_rate_never_falls_below_the_floor() -> None:
    """A controller that can reach zero can stall a query permanently.

    A trigger admitting nothing publishes no progress record, and a controller with no
    progress never revises the cap that stalled it.
    """
    est = PIDRateEstimator(min_rate=500.0)
    est.compute(rows=100_000, processing_seconds=0.1, behind_seconds=0.0)
    for _ in range(20):
        est.compute(rows=10, processing_seconds=5.0, behind_seconds=60.0)
    assert est.rate == 500.0


def test_the_integral_term_is_what_removes_steady_state_error() -> None:
    """The reason a purely proportional controller is not enough.

    Held at a persistent backlog, a proportional-only controller settles at a rate slightly
    above what the query sustains and stays permanently a little behind — the compounding
    case, arrived at more slowly. The integral term keeps pulling while the backlog persists.
    """
    sustained = 1000.0
    proportional_only = PIDRateEstimator(proportional=1.0, integral=0.0)
    with_integral = PIDRateEstimator(proportional=1.0, integral=0.2)
    for est in (proportional_only, with_integral):
        est.compute(rows=8000, processing_seconds=1.0, behind_seconds=0.0)
        for _ in range(10):
            est.compute(rows=int(sustained), processing_seconds=1.0, behind_seconds=2.0)
    assert with_integral.rate is not None and proportional_only.rate is not None
    assert with_integral.rate < proportional_only.rate, (
        "the backlog term must keep pulling after the instantaneous error is gone"
    )


def test_a_query_that_keeps_up_is_not_throttled_below_what_it_sustains() -> None:
    """Backpressure must not cost throughput a query was already delivering."""
    est = PIDRateEstimator()
    est.compute(rows=5000, processing_seconds=1.0, behind_seconds=0.0)
    for _ in range(10):
        est.compute(rows=5000, processing_seconds=1.0, behind_seconds=0.0)
    assert est.rate == pytest.approx(5000.0, rel=0.05)


# --- StreamingRateController --------------------------------------------------------


def test_it_satisfies_the_contract_the_loop_drives() -> None:
    assert isinstance(StreamingRateController(1.0), RateController)


def test_the_first_batch_earns_no_opinion() -> None:
    """A cold cache, a cold JIT and a connection handshake are not the steady state.

    Acting on batch one throttles a healthy query to a fraction of its capability, and
    because the throttle shrinks the next batch the estimate never recovers on its own.
    """
    ctrl = StreamingRateController(1.0)
    assert ctrl.next_limit(_progress(10_000, 1000.0)) is None


def test_the_cap_is_the_rate_spread_over_the_trigger_cadence() -> None:
    """A source is told a row count, not a rate, so the cadence has to be multiplied through."""
    ctrl = StreamingRateController(2.0)  # a 2-second trigger
    ctrl.next_limit(_progress(1000, 1000.0))
    limit = ctrl.next_limit(_progress(1000, 1000.0))
    assert limit is not None
    assert limit.max_rows == pytest.approx(limit.rows_per_second * 2.0, rel=0.01)


def test_a_trigger_with_no_cadence_falls_back_to_the_batch_duration() -> None:
    """`once` / `available_now` / `continuous` have no interval to be late for."""
    ctrl = StreamingRateController(None)
    ctrl.next_limit(_progress(1000, 500.0))
    limit = ctrl.next_limit(_progress(1000, 500.0))
    assert limit is not None and limit.max_rows >= 1


def test_the_cap_is_never_zero() -> None:
    """A cap of zero reads nothing, publishes no progress, and never revises itself."""
    cfg = Config(streaming=StreamingConfig(backpressure_min_rate=1.0))
    ctrl = StreamingRateController(0.001, cfg)
    ctrl.next_limit(_progress(1, 100_000.0, behind_ms=100_000.0))
    for _ in range(10):
        limit = ctrl.next_limit(_progress(1, 100_000.0, behind_ms=100_000.0))
        assert limit is None or limit.max_rows >= 1


def test_an_operator_ceiling_holds_the_derived_cap_down() -> None:
    cfg = Config(streaming=StreamingConfig(backpressure_max_rows_per_trigger=50))
    ctrl = StreamingRateController(1.0, cfg)
    ctrl.next_limit(_progress(1_000_000, 1000.0))
    limit = ctrl.next_limit(_progress(1_000_000, 1000.0))
    assert limit is not None and limit.max_rows == 50


# --- the source seam ----------------------------------------------------------------


def test_a_broker_source_accepts_an_admission_cap() -> None:
    from batcher.io.formats.streaming.kafka import KafkaSource

    source = KafkaSource("t", poll_size=1000)
    assert is_rate_limited(source)
    source.set_admission_limit(250)
    assert source.poll_size == 250


def test_the_cap_can_only_narrow_what_the_operator_configured() -> None:
    """A controller must never hand a source more than its operator allowed."""
    from batcher.io.formats.streaming.kafka import KafkaSource

    source = KafkaSource("t", poll_size=1000)
    source.set_admission_limit(10_000_000)
    assert source.poll_size == 1000
    source.set_admission_limit(None)
    assert source.poll_size == 1000, "lifting the cap returns to the configured size"


def test_one_property_throttles_every_broker() -> None:
    """Kafka, Kinesis, Pulsar, Pub/Sub and Event Hubs each bound their poll by `poll_size`.

    Narrowing it on the shared base is what reaches all five without a line in any of them —
    and a regression here would silently un-throttle four of the connectors.
    """
    from batcher.io.formats.streaming.eventhubs import EventHubsSource
    from batcher.io.formats.streaming.kafka import KafkaSource
    from batcher.io.formats.streaming.kinesis import KinesisSource
    from batcher.io.formats.streaming.pubsub import PubSubSource
    from batcher.io.formats.streaming.pulsar import PulsarSource

    for cls in (KafkaSource, KinesisSource, PulsarSource, PubSubSource, EventHubsSource):
        source = cls("t", poll_size=800)
        assert is_rate_limited(source), f"{cls.__name__} is not rate limited"
        source.set_admission_limit(40)
        assert source.poll_size == 40, f"{cls.__name__} ignored its admission cap"


def test_a_source_with_no_per_trigger_admission_is_simply_never_throttled() -> None:
    import pyarrow as pa

    from batcher.io import InMemorySource

    assert not is_rate_limited(InMemorySource([pa.record_batch({"x": [1]})]))


# --- config -------------------------------------------------------------------------


def test_backpressure_is_off_by_default() -> None:
    """Matching Spark. A controller acting on an estimate it has not earned is worse than none."""
    assert StreamingConfig().backpressure_enabled is False


def test_the_pid_defaults_are_sparks() -> None:
    """So an operator's existing `spark.streaming.backpressure.pid.*` tuning carries over."""
    cfg = StreamingConfig()
    assert (cfg.backpressure_pid_proportional, cfg.backpressure_pid_integral) == (1.0, 0.2)
    assert cfg.backpressure_pid_derivative == 0.0
    assert cfg.backpressure_min_rate == 100.0


def test_a_floor_of_zero_is_refused() -> None:
    """Unlike a transient tunable, this one cannot recover on its own once it stalls a stream."""
    with pytest.raises(ValueError, match="backpressure_min_rate"):
        StreamingConfig(backpressure_min_rate=0.0)


def test_a_negative_weight_is_refused() -> None:
    """It would make the controller answer an overrun by admitting *more*."""
    with pytest.raises(ValueError, match="backpressure_pid_integral"):
        StreamingConfig(backpressure_pid_integral=-1.0)


def test_the_conductor_builds_a_controller_only_when_it_is_enabled() -> None:
    """`api` is the one place Carbonite's policy and Core's loop are joined."""
    from batcher.api.streaming._launch import _rate_controller
    from batcher.plan.streaming import Trigger

    trigger = Trigger.processing_time(1.0)
    with config_context(Config(streaming=StreamingConfig(backpressure_enabled=False))):
        assert _rate_controller(trigger) is None
    with config_context(Config(streaming=StreamingConfig(backpressure_enabled=True))):
        assert isinstance(_rate_controller(trigger), RateController)


def test_a_rate_limit_reports_the_rate_behind_the_cap() -> None:
    """A progress reader needs to see *why* a source was narrowed, not only that it was."""
    limit = RateLimit(max_rows=200, rows_per_second=100.0)
    assert (limit.max_rows, limit.rows_per_second) == (200, 100.0)


# --- the loop actually applies it ---------------------------------------------------


class _CountingSource:
    """A rate-limitable stand-in that records every cap it was handed."""

    bounded = False

    def __init__(self) -> None:
        self.caps: list[int | None] = []

    def set_admission_limit(self, max_rows: int | None) -> None:
        self.caps.append(max_rows)


class _FixedController:
    """Names the same cap every time, so the test asserts on wiring rather than on a curve."""

    def next_limit(self, progress: StreamingQueryProgress) -> RateLimit | None:
        return RateLimit(max_rows=77, rows_per_second=77.0)


def _engine(source: object, controller: object | None):
    from batcher.core.streaming_query import StreamingQueryEngine
    from batcher.plan.streaming import Trigger

    return StreamingQueryEngine(
        name="q",
        source=source,
        sink=None,
        processor=None,
        trigger=Trigger.processing_time(1.0),
        output_mode="append",
        runner_factory=lambda _stop: object(),
        rate_controller=controller,
    )


def test_the_loop_pushes_the_cap_into_the_source() -> None:
    """The wiring, end to end: a derived limit has to reach the thing that reads."""
    source = _CountingSource()
    engine = _engine(source, _FixedController())
    engine._apply_rate_limit(_progress(1000, 1000.0))
    assert source.caps == [77]


def test_no_controller_leaves_the_source_alone() -> None:
    """The default. An unconfigured query must behave exactly as it did before this existed."""
    source = _CountingSource()
    _engine(source, None)._apply_rate_limit(_progress(1000, 1000.0))
    assert source.caps == []


def test_a_source_that_cannot_be_paced_is_not_asked_to_be() -> None:
    """A file or in-memory source has no per-trigger admission; calling it would raise."""

    class _Unpaceable:
        bounded = False

    engine = _engine(_Unpaceable(), _FixedController())
    engine._apply_rate_limit(_progress(1000, 1000.0))  # must not raise


def test_an_abstaining_controller_does_not_touch_the_source() -> None:
    """`None` means "leave the configured cap governing", not "cap at nothing"."""

    class _Abstains:
        def next_limit(self, progress: StreamingQueryProgress) -> RateLimit | None:
            return None

    source = _CountingSource()
    _engine(source, _Abstains())._apply_rate_limit(_progress(1000, 1000.0))
    assert source.caps == []
