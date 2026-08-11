"""Adaptive ingestion rate for a streaming query — the micro-batch loop's backpressure.

The credit window paces a *shuffle channel*; this paces the *source*. They are the two halves
of the same discipline and neither substitutes for the other: a credit window cannot stop a
trigger from reading a backlog that will not fit, and a rate limit cannot stop a fast mapper
from flooding a slow reducer.

**The failure this exists to prevent.** A micro-batch that overruns its trigger interval leaves
the next one starting late against a larger backlog, which overruns by more. The divergence
compounds, and it does not end in a slow query — it ends in the epoch that no longer fits in
memory. The static caps (``max_offsets_per_trigger``, ``max_files_per_trigger``) bound it, but
they have to be hand-set for the worst trigger the query will ever see, so they throttle every
other one and go stale as soon as the cluster, the data, or the plan changes.

**The controller.** `PIDRateEstimator` is Spark's ``PIDRateEstimator``, including its knob
names and defaults, because the tuning advice an operator already has should carry over
verbatim. Each completed micro-batch supplies a measured processing rate (rows over the time
spent processing them); the error against the rate currently being admitted drives a
proportional term, the accumulated backlog drives an integral term, and the error's rate of
change drives a derivative term. The sum is the next admission rate.

Its three qualities matter more than its formula. The proportional term reacts to the current
error, so a query that suddenly slows is throttled on the next trigger rather than after a
drift. The integral term is what removes *steady-state* error: a purely proportional controller
settles at a rate slightly above what the query can sustain and stays permanently a little
behind, which is precisely the compounding case. The derivative term damps the overshoot the
integral term would otherwise cause on a bursty source.

**It only ever lowers.** The cap is held under whatever the source was already configured with,
so enabling this can tighten a static limit and never loosen one. An admission cap changes how
much of a stream a trigger reads, never what the query computes from it, so this cannot change
a result.

**Off by default**, as it is in Spark. A controller acting on an estimate it has not earned is
worse than none, so it also abstains until it has measured enough to have an opinion.
"""

from __future__ import annotations

from dataclasses import dataclass

from batcher.config import Config, active_config
from batcher.plan.streaming.progress import StreamingQueryProgress
from batcher.plan.streaming.rate import RateLimit

__all__ = ["PIDRateEstimator", "StreamingRateController"]

#: Micro-batches that must complete before the controller will name a rate.
#:
#: The first batch of a query measures a cold cache, a cold JIT and a connection handshake, so
#: its processing rate is far below the steady-state one. Acting on it throttles a healthy
#: query to a fraction of its capability, and because the throttle then *shrinks* the next
#: batch, the estimate never recovers on its own. Two batches is the fewest that can supply a
#: derivative term at all.
_WARMUP_BATCHES = 2


@dataclass(slots=True)
class PIDRateEstimator:
    """Spark's `PIDRateEstimator`: a sustainable ingestion rate from measured progress.

    Stateful — one per streaming query. Feed it every completed micro-batch in order.

    Args:
        proportional: Weight on the current error (Spark ``pid.proportional``).
        integral: Weight on the accumulated backlog (Spark ``pid.integral``).
        derivative: Weight on the error's rate of change (Spark ``pid.derived``).
        min_rate: Floor on the derived rate, in rows per second. A controller that can reach
            zero is one that can stall a query permanently, because a stream reading nothing
            emits no progress and so never revises its own estimate.

    Examples:
        .. doctest::

            >>> from batcher.carbonite.policies import PIDRateEstimator
            >>> est = PIDRateEstimator()
            >>> est.min_rate
            100.0
    """

    proportional: float = 1.0
    integral: float = 0.2
    derivative: float = 0.0
    min_rate: float = 100.0
    #: The rate the last estimate settled on. `None` until the first one is computed, which is
    #: also what makes the first call a pure measurement rather than a correction.
    _rate: float | None = None
    #: The previous error, for the derivative term.
    _last_error: float = 0.0

    def compute(
        self, *, rows: int, processing_seconds: float, behind_seconds: float
    ) -> float | None:
        """The next sustainable rate in rows per second, or `None` when unmeasurable.

        Args:
            rows: Rows the micro-batch consumed.
            processing_seconds: How long it spent consuming them.
            behind_seconds: How far past its trigger interval the batch ran. This is the
                *scheduling delay* in Spark's formulation, and it is the term that makes the
                controller act on a backlog rather than only on an instantaneous rate.

        Returns:
            The rate, or `None` for a batch that measured nothing — an empty trigger or one
            that reported no duration. Abstaining is deliberate: an empty batch's "rate" is
            zero, and folding that in would throttle a healthy query to the floor the first
            time its source went quiet.
        """
        if rows <= 0 or processing_seconds <= 0:
            return None
        measured = rows / processing_seconds
        if self._rate is None:
            # Nothing to correct against yet: the first measurement *is* the estimate.
            self._rate = max(self.min_rate, measured)
            return self._rate

        # How far the rate being admitted sits above what the query actually sustained. A
        # positive error means the source is being let in faster than it can be processed.
        error = self._rate - measured
        # The backlog term. `behind_seconds` of unprocessed arrivals at the measured rate is
        # the number of rows already owed, spread over the interval it must be repaid in.
        historical_error = (
            behind_seconds * measured / processing_seconds if processing_seconds > 0 else 0.0
        )
        d_error = (error - self._last_error) / processing_seconds
        self._last_error = error
        self._rate = max(
            self.min_rate,
            self._rate
            - self.proportional * error
            - self.integral * historical_error
            - self.derivative * d_error,
        )
        return self._rate

    @property
    def rate(self) -> float | None:
        """The current estimate, or `None` before the first measurable batch."""
        return self._rate


class StreamingRateController:
    """Turns a query's progress records into the next micro-batch's admission cap.

    The `plan.streaming.RateController` the micro-batch loop drives. It converts the
    estimator's rows-per-second into a row count by multiplying through the trigger's cadence,
    because a cap is what a source can actually be told.

    Args:
        interval_seconds: The trigger's cadence. A trigger with no interval (``once`` /
            ``available_now`` / ``continuous``) has no cadence to multiply through, so the
            controller measures the last batch's own duration instead — the two agree for a
            query that is keeping up, and the second is the only figure available when there
            is no interval to be late for.
        config: The active config, for the PID weights and the floor.
    """

    __slots__ = ("_batches", "_estimator", "_interval", "_max_rows")

    def __init__(self, interval_seconds: float | None = None, config: Config | None = None) -> None:
        cfg = (config or active_config()).streaming
        self._estimator = PIDRateEstimator(
            proportional=cfg.backpressure_pid_proportional,
            integral=cfg.backpressure_pid_integral,
            derivative=cfg.backpressure_pid_derivative,
            min_rate=cfg.backpressure_min_rate,
        )
        self._interval = interval_seconds if interval_seconds and interval_seconds > 0 else None
        # An operator's own ceiling on the derived cap, so backpressure can only ever tighten
        # what they configured. `0` leaves the estimator unbounded above.
        self._max_rows = max(0, cfg.backpressure_max_rows_per_trigger)
        self._batches = 0

    def next_limit(self, progress: StreamingQueryProgress) -> RateLimit | None:
        """Fold one micro-batch in and return the next cap, or `None` to abstain.

        Args:
            progress: The record the completed micro-batch published.

        Returns:
            The cap, or `None` during warm-up and for a batch that measured nothing. The loop
            reads `None` as "leave the source's configured limit alone".
        """
        self._batches += 1
        seconds = progress.duration_ms / 1000.0
        rate = self._estimator.compute(
            rows=progress.num_input_rows,
            processing_seconds=seconds,
            behind_seconds=progress.behind_by_ms / 1000.0,
        )
        if rate is None or self._batches < _WARMUP_BATCHES:
            return None
        # Rows the next trigger may read: the sustainable rate over the window it has to read
        # them in. This is Little's Law, `L = lambda W`, and it is the same identity the credit
        # window is sized by — in-flight work equals arrival rate times time in system. The
        # shuffle's version measures `lambda` as bytes per second and `W` as a round trip; this
        # one measures `lambda` as rows per second and `W` as the trigger interval. Two
        # controllers, one law, which is why they compose rather than fight.
        #
        # Falls back to the batch's own duration when the trigger has no cadence, which is the
        # only measurement available for a drain or continuous trigger.
        window = self._interval if self._interval is not None else seconds
        rows = int(rate * window)
        if self._max_rows:
            rows = min(rows, self._max_rows)
        # Never zero. A cap of zero reads nothing, a trigger that reads nothing publishes no
        # progress, and a controller with no progress never revises the cap that stalled it.
        return RateLimit(max_rows=max(1, rows), rows_per_second=rate)
