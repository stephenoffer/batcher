"""The micro-batch loop — trigger cadence, checkpointing, recovery, and progress.

`core`'s lane is *making it happen and measuring*: this drives a long-running query by
pulling micro-batches from an unbounded `Source`, handing each to a `MicroBatchProcessor`
(built by the conductor in `api`), emitting the result to a `StreamSink` per the
`OutputMode`, and firing on the `Trigger`'s cadence. It never optimizes (Kyber ran once at
start) and never owns resources (Carbonite) — it only runs and records
`StreamingQueryProgress`.

The loop runs on a background thread so `start()` returns a handle the caller can `stop()` /
`await`; per-micro-batch metrics already flow to the `MetadataHub` through
`core.execute_local`, so a streaming query improves future plans the same way a batch query
does.
"""

from __future__ import annotations

import contextlib
import threading
from collections import deque
from collections.abc import Callable
from time import perf_counter, time
from typing import TYPE_CHECKING

from batcher._internal.concurrency import start_context_thread
from batcher.core.streaming_query.state_policy import write_state
from batcher.plan.streaming import (
    SinkProgress,
    SourceProgress,
    StateOperatorProgress,
    StreamingQueryProgress,
    StreamingQueryStatus,
    Trigger,
    notify_query_progress,
    notify_query_started,
    notify_query_terminated,
)

if TYPE_CHECKING:
    from batcher.core.streaming_query.processors import MicroBatchProcessor
    from batcher.core.streaming_runner import MicroBatchRunner
    from batcher.io.source import Source
    from batcher.plan.streaming.rate import RateController

__all__ = ["StreamingQueryEngine"]

#: Seconds between "still stopping" warnings while `stop()` waits for the loop thread.
_STOP_WARN_SECONDS = 30


def _progress_history() -> int:
    """How many micro-batch progress records a query retains (`streaming.progress_history`)."""
    from batcher.config import active_config

    return active_config().streaming.progress_history


def _describe(obj: object) -> str:
    """A short human name for a source or sink, for the progress record.

    A source's `identity()` already exists and is exactly this string (``kafka:events:…``);
    anything else falls back to its class name, which is still more use to an operator
    reading a progress record than an empty field.
    """
    identity = getattr(obj, "identity", None)
    if identity is not None:
        try:
            return str(identity())
        except Exception:
            # Not silence — the fall-through to the class name below *is* the documented
            # answer. This only labels a progress record, so a source whose `identity()`
            # raises (a broker option it cannot render, say) must still produce a query
            # that runs; degrading the label is the whole point of having a fallback.
            pass
    return type(obj).__name__


class StreamingQueryEngine:
    """Drives a streaming query on a background thread; the `api` handle wraps it."""

    def __init__(
        self,
        *,
        name: str,
        source: Source,
        sink,
        processor: MicroBatchProcessor,
        trigger: Trigger,
        output_mode: str,
        checkpoint=None,
        runner_factory: Callable[[Callable[[], bool]], MicroBatchRunner] | None = None,
        projection: list[str] | None = None,
        predicate: dict | None = None,
        rate_controller: RateController | None = None,
    ) -> None:
        from batcher.core.streaming_runner import LocalRunner

        self._name = name
        self._source = source
        self._sink = sink
        self._processor = processor
        self._trigger = trigger
        self._output_mode = output_mode
        self._checkpoint = checkpoint
        # Adaptive ingestion backpressure, injected by the conductor because the policy is
        # Carbonite's and `core` must not import it. `None` (the default) leaves the source's
        # configured per-trigger cap governing, exactly as before. Core measures and applies;
        # it does not decide the rate.
        self._rate_controller = rate_controller
        self._stop = threading.Event()
        # How a micro-batch actually runs. Default: on this thread. The conductor injects
        # the Ray fan-out for `distributed=True` — `core` never imports `dist`. The factory
        # takes the stop predicate, so a runner that waits for data on an idle stream still
        # observes `stop()` promptly.
        # `projection`/`predicate` are Kyber's source pushdown for this plan. They reach the
        # runner so the *source* decodes only the columns the plan needs; without them a
        # `select` over a wide stream decoded every column of every message forever.
        self._runner: MicroBatchRunner = (
            runner_factory(self._stop.is_set)
            if runner_factory is not None
            else LocalRunner(
                source,
                processor,
                sink,
                projection=projection,
                predicate=predicate,
                should_stop=self._stop.is_set,
                # Only a draining trigger reads an idle unbounded source as a finished
                # one. Without this the runner stopped an Auto Loader stream after its
                # first discovery pass — see `LocalRunner`.
                drain=trigger.is_drain,
            )
        )
        self._thread: threading.Thread | None = None
        self._progress: deque[StreamingQueryProgress] = deque(maxlen=_progress_history())
        self._batches = 0
        # Changelog deltas written since the last whole snapshot. Bounds how long a chain
        # recovery has to replay — see `_write_state`.
        self._deltas_written = 0
        self._error: BaseException | None = None
        self._active = False

    # --- lifecycle --------------------------------------------------------
    def start(self) -> None:
        """Recover from the checkpoint (if any), open the sink, launch the loop."""
        self._active = True
        self._recover()
        if self._sink is not None:
            self._sink.open()  # a distributed runner owns its sinks (one per worker)
        # The loop runs under a *snapshot of the caller's context*, not a fresh one.
        #
        # `threading.Thread` does not copy context variables, and the control plane keeps
        # everything that answers "what does this query think the machine looks like" in
        # one: the active `Config`, the cancellation scope, the machine-scoping key the
        # learned statistics are filed under, the shuffle fleet. A bare thread target reads
        # every one of them at its *default*, so a `config_context` wrapped around
        # `write_stream(...)` governed the setup this method does and then silently stopped
        # applying the moment the loop started — a pinned `max_memory_bytes` reverted to the
        # static 8 GiB fallback, an adjusted morsel size reverted to 16,384 rows, and a
        # spill directory reverted to the system tempdir, all without an error anywhere.
        #
        # Copying at `start()` also gives the right *lifetime*. A streaming query outlives
        # the `with config_context(...)` block that launched it, so the config it runs under
        # has to be frozen at launch rather than read live — which is what a snapshot is,
        # and what Spark does with a query's configuration for the same reason.
        self._thread = start_context_thread(
            self._run, name=f"batcher-stream-{self._name}", daemon=True
        )
        # After the thread is running, so a listener that inspects the query sees a live
        # one. Before this existed the start of a query was the one event nothing could
        # observe: polling `recent_progress` can only ever see batches that already ran.
        notify_query_started(self._name, time())

    def _recover(self) -> None:
        """Restore source position, batch counter, and running state from a checkpoint."""
        if self._checkpoint is None:
            return
        from batcher.io.formats.streaming.checkpoint import recover

        plan = recover(self._checkpoint)
        self._batches = plan.start_batch
        if plan.seek and 0 in plan.seek:
            self._runner.seek(plan.seek[0])
        if plan.state is None:
            return
        # Continue the chain rather than restarting the count. A restart that reset this to
        # zero could write another full interval of deltas on top of the ones it just
        # replayed, so the chain recovery has to walk grows by an interval per restart — a
        # query restarted often enough would replay an unbounded one.
        self._deltas_written = len(plan.state_deltas)
        if plan.state_deltas and getattr(self._runner, "restore_state_chain", None) is None:
            # A multi-part *snapshot* (a spilled fold), not a changelog: the parts are halves
            # of one state and are combined, which `restore_state_parts` does.
            self._runner.restore_state_parts([plan.state, *plan.state_deltas])
            return
        chain = getattr(self._runner, "restore_state_chain", None)
        if plan.state_deltas and chain is not None:
            # A base plus the changelog after it. Combining is the aggregate algebra, so the
            # runner does it — this layer only knows the checkpoint held a sequence.
            chain([plan.state, *plan.state_deltas])
            return
        restore = getattr(self._runner, "restore_state", None)
        if restore is not None:
            restore(plan.state)

    def stop(self) -> None:
        """Signal the loop to halt at the next micro-batch boundary and join.

        The join is unbounded on purpose — returning while the loop still writes to the sink
        would be worse than waiting — but it is no longer *silent*. A stop is only observed
        between micro-batches, so a source parked on a blocking read holds the join for as
        long as that read takes, and with no output the caller cannot tell a slow drain from
        a permanent hang. A periodic warning names the condition and what to check, the way
        Carbonite's distributed barrier does.
        """
        self._stop.set()
        if self._thread is None:
            return
        while self._thread.is_alive():
            self._thread.join(_STOP_WARN_SECONDS)
            if self._thread.is_alive():
                from batcher._internal.logging import get_logger

                get_logger("core").warning(
                    "streaming query %r has not stopped after %ds: the source is inside a "
                    "read that has not returned. A stop is observed only between "
                    "micro-batches, so an unbounded poll timeout (a broker's poll_timeout, "
                    "a socket with no traffic) holds it here.",
                    self._name,
                    _STOP_WARN_SECONDS,
                )

    def await_termination(self, timeout: float | None = None) -> bool:
        """Block until the query stops (or `timeout` seconds); return whether it stopped."""
        if self._thread is None:
            return True
        self._thread.join(timeout)
        if self._thread.is_alive():
            return False
        if self._error is not None:
            raise self._error
        return True

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def exception(self) -> BaseException | None:
        return self._error

    def recent_progress(self) -> list[StreamingQueryProgress]:
        return list(self._progress)

    def status(self) -> StreamingQueryStatus:
        if self._active:
            message = "Waiting for data"
        elif self._error is not None:
            # A failed query is stopped too, so `is_active=False` alone read identically to a
            # clean stop. Name the failure in the message so a caller polling `status` — not
            # just one that calls `await_termination` — can see the query died and why.
            message = f"Failed: {type(self._error).__name__}: {self._error}"
        else:
            message = "Stopped"
        return StreamingQueryStatus(
            is_active=self._active,
            is_data_available=bool(self._progress) and self._progress[-1].num_input_rows > 0,
            is_trigger_active=self._active and not self._stop.is_set(),
            message=message,
            batches_processed=self._batches,
            # From the last completed micro-batch, not re-measured: a `status()` call must
            # not touch the running fold from another thread.
            state_operators=self._progress[-1].state_operators if self._progress else (),
        )

    # --- the loop ---------------------------------------------------------
    def _run(self) -> None:
        try:
            self._run_resilient()
            self._emit_finalize()
        except BaseException as exc:
            self._error = exc
        finally:
            self._active = False
            notify_query_terminated(self._name, self._error)
            if self._sink is not None:
                with contextlib.suppress(Exception):
                    self._sink.close()
            # The checkpoint store holds two SQLite connections for the life of the query.
            # Nothing closed them, so a driver that starts and stops queries — a scheduler,
            # a notebook, a test suite — leaked two file descriptors per query until the
            # objects happened to be collected, and under WAL left the `-wal`/`-shm`
            # sidecars behind with them. The store's lifetime is exactly this loop's.
            if self._checkpoint is not None:
                with contextlib.suppress(Exception):
                    self._checkpoint.close()
            # A spilling fold owns a scratch directory with the same lifetime as this loop.
            with contextlib.suppress(Exception):
                close = getattr(self._runner, "close", None)
                if close is not None:
                    close()

    def _run_resilient(self) -> None:
        """Run the micro-batch loop, restarting from the checkpoint on a transient fault.

        A preempted worker or dropped connection raises a transient error mid-stream.
        Rather than kill the query, restore the last *committed* checkpoint (rolling
        back a half-applied micro-batch) and resume — the same exactly-once recovery a
        manual restart gives, done in-process so a spot-cluster blip doesn't end a
        long-running stream. Bounded by `recovery_max_attempts` *consecutive* restarts;
        the budget resets the moment the stream makes progress, so a flaky cluster
        self-heals while a persistently broken query still surfaces. With no checkpoint
        there is no clean restore point, so a fault surfaces immediately (unchanged); a
        non-transient error always surfaces.
        """
        restarts = 0
        while True:
            progress_mark = self._batches
            try:
                self._loop()
                return
            except self._restartable_errors() as exc:
                if self._checkpoint is None or self._stop.is_set():
                    raise
                restarts = 0 if self._batches > progress_mark else restarts + 1
                if restarts > self._max_consecutive_restarts():
                    raise
                from batcher._internal.logging import get_logger

                get_logger("core").warning(
                    "streaming fault (%s); restart %d from checkpoint at batch %d",
                    type(exc).__name__,
                    restarts,
                    self._batches,
                )
                self._recover()  # roll back to the last committed state, then replay

    @staticmethod
    def _max_consecutive_restarts() -> int:
        """How many *consecutive* (no-progress) restarts before a stream surfaces the
        fault — the profile-aware recompute budget (higher under `resilience="spot"`)."""
        from batcher.config import active_config

        return active_config().distributed.recovery_max_attempts

    @staticmethod
    def _restartable_errors() -> tuple[type[BaseException], ...]:
        """Transient fault types worth a checkpoint restart (a lost worker / dropped
        connection), not a logic error. Built lazily so `ray`/`_native` stay optional."""
        from batcher._internal.errors import ResourceError

        errs: tuple[type[BaseException], ...] = (ResourceError,)
        with contextlib.suppress(Exception):
            from batcher._internal.native import engine_or_none

            if (mod := engine_or_none()) is not None:
                errs = (*errs, mod.RetryableShuffleError)
        with contextlib.suppress(Exception):
            import ray

            errs = (*errs, ray.exceptions.RayActorError, ray.exceptions.RayTaskError)
        return errs

    def _emit_finalize(self) -> None:
        """Flush any windows still open when the loop ends, and *claim* the batch id it used.

        The flush writes to the sink under `self._batches`, the id the next micro-batch would
        have taken — and it recorded no commit, so recovery still resumed at that same id. The
        restarted query's first epoch then hit the sink's exactly-once check against a
        transaction the *previous run's flush* had already written: a Delta sink found
        ``(app_id, batch_id)`` in the log and committed nothing, a file sink found its
        ``part-batch<id>`` already present and skipped it. The whole first epoch after every
        restart was dropped, silently, and only for queries with open windows to flush.

        Recording and committing the flushed batch closes it: recovery resumes *after* it,
        and the positions written are the ones the loop had already consumed.
        """
        finalize = getattr(self._runner, "finalize", None)
        if finalize is None:
            return
        flushed = [rows for rows in finalize() if rows.num_rows]
        if not flushed:
            return
        for rows in flushed:
            self._runner.emit_final(self._batches, rows)
        if self._checkpoint is not None:
            self._checkpoint.record_offsets(self._batches, self._runner.positions())
            # Same reason as the drain marker: the snapshot must accompany the commit, or
            # recovery restores nothing for the batch it resumes after.
            self._commit_microbatch()
        self._batches += 1

    def _loop(self) -> None:
        """Drive micro-batches at the trigger's cadence until the query is over.

        `once` drains with `available_now` rather than running a single `_process_next`.
        Staging one epoch reads **one source batch**, which is an internal artifact — it
        varies with file size, poll size and morsel size — so `Trigger.once()` processed
        whatever fraction of the available data happened to land in the first batch and
        reported success. A one-shot backfill over five batches wrote one of them, with
        nothing raised and nothing in the progress record to say so.

        Spark's `Trigger.Once` processes *all* available data, which is what this
        trigger's own docstring has always promised. Spark deprecated it in favour of
        `AvailableNow` because doing that in a single micro-batch is an unbounded memory
        risk; draining across as many micro-batches as the data needs keeps the promise
        and the memory bound, so the two triggers are now the same execution under two
        names — the name kept working for a ported job, executed the safer way.
        """
        kind = self._trigger.kind
        if kind in ("once", "available_now", "continuous"):
            # Continuous: process micro-batches back-to-back with no inter-batch
            # delay (lowest latency), committing a checkpoint epoch per batch, until
            # the query is stopped or the source is exhausted. (`available_now` and
            # `once` share the loop; they are simply expected to drain what is
            # available and end — see the docstring for why `once` is one of them.)
            while not self._stop.is_set() and self._process_next():
                pass
            return
        # processing_time: fire a micro-batch, then sleep the remainder of the interval.
        interval = self._trigger.interval_seconds or 0.0
        while not self._stop.is_set():
            t0 = perf_counter()
            if not self._process_next():
                break  # bounded source exhausted
            remaining = interval - (perf_counter() - t0)
            if remaining > 0:
                self._stop.wait(remaining)

    def _process_next(self) -> bool:
        """Run one micro-batch through the runner; False when the source is spent.

        The ordering is the exactly-once contract, and it is the same whether the runner
        works on this thread or across a cluster: **stage** the epoch (read it, prepare
        its output, publish nothing), **write-ahead** the source position it consumed,
        then **publish** it. A crash can therefore only lose an epoch that was staged and
        not published — which the next run replays into an idempotent sink, so the rows
        land once and the log records one transaction for it.
        """
        t0 = perf_counter()
        staged = self._runner.stage(self._batches)
        if staged is None:
            self._checkpoint_drain()
            return False
        staged_at = perf_counter()
        start_offset = self._runner.positions()
        if self._checkpoint is not None:
            self._checkpoint.record_offsets(self._batches, start_offset)
        wal_at = perf_counter()
        consumed, emitted = self._runner.publish(self._batches, staged)
        published_at = perf_counter()
        if self._checkpoint is not None:
            self._commit_microbatch()
        end = perf_counter()
        duration_ms = (end - t0) * 1000.0
        # Keyed as Spark keys `durationMs`, because a total alone cannot distinguish a slow
        # query from a slow *checkpoint* — and those have opposite remedies. `walCommit`
        # covers both log writes (the write-ahead before publish and the commit after).
        breakdown = (
            ("latestOffset", (staged_at - t0) * 1000.0),
            ("addBatch", (published_at - wal_at) * 1000.0),
            ("walCommit", ((wal_at - staged_at) + (end - published_at)) * 1000.0),
        )
        progress = StreamingQueryProgress(
            batch_id=self._batches,
            num_input_rows=consumed,
            num_output_rows=emitted,
            duration_ms=duration_ms,
            timestamp=time(),
            behind_by_ms=self._behind_by(duration_ms),
            name=self._name,
            state_operators=self._state_metrics(),
            sources=self._source_progress(consumed, start_offset),
            sink=self._sink_progress(emitted),
            duration_breakdown_ms=breakdown,
        )
        self._progress.append(progress)
        notify_query_progress(self._name, progress)
        self._apply_rate_limit(progress)
        self._batches += 1
        return True

    def _apply_rate_limit(self, progress: StreamingQueryProgress) -> None:
        """Pace the *next* trigger from what this one measured.

        A micro-batch that overruns its interval leaves the next one starting late against a
        larger backlog, which overruns by more. The divergence compounds, and it ends not in a
        slow query but in the epoch that no longer fits in memory. The controller reads this
        batch's throughput and how far past its cadence it ran, and narrows what the source
        may hand over next time.

        Applied after the progress record is published, so a listener always sees the batch
        that *caused* a throttle before the throttle takes effect.

        Silent about a source it cannot pace: a file or in-memory source has no per-trigger
        admission to narrow, and the query runs at its configured cap as it always did.
        """
        if self._rate_controller is None:
            return
        from batcher.io.source import is_rate_limited

        if not is_rate_limited(self._source):
            return
        limit = self._rate_controller.next_limit(progress)
        if limit is not None:
            self._source.set_admission_limit(limit.max_rows)

    def _state_metrics(self) -> tuple[StateOperatorProgress, ...]:
        """What the runner's stateful operators are holding, if it has any.

        Duck-typed, like `snapshot_state` and `finalize` beside it: a stateless pipeline
        and the distributed runner simply do not define it, and report no operators.
        """
        metrics = getattr(self._runner, "state_metrics", None)
        if metrics is None:
            metrics = getattr(self._processor, "state_metrics", None)
        return tuple(metrics()) if metrics is not None else ()

    def _source_progress(
        self, consumed: int, offsets: dict[int, dict]
    ) -> tuple[SourceProgress, ...]:
        """One `SourceProgress` for the query's source, with the position it reached.

        A single source today — the launcher rejects a multi-source streaming plan — but
        reported as a tuple because Spark's shape is per-source and a stream-stream join
        will have two.
        """
        return (
            SourceProgress(
                description=_describe(self._source),
                num_input_rows=consumed,
                start_offset=offsets.get(0),
                end_offset=self._runner.positions().get(0),
            ),
        )

    def _sink_progress(self, emitted: int) -> SinkProgress | None:
        """What the sink accepted, when there is one on this side of the seam.

        None for the distributed runner, whose sinks live on the workers — reporting a
        driver-side sink there would name an object that wrote nothing.
        """
        if self._sink is None:
            return None
        token_of = getattr(self._runner, "last_sink_token", None)
        return SinkProgress(
            description=_describe(self._sink),
            num_output_rows=emitted,
            token=token_of() if token_of is not None else None,
        )

    def _behind_by(self, duration_ms: float) -> float:
        """How far this micro-batch overran the cadence it fires on, in milliseconds.

        The one question a low-latency query needs answered and the one the progress record
        could not answer: throughput says how fast a batch ran, never whether that was fast
        *enough*, because "enough" is the trigger interval and the record did not carry it.
        A trigger with no interval — `once`, `available_now`, `continuous` — has no cadence
        to be late for, so it is never behind.

        Args:
            duration_ms: How long the micro-batch took.

        Returns:
            Milliseconds over the trigger interval, or ``0.0`` when it kept up.
        """
        interval = self._trigger.interval_seconds
        if not interval:
            return 0.0
        return max(0.0, duration_ms - interval * 1000.0)

    def _checkpoint_drain(self) -> None:
        """Checkpoint where the source finally stood once it reported itself spent.

        **A drained source's final position was never recorded**, and for any source whose
        cursor advances at the *end* of a read rather than per batch that loses a whole
        window on every restart. The Delta change feed is the clear case: it reads every
        commit since the last pass as one window, streams it batch by batch, and advances its
        version only once the window has been fully handed over — deliberately, so a consumer
        that dies mid-drain replays rather than skips. The engine write-aheads the position
        after each `stage()`, and at that moment the cursor has not moved, so every batch of
        the window recorded the *pre-window* version. Recovery then resumed at the last
        committed batch id with that stale version and replayed the entire window — under
        **new** batch ids, so the sink's by-batch-id idempotency could not absorb it and the
        rows were written a second time. Silent duplication, once per restart.

        The moment the source says it is spent is the first moment its terminal position is
        both true and stable, so it is recorded here under the next batch id and committed.
        The batch carries no rows and writes to no sink; it exists so recovery resumes
        *after* the drain instead of before it.
        """
        if self._checkpoint is None:
            return
        positions = self._runner.positions()
        if not positions:
            return  # a source with no checkpointable position has nothing to record
        self._checkpoint.record_offsets(self._batches, positions)
        # Through `_commit_microbatch`, not a bare `commit`: recovery restores the snapshot
        # belonging to the *last committed* batch, so a committed batch with no snapshot
        # beside it silently resumes a stateful query with **empty** state. A drain marker
        # that carried only the position therefore fixed the position and broke the fold.
        self._commit_microbatch()
        self._batches += 1

    def _commit_microbatch(self) -> None:
        """Snapshot running state (if any), commit, and prune superseded snapshots."""
        stateful = getattr(self._runner, "has_state", None)
        snap = self._runner.snapshot_state if stateful is not None and stateful() else None
        if snap is not None:
            write_state(self, snap)
        # Record *what* the sink wrote, not merely that the batch committed. Every sink
        # returns a token and the commit log has always had a column for it; nothing
        # carried the value between them, so the column was NULL for every row ever
        # written. A runner with no single token (the distributed one, whose epoch is many
        # workers' files) simply does not define the accessor.
        token_of = getattr(self._runner, "last_sink_token", None)
        self._checkpoint.commit(self._batches, token_of() if token_of is not None else None)
        # Recovery only ever needs the newest snapshot at or before the last committed
        # batch, plus the deltas after it — so everything older goes now, and a long-running
        # stateful stream keeps a bounded `state/` dir instead of accumulating one file per
        # micro-batch forever.
        if snap is not None:
            self._checkpoint.prune_state(self._batches)
        # The offset/commit logs grow one row per micro-batch even for a *stateless* stream;
        # recovery consults only the last committed batch, so bound them the same way.
        self._checkpoint.prune_logs(self._batches)
