"""The engine's one observability event bus — every subsystem publishes here.

Batcher reports what it is doing through exactly one channel. `kyber`, `carbonite`,
`core`, `dist`, `io`, and `api` all call `publish`; the *sinks* (the terminal progress
renderer, the in-memory store behind the web UI, the JSON event log) all `subscribe`.
Neither side knows the other exists, which is what lets a new sink appear without
touching a single call site — and what keeps the bus in layer 0, importable by every
package without crossing a layer boundary.

The bus is **free when nobody is listening**. `_subscribers` is a plain tuple swapped
under a lock, so `publish` is a tuple-truthiness check on the hot path and returns
before building an `Event` when no sink is attached. That matters because progress is
published per morsel batch: a cost paid per batch must be zero by default.

Delivery is best-effort and never propagates: a sink that raises is logged once and
dropped from that emit, because observability must not be able to fail a query. Events
carry a monotonic `ts` for durations plus a wall-clock `wall` for display, since a UI
needs both and deriving one from the other after the fact is lossy.
"""

from __future__ import annotations

import contextlib
import contextvars
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "DECISION",
    "DQ",
    "GPU",
    "INFER",
    "LOG",
    "MALFORMED",
    "PARTITION",
    "POOL",
    "PROGRESS",
    "QUERY_END",
    "QUERY_START",
    "RECOVERY",
    "RECOVERY_EVENTS",
    "RESOURCE",
    "SKIPPED",
    "STAGE_END",
    "STAGE_START",
    "STREAM",
    "WRITE",
    "Event",
    "Subscriber",
    "current_query_id",
    "listening",
    "publish",
    "query_scope",
    "subscribe",
]

# --- Event kinds -------------------------------------------------------------
# String constants (not an Enum) because they cross the JSON boundary to the web UI
# verbatim; an Enum would only add a `.value` at every serialization point.

#: A query began. Fields: ``sql``/``plan`` summary, ``n_ops``, ``est_rows``.
QUERY_START = "query_start"
#: A query finished. Fields: ``rows``, ``total_ms``, ``ok``, and the profile ``dict``.
QUERY_END = "query_end"
#: An operator/stage began. Fields: ``op_id``, ``kind``, ``est_rows``.
STAGE_START = "stage_start"
#: An operator/stage finished. Fields: ``op_id``, ``rows_out``, ``elapsed_ms``, ``spilled``.
STAGE_END = "stage_end"
#: Incremental progress within a stage. Fields: ``rows``, ``bytes``, ``total`` (may be None).
PROGRESS = "progress"
#: A subsystem hand-off worth explaining (see `plan.profile.Decision`).
DECISION = "decision"
#: A `batcher.*` log record, bridged onto the bus so the UI shows logs beside metrics.
LOG = "log"

# --- Data-quality observability ----------------------------------------------
# A contract that is checked and never charted is a contract nobody notices degrading. The
# report `ds.dq.validate()` returns is a per-run value: it answers "is today's data good",
# and cannot answer "has the null rate been climbing for a week", which is the question that
# catches an upstream change before it becomes an incident. Publishing each constraint's
# result puts that series on the bus every other subsystem already reports to.

#: One data-quality constraint was evaluated. Fields: ``constraint`` (its name, also carried
#: as the event `name`), ``check`` (``row``/``unique``/``reference``/``aggregate``/
#: ``schema``), ``severity`` (``error``/``warn``), ``violations`` (violating rows, or 1 for a
#: failed relation-level or schema check), ``rows`` (rows considered), ``ok`` (whether it
#: passed after its tolerance), and ``value`` (the measured number, for a relation-level
#: check). Emitted once per constraint per `validate` — including for constraints that
#: passed, because a series that only appears when something breaks has no baseline.
DQ = "dq"

# --- Distributed / inference observability -----------------------------------
# A multi-hour batch-inference or distributed job needs progress the query/stage vocabulary
# above cannot express: *which partition* of a stage finished (so "N of M" is answerable
# while the stage runs, not only when it returns), how each GPU is loaded *right now* (not
# only at pool teardown), how fast a worker is inferring, and how many rows were *silently*
# dropped. These kinds carry that. Every subsystem on the distributed path publishes them;
# `observe` folds them into live progress and cumulative metrics. Like the kinds above they
# are plain strings because they cross the JSON boundary to the web UI verbatim.

#: One distributed partition of a stage finished. Fields: ``op_id``, ``total`` (M partitions
#: in the stage, may be None when unknown), ``rows`` (rows this partition produced). `name`
#: is the stage/operator label. Emit one per partition as it completes.
PARTITION = "partition"
#: A GPU utilization / VRAM sample from one actor. Fields: ``device`` (id/name), ``actor``
#: (actor id), ``util_pct`` (0-100), ``mem_used_bytes``, ``mem_total_bytes``. Emit on a
#: sampling interval from inside the worker, not only when the pool tears down.
GPU = "gpu"
#: One inference micro-batch completed on a worker. Fields: ``rows``, ``latency_ms``,
#: ``blocked_ms`` (time the worker waited for its next input — the pipeline-starvation
#: signal). `name` is the stage label. This is the per-batch reading `InferencePool` already
#: measures for its controller and otherwise discards.
INFER = "infer"
#: An input was dropped under ``on_error="skip"`` — one unreadable file or split, not one
#: row: the file could not be read, so how many rows it held is exactly what is unknown.
#: Fields: ``count`` (this increment), ``reason`` (the exception's type), ``source`` (the
#: format). Published by `io.base._tolerance.ErrorPolicy` as each drop is decided.
#:
#: The path is deliberately absent. A metrics label built from a path is unbounded
#: cardinality and a path can itself be sensitive; `Source.corrupt_files()` and the warning
#: log both carry it for whoever needs it. This carries the bounded fact worth alerting on,
#: because a job that quietly read 98% of its corpus produces a plausible answer and no error.
SKIPPED = "skipped"
#: Rows dropped inside a file that was otherwise read successfully — a malformed CSV line
#: under ``on_bad_lines="skip"``. Fields: ``count`` (this increment), ``reason`` (a bounded
#: cause label), ``source`` (the format). Published by the format's bad-row policy.
#:
#: Deliberately *not* folded into `SKIPPED`, which counts whole unreadable inputs. A total
#: that adds files to rows answers no question anyone has, and the two failures want
#: different responses: an unreadable file is usually infrastructure, a malformed row is
#: usually the producer upstream.
MALFORMED = "malformed"
#: Actor-pool size observation. Fields: ``size`` (live actors), ``pending`` (queued tasks).
#: `name` is the stage label. Emit on a scale-up / scale-down or on the sampling interval.
POOL = "pool"

# --- Resource-utilization observability ---------------------------------------
# Carbonite already measures every resource it governs — the buffer pool's envelope and its
# high-water mark, the spill store's per-tier bytes and free disk, the result cache's hit
# rate, the admission limiter's queue depth, the shuffle session's locality and credit
# window. Every one of those is a `stats()` method returning a plain dict of numbers, and
# before this kind existed *none of them reached the metrics export*: they were readable
# only by holding the object that owned them, so the process-wide counters could report how
# many queries spilled but not how many bytes, how full the envelope got, or whether the
# shuffle was network-bound or memory-bound.
#
# One kind carrying a whole named group rather than a constant per resource, for the same
# reason `RECOVERY` uses an `event` discriminator: the kind vocabulary crosses to the web UI
# verbatim, and the set of things Carbonite governs grows.

#: A group of resource gauges was read. `name` is the group (``memory``, ``spill``,
#: ``shuffle``, ``admission``, ``result_cache``); ``stats`` is that group's reading, the
#: dict its owner's `stats()` returns, nested at most one level deep.
#:
#: **Gauges, not counters.** Each event replaces the previous reading for its group rather
#: than adding to it, because these describe a level (bytes held, queue depth, window size)
#: and not an accumulation. A consumer that differences successive readings gets noise.
#: Publish on a query boundary or a teardown, never per batch: the reading costs a `stat`
#: of the spill volume and a walk of the pool's accounting, which is cheap once a query and
#: not cheap once a morsel.
RESOURCE = "resource"

# --- Streaming observability --------------------------------------------------
# A continuous query already produces a full Spark-parity `StreamingQueryProgress` per
# micro-batch — input and output rows, the per-phase duration breakdown, how far behind the
# trigger cadence it is, and per-operator state size. All of it was delivered *only* to a
# `StreamingQueryListener` the user had to write and register, and to `query.last_progress`.
# Neither reaches a metrics backend, which left the one workload that runs for weeks as the
# one workload a scrape loop could not see at all: no lag gauge, no state-store growth, no
# throughput series.

#: One streaming micro-batch completed. `name` is the query's name; the fields are the flat
#: numbers off `plan.streaming.StreamingQueryProgress` — ``batch_id``, ``input_rows``,
#: ``output_rows``, ``duration_ms``, ``behind_by_ms``, ``input_rows_per_second``,
#: ``processed_rows_per_second``, ``state_rows``, ``state_bytes``, and the per-phase
#: ``duration_*_ms`` breakdown.
#:
#: The counters accumulate and the rates are gauges, which is why they are carried together
#: rather than split: a lag figure without the throughput that produced it says nothing about
#: whether the query is recovering.
STREAM = "stream"

# --- Sink observability --------------------------------------------------------
# The read side of a job has always been countable and the write side never was, which is
# backwards for an ETL job: the thing it exists to produce is the thing nothing measured.
# Every sink already returns a `WrittenFile` per file carrying its row count and its size on
# storage, and `WriteManifest` already rolls them up — the numbers were there, and stopped
# at whoever held the manifest.

#: A write committed. `name` is the format (``parquet``, ``delta``, ...); fields are
#: ``files``, ``rows`` and ``bytes`` from the `io.WriteManifest`. One event per commit, from
#: the single funnel every write branch already routes through, so a partitioned write is one
#: event rather than one per file.
WRITE = "write"

# --- Fault-tolerance observability -------------------------------------------
# The distributed path has a lot of recovery machinery — lineage recompute with epoch
# fencing, speculative backups, shuffle-output replication, proactive spot-preemption
# migration — and until this kind existed it ran *entirely silently*. `ShuffleRecovery`
# counted its recomputes into an attribute nobody read; the preemption migration sat inside
# a bare `contextlib.suppress`. So a query that transparently survived losing two workers
# and one that was simply four times too slow looked identical from outside, and the only
# way to tell them apart was to already suspect the answer.
#
# One kind with an `event` discriminator rather than seven constants: the kind vocabulary
# crosses to the web UI verbatim, and seven entries for one concern would swamp it.

#: A fault-tolerance action happened on the distributed path. Every publisher runs on the
#: driver, where the decision is made — nothing subscribes inside a Ray worker, so an event
#: published there would be silently dropped.
#:
#: Fields: ``event`` (one of `RECOVERY_EVENTS`), ``shuffle`` (``aggregate``/``join``/
#: ``sort``/``window``), and then per-event detail: ``worker``/``src`` (the lost or
#: relocated source), ``epoch`` (the fence a recompute bumped), ``round`` and
#: ``attempt_budget`` (which recovery round, of how many).
RECOVERY = "recovery"

#: The `event` values a `RECOVERY` event may carry.
#:
#: - ``worker_lost``: a worker was first observed dead, and its buckets are gone with it.
#: - ``recompute``: a recovery round re-ran a lost source and bumped its epoch.
#: - ``straggler_backup``: a speculative duplicate was launched for a slow task.
#: - ``doomed_backup``: a speculative duplicate was launched for a task whose *host is being
#:   reclaimed*, before it looked slow at all. Distinguished from ``straggler_backup``
#:   because the two mean opposite things operationally — one says a node is degraded, the
#:   other says a node is leaving on schedule — and counting them together makes a healthy
#:   autoscaling cluster read as a sick one.
#: - ``backup_won``: the speculative duplicate finished first; the original was cancelled.
#: - ``replica_retired``: a stale replica was dropped before its source was reincarnated.
#:   The one that most needs to be visible — reading a retired replica returns an *empty
#:   bucket rather than an error*, so this is the last event before a silent row loss would
#:   have happened.
#: - ``preempt_migrate``: a draining (spot-preempted) worker's work moved before it died.
#: - ``give_up``: the recovery budget was exhausted and the query is failing.
#: - ``budget_exhausted``: the job-wide retry budget ran out, so the next failure is raised
#:   rather than retried. Published once, on the transition — every task still in flight will
#:   ask again, and an event per refusal would bury the one that matters.
#: - ``quarantined``: a node or device stopped being scheduled because work placed on it kept
#:   failing. Carries ``target``, the decayed failure ``weight``, and the ``reasons`` seen.
#: - ``released``: a quarantined target succeeded on probation and is back in rotation. The
#:   counterpart of ``quarantined``, and the one that says a fleet is recovering rather than
#:   shrinking.
#: - ``oom_backoff``: an accelerator refused an allocation and the work is being retried at a
#:   smaller size. Carries ``failed_rows`` and ``retry_rows``. The only record that a stage was
#:   memory-bound rather than merely slow — it produced the right answer either way.
#: - ``quarantine_capped``: a target met the quarantine threshold and was *not* taken out,
#:   because doing so would have exceeded the blast-radius cap. The signal that the failures
#:   have gone systemic — at that point the job's own cause is the likelier culprit, and
#:   condemning more of the fleet only turns a degraded run into a dead one.
#: - ``shard_degraded``: a GPU fan-out finished, but not every shard ran the way it was meant
#:   to — some were subdivided to fit a device, or recomputed on the CPU engine because their
#:   device was lost. The answer is the same either way, which is exactly why it needs an
#:   event: a run where a third of the shards fell back to the host is a very different run
#:   from one where none did, and the two are otherwise indistinguishable from the result.
RECOVERY_EVENTS = (
    "worker_lost",
    "recompute",
    "straggler_backup",
    "doomed_backup",
    "backup_won",
    "replica_retired",
    "preempt_migrate",
    "give_up",
    "shard_degraded",
    "budget_exhausted",
    "quarantined",
    "released",
    "quarantine_capped",
    "oom_backoff",
)


@dataclass(frozen=True, slots=True)
class Event:
    """One thing that happened, timestamped and addressed to a query.

    `kind` is one of the module-level constants; `fields` carries the kind-specific
    payload (documented on each constant). `query_id` is the id the event log assigns,
    or `""` for engine-level events that precede a query.
    """

    kind: str
    #: Monotonic seconds — safe to subtract for durations, meaningless as a date.
    ts: float
    #: Unix wall-clock seconds — meaningful as a date, unsafe to subtract.
    wall: float
    query_id: str = ""
    name: str = ""
    fields: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """The event as a JSON-encodable dict (the wire shape the web UI consumes)."""
        return {
            "kind": self.kind,
            "ts": self.ts,
            "wall": self.wall,
            "query_id": self.query_id,
            "name": self.name,
            "fields": self.fields,
        }


#: A sink: called with each `Event`. Must not raise; if it does, it is skipped.
Subscriber = Callable[[Event], None]

# Swapped wholesale under `_lock` rather than mutated, so `publish` can read it without
# taking the lock at all — the reader sees either the old tuple or the new one, never a
# half-mutated list. This is what keeps per-batch progress publishing lock-free.
_subscribers: tuple[Subscriber, ...] = ()
_lock = threading.Lock()

# Re-entrancy guard, per thread. Publishing can re-enter itself: a sink raises, `publish`
# logs that at DEBUG, the logging bridge turns the record into a LOG event, and that event
# goes back to the same failing sink. Left open, that is unbounded recursion — a broken sink
# at DEBUG verbosity blew the stack and took the query down with it, which is precisely the
# failure this module promises cannot happen. The guard makes the nested publish a no-op, so
# the first failure is still reported and the cycle cannot form.
_publishing = threading.local()


def listening() -> bool:
    """Whether any sink is attached — the guard for building an expensive payload.

    `publish` performs this check itself; call it directly only to skip *computing* the
    fields (a row count that costs something to obtain) rather than to skip the emit.

    Examples:
        .. doctest::

            >>> from batcher._internal.events import listening
            >>> listening()
            False
    """
    return bool(_subscribers)


def subscribe(sink: Subscriber) -> Callable[[], None]:
    """Attach `sink` to the bus and return a callable that detaches it.

    The returned unsubscribe is idempotent, so a sink shut down twice (a UI server
    stopped, then stopped again at interpreter exit) is not an error.

    Args:
        sink: Called with every subsequent `Event`.

    Returns:
        A zero-argument callable that removes `sink` from the bus.
    """
    global _subscribers
    with _lock:
        _subscribers = (*_subscribers, sink)

    def _unsubscribe() -> None:
        global _subscribers
        with _lock:
            _subscribers = tuple(s for s in _subscribers if s is not sink)

    return _unsubscribe


# The query a publisher's events belong to, when one is in flight. Ambient because the
# subsystems that have the most to say about a query — `dist` deciding a fan-out, a placement,
# a transport — are the ones furthest from where the id is minted, and `dist` MUST NOT import
# `api` to ask for it.
#
# This is what `observe.store` matches against: it drops any event whose id names no live
# record, silently and by design, so that a late event cannot resurrect an aged-out query as a
# ghost. Without an ambient id every scheduling `Decision` published from `dist` reached the
# bus and was discarded by the one consumer that matters — which is why `FanoutTrace` reported
# to the logger instead and documented the gap rather than filling it.
_QUERY_ID: contextvars.ContextVar[str] = contextvars.ContextVar("batcher_query_id", default="")


def current_query_id() -> str:
    """The id of the query in flight on this context, or `""` when none is.

    Returns:
        The ambient query id, empty outside any `query_scope`.
    """
    return _QUERY_ID.get()


@contextlib.contextmanager
def query_scope(query_id: str) -> Iterator[str]:
    """Make `query_id` the ambient owner of every event published inside the block.

    Nesting is allowed and the innermost wins, which is what a sub-query (an adaptive stage
    re-run, a `ds.dq` probe) should see. An empty id is a no-op rather than an error, so a
    caller that has not minted one yet does not have to branch.

    Args:
        query_id: The id the event log assigned, or `""` to leave the scope unchanged.

    Yields:
        The id now in force.
    """
    if not query_id:
        yield _QUERY_ID.get()
        return
    token = _QUERY_ID.set(query_id)
    try:
        yield query_id
    finally:
        _QUERY_ID.reset(token)


def publish(kind: str, *, query_id: str = "", name: str = "", **fields: Any) -> None:
    """Emit an event to every attached sink; a no-op when none are attached.

    Best-effort by contract: a sink that raises is reported at DEBUG and skipped, so a
    broken UI or a full disk can never fail the query that was being observed.

    Args:
        kind: One of the module-level kind constants.
        query_id: The owning query's id. Left empty, the ambient `query_scope` id is used,
            so a subsystem too far from the conductor to be handed one still attributes its
            events to the right query; empty with no scope means an engine-level event.
        name: A short human label (the operator kind, the stage name).
        **fields: The kind-specific payload; must be JSON-encodable.
    """
    sinks = _subscribers
    if not sinks:
        return
    query_id = query_id or _QUERY_ID.get()
    if getattr(_publishing, "active", False):
        # Already delivering on this thread — this call is a sink's own side effect (almost
        # always the failure log). Dropping it keeps delivery acyclic; see `_publishing`.
        return
    event = Event(
        kind=kind,
        ts=time.monotonic(),
        wall=time.time(),
        query_id=query_id,
        name=name,
        fields=fields,
    )
    _publishing.active = True
    try:
        for sink in sinks:
            try:
                sink(event)
            except Exception:  # pragma: no cover - a sink must never fail a query
                _report_sink_failure()
    finally:
        _publishing.active = False


def _report_sink_failure() -> None:
    """Log a sink exception at DEBUG, without recursing back onto the bus.

    Imported lazily and called only on the failure path: `logging` bridges records *onto*
    the bus, so doing this eagerly at module scope would make the two modules mutually
    importable at load time for no benefit.
    """
    from batcher._internal.logging import get_logger

    get_logger("observe").debug("event sink raised; skipped", exc_info=True)
