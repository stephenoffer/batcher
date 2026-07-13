"""The ShuffleSession — Carbonite's operator-agnostic data-movement engine.

A session owns one node-local Flight server and moves shuffle partitions between
workers under two Carbonite governors: the credit window (flow control, bounds a
channel's memory) and the locality selector (move co-located data without a network
hop). It is *operator-agnostic* — it ships opaque `RecordBatch`es for whatever
partition/`partial`/`combine` the relational layer supplies, so aggregation, join,
and sort shuffles all reuse one engine.

It depends only on `batcher._native` and the sibling transfer modules, so it is
usable and testable on its own: spin up N sessions in one process, publish, and
`gather` — no Ray, no `dist`, no optimizer or executor. That standalone shape is
what makes Carbonite a transfer sublibrary rather than glue inside the engine.
"""

from __future__ import annotations

import atexit
import contextlib
import threading
import weakref
from typing import TYPE_CHECKING

import pyarrow as pa

from batcher.carbonite.transfer.locality import (
    TransferMode,
    locality_ratio_counts,
    select_mode,
)
from batcher.carbonite.transfer.server import FlightShuffleServer, ShuffleClient, ShuffleTicket

if TYPE_CHECKING:
    from batcher.carbonite.memory.pressure import PressureMonitor
    from batcher.carbonite.policies import AIMDFlowControl
    from batcher.carbonite.transfer.tls import ShuffleTlsMaterial

__all__ = ["ShuffleSession"]

# Default concurrent-fetch fan-in for a reducer's gather: at most this many mapper
# fetches stream at once, bounding peak memory to ~fan_in in-flight buckets plus the
# running state. Matches the `shuffle_fan_in` config default (the same Carbonite
# fan-in governor as the combiner tree).
_DEFAULT_FAN_IN = 8

# One pooled consumer per process, shared by every session: its channel pool is
# keyed by peer address, so sharing is correct, and it bounds the process to a
# single client runtime no matter how many sessions exist (a per-session runtime
# would accumulate background threads and destabilize a many-actor worker).
_shared_client: ShuffleClient | None = None


def _process_client() -> ShuffleClient:
    global _shared_client
    if _shared_client is None:
        _shared_client = ShuffleClient()
    return _shared_client


# Every live session, so a process-exit hook can retire the pyarrow-backed batches
# their Flight servers still hold *while the interpreter is alive*. A published
# partition is a zero-copy view of a pyarrow array whose release callback needs the
# GIL; if a tokio serve thread drops it *after* Python has begun finalizing, the
# GIL acquire turns into a thread-exit that unwinds through Rust and aborts the
# process (`std::terminate`). Clearing here — on the main thread, GIL held, before
# finalization — drops that data first, so no background thread touches the GIL at
# shutdown. WeakSet ⇒ an already-collected session drops out on its own.
_live_sessions: weakref.WeakSet = weakref.WeakSet()
_atexit_registered = False


def _drain_live_sessions() -> None:
    """Evict every live session's published partitions at interpreter exit."""
    for session in list(_live_sessions):
        with contextlib.suppress(Exception):  # best-effort teardown must never raise
            session.clear()


def _register_session(session: ShuffleSession) -> None:
    """Track `session` for exit-time draining, registering the hook once."""
    global _atexit_registered
    _live_sessions.add(session)
    if not _atexit_registered:
        atexit.register(_drain_live_sessions)
        _atexit_registered = True


def _host(addr: str) -> str:
    """The node identity of a shuffle address — its host, dropping the `:port` (the
    advertised address is `{node_ip}:{port}`, so equal hosts ⇒ same node)."""
    return addr.rsplit(":", 1)[0]


class ShuffleSession:
    """Moves shuffle partitions for one worker, credit-bounded and locality-aware.

    Construct one per worker process. Mappers `publish` their output buckets;
    reducers `fetch`/`gather` their bucket from every upstream. A fetch from this
    session's own server takes the `DIRECT_MEMORY` path (no socket); a remote fetch
    streams over credit-bounded Flight with the window Carbonite granted.
    """

    def __init__(
        self,
        credits: int | None = None,
        *,
        flow_control: AIMDFlowControl | None = None,
        pressure: PressureMonitor | None = None,
        advertise_host: str | None = None,
        token: str | None = None,
        shm: bool = False,
        tls: ShuffleTlsMaterial | None = None,
    ) -> None:
        self._server = FlightShuffleServer(advertise_host, token, tls)
        self._credits = credits
        self._token = token
        # Same-node shared-memory transfer (opt-in): a mapper mirrors each bucket to an
        # mmap'd Arrow IPC file, and a same-node reducer in another process reads it
        # without a gRPC/loopback hop. Off by default ⇒ no shm writes, behavior
        # unchanged. A reducer detects "same node" by comparing the host of the peer's
        # advertised address to its own (the address already carries the node IP).
        self._shm = shm
        # Locality is tracked as two counters, not a per-fetch list: a long-lived
        # reducer does an unbounded number of fetches, so an append-per-fetch list
        # would grow without bound (C13). off_network / total reconstruct the ratio.
        self._off_network = 0
        self._fetches = 0
        # Guards the read-modify-write bookkeeping (locality counters + the AIMD window)
        # against the concurrent gathers a join reducer issues for its two sides.
        self._stats_lock = threading.Lock()
        # Opt-in adaptive flow control: when a controller is supplied, the credit
        # window grows/shrinks per remote fetch from observed memory backpressure
        # (the AIMD law) instead of staying at the static grant. Off by default, so
        # the static path — and distributed==single-node equivalence — is unchanged.
        self._flow_control = flow_control
        self._pressure = pressure
        # Shared memory writes a second (tmpfs) copy of each bucket, so it must be
        # pressure-aware. If shm is on but no monitor was supplied (the non-adaptive
        # path), attach one purely to gate the shm mirror — it drives no flow control
        # (that needs `flow_control`), so behavior is otherwise unchanged.
        if shm and self._pressure is None:
            from batcher.carbonite.memory.pressure import PressureMonitor

            self._pressure = PressureMonitor()
        _register_session(self)

    def set_credits(self, credits: int | None) -> None:
        """Re-grant this session's static credit window.

        The window is read per fetch (`_window`), never baked into the Flight server, so a
        session outliving one query can be re-granted for the next instead of being torn
        down. That is what lets a *reused* shuffle fleet serve a query other than the one
        that spawned it: without it, a fleet spawned by a global `COUNT(*)` (granted 1
        credit — one in-flight batch) keeps that window for every later query on it, and
        the next join's exchange serializes behind it.

        A no-op under adaptive flow control, which owns the window itself (`_flow_control`).

        Args:
            credits: The new static credit window (1 credit = 1 in-flight batch).
        """
        if self._flow_control is None:
            self._credits = credits

    def _window(self) -> int | None:
        """The credit window for the next fetch — adaptive when a controller is set."""
        return self._flow_control.window if self._flow_control is not None else self._credits

    def _observe_backpressure(self) -> None:
        """Feed one round's congestion signal to the AIMD controller (if adaptive).

        Congestion = memory past the soft (spill) threshold: cut the window to
        relieve pressure; otherwise grow it. This consumes the measured
        `PressureMonitor.level()` — the signal that was previously gathered but never
        acted on."""
        if self._flow_control is None:
            return
        congested = False
        if self._pressure is not None:
            from batcher.carbonite.memory.pressure import PressureLevel

            congested = self._pressure.level() >= PressureLevel.SPILL
        self._flow_control.observe(congested=congested)

    @property
    def addr(self) -> str:
        """The `host:port` to advertise so reducers can fetch from this session."""
        return self._server.addr

    def publish(self, ticket: ShuffleTicket, batches: list[pa.RecordBatch]) -> None:
        """Expose `batches` under `ticket` for reducers to fetch.

        When shared memory is on, also mirror the bucket to an mmap'd file so a
        same-node reducer in another process reads it without a gRPC hop — unless the
        node is under memory pressure, where the extra tmpfs copy is skipped (the
        reducer falls back to Flight, which stays correct).
        """
        self._server.publish(ticket, batches)
        if self._shm and batches and self._shm_mirror_ok():
            self._server.publish_shared(ticket, batches)

    def _shm_mirror_ok(self) -> bool:
        """Whether to mirror this bucket to shared memory now.

        The shm file is a *second* copy of the bucket (in tmpfs = RAM) on top of the
        in-memory store Flight serves remote reducers from, so under memory pressure it
        is skipped to protect a tight node from OOM — the case that matters on churning
        spot clusters, where recompute transiently doubles live state. Without a pressure
        monitor (the non-adaptive path) it is always allowed; the ample-memory common
        case keeps the same-node fast path.
        """
        if self._pressure is None:
            return True
        from batcher.carbonite.memory.pressure import PressureLevel

        # Pure read — `_observe_backpressure` is this session's sampler.
        return self._pressure.classify() < PressureLevel.SPILL

    def fetch(self, addr: str, ticket: ShuffleTicket) -> list[pa.RecordBatch]:
        """Fetch one partition from `addr`, choosing the cheapest transfer mode.

        Same address ⇒ `DIRECT_MEMORY` (local store, no socket). Same node, different
        process (shared memory on) ⇒ `SHARED_MEMORY` (mmap'd Arrow IPC, no gRPC), with a
        Flight fallback when the peer didn't shm the bucket. Otherwise credit-bounded
        Flight (`NETWORK`). The chosen mode is recorded for `locality_ratio`.
        """
        # Pass node identity (the address host) only when shm is on, so the default
        # path's mode selection — and behavior — is exactly as before.
        if self._shm:
            mode = select_mode(
                addr, self.addr, source_node=_host(addr), local_node=_host(self.addr)
            )
        else:
            mode = select_mode(addr, self.addr)
        self._fetches += 1
        if mode is TransferMode.DIRECT_MEMORY:
            self._off_network += 1
            local = self._server.local_fetch(ticket)
            return local if local is not None else []
        if mode is TransferMode.SHARED_MEMORY:
            shared = self._server.shm_fetch(addr, ticket)
            if shared is not None:  # a miss (empty/un-shm'd bucket) falls back to Flight
                self._off_network += 1
                return shared
        # NETWORK (or a shared-memory miss): stream over credit-bounded Flight. The
        # process-wide pooled client reuses one channel per peer across every session's
        # fetches. The window is adaptive when a flow controller is set.
        out = _process_client().fetch(addr, ticket, credits=self._window(), token=self._token)
        self._observe_backpressure()
        return out

    def gather(self, sources: list[tuple[str, ShuffleTicket]]) -> list[pa.RecordBatch]:
        """Fetch from every `(addr, ticket)` and concatenate into one batch list.

        The reducer pattern: pull this reducer's bucket from every mapper. A mapper
        that produced no rows for this bucket never published the ticket; the
        transport resolves that *expected* empty-bucket case to an empty result, so
        it contributes nothing here. Any *other* fetch failure (an unreachable peer,
        a decode error) propagates — a real fault must not be silently swallowed into
        an empty bucket, which would yield wrong results.
        """
        out: list[pa.RecordBatch] = []
        for addr, ticket in sources:
            out += self.fetch(addr, ticket)
        return out

    def gather_combine(
        self,
        group_keys_json: str,
        aggregates_json: str,
        sources: list[tuple[str, ShuffleTicket]],
        *,
        finalize: bool,
        fan_in: int = _DEFAULT_FAN_IN,
        replicas: list[list[str]] | None = None,
    ) -> tuple[pa.RecordBatch | None, list[int]]:
        """Concurrently fetch + `combine` aggregate partials from every mapper.

        The reducer's bounded-memory merge, but fetching all mappers at once (bounded by
        `fan_in`) instead of one blocking round-trip each — the dominant shuffle-reduce
        cost at scale. The combine spec (`group_keys_json`/`aggregates_json`) is supplied
        by the relational layer; the session stays operator-agnostic. Returns
        `(payload, unreachable)` — a non-empty `unreachable` is the `("retry", srcs)`
        signal. When same-node shared memory is enabled, same-node sources are read
        zero-copy from shared memory *inside* the concurrent gather (Flight fallback on a
        miss), so cross-node fetches still fan out in parallel.

        `replicas[i]` are the peers holding a copy of mapper `i`'s bucket: a lost mapper is
        then served from a survivor rather than reported unreachable, so the driver pays no
        recompute at all. A source is `unreachable` only once every copy of it is gone.
        """
        payload, unreachable = self._server.gather_combine(
            _process_client(),
            group_keys_json,
            aggregates_json,
            sources,
            fan_in,
            finalize,
            credits=self._window(),
            token=self._token,
            shm=self._shm,
            replicas=replicas,
        )
        self._fetches += len(sources)
        self._observe_backpressure()
        return payload, unreachable

    def gather_concat(
        self,
        sources: list[tuple[str, ShuffleTicket]],
        *,
        fan_in: int | None = None,
        replicas: list[list[str]] | None = None,
    ) -> tuple[list[pa.RecordBatch], list[int]]:
        """Concurrently fetch every mapper's raw bucket into one list (window/sort/join).

        Like `gather`, but fetches concurrently (bounded by `fan_in`) and returns the
        lost-source indices instead of raising, so the reducer can report `("retry",
        srcs)`. When shared memory is enabled, same-node sources are read zero-copy from
        shared memory within the concurrent gather (Flight fallback on a miss).

        `fan_in` defaults to `flow_control.shuffle_fetch_fan_in` — the *flat*-gather bound,
        not the combiner tree's `shuffle_fan_in`. This gather concatenates every source, so
        the reducer holds all of it no matter how few peers stream at once: throttling to
        the tree's fan-in only idled the network. In-flight memory is still bounded by the
        per-channel credit window.

        `replicas[i]` are the peers holding a copy of mapper `i`'s bucket: a lost mapper is
        served from a survivor rather than reported unreachable, so the driver pays no
        recompute. A source is `unreachable` only once every copy of it is gone.
        """
        if fan_in is None:
            from batcher.config import active_config

            fan_in = max(1, active_config().flow_control.shuffle_fetch_fan_in)
        fan_in = min(fan_in, max(1, len(sources)))  # never dial more peers than exist
        rows, unreachable = self._server.gather_concat(
            _process_client(),
            sources,
            fan_in,
            credits=self._window(),
            token=self._token,
            shm=self._shm,
            replicas=replicas,
        )
        # A join reducer gathers its left and right sides on two threads at once (they are
        # independent streams, so serializing them idled the link), and both land here. The
        # fetch itself is outside the lock — it is the part that must overlap — but the AIMD
        # window and the locality counter are read-modify-write, so guard just the bookkeeping.
        with self._stats_lock:
            self._fetches += len(sources)
            self._observe_backpressure()
        return rows, unreachable

    @property
    def locality_ratio(self) -> float:
        """Fraction of this session's fetches that stayed off the network so far.

        Empty (no fetches yet) reports 1.0 by the shared `locality_ratio` convention.
        """
        return locality_ratio_counts(self._off_network, self._fetches)

    def release(self, ticket: ShuffleTicket) -> None:
        """Evict one published partition once its reducers have fetched it (C8)."""
        self._server.release(ticket)

    def clear_plan(self, plan_id: int) -> None:
        """Evict every partition for `plan_id` at plan teardown (C8/C9)."""
        self._server.clear_plan(plan_id)

    def clear(self) -> None:
        """Evict every published partition on this session's server."""
        self._server.clear()
        if self._shm:
            self._server.clear_shared()

    @property
    def partition_count(self) -> int:
        """Partitions currently retained by this session's server (leak tests)."""
        return self._server.partition_count

    def max_inflight(self, ticket: ShuffleTicket) -> int | None:
        """Peak in-flight batches for a locally published `ticket` (test hook)."""
        return self._server.max_inflight(ticket)
