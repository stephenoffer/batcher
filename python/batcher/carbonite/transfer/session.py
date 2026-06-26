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
    ) -> None:
        self._server = FlightShuffleServer(advertise_host, token)
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
        # Opt-in adaptive flow control: when a controller is supplied, the credit
        # window grows/shrinks per remote fetch from observed memory backpressure
        # (the AIMD law) instead of staying at the static grant. Off by default, so
        # the static path — and distributed==single-node equivalence — is unchanged.
        self._flow_control = flow_control
        self._pressure = pressure

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
        same-node reducer in another process reads it without a gRPC hop.
        """
        self._server.publish(ticket, batches)
        if self._shm and batches:
            self._server.publish_shared(ticket, batches)

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
    ) -> tuple[pa.RecordBatch | None, list[int]]:
        """Concurrently fetch + `combine` aggregate partials from every mapper.

        The reducer's bounded-memory merge, but fetching all mappers at once (bounded by
        `fan_in`) instead of one blocking round-trip each — the dominant shuffle-reduce
        cost at scale. The combine spec (`group_keys_json`/`aggregates_json`) is supplied
        by the relational layer; the session stays operator-agnostic. Returns
        `(payload, unreachable)` — a non-empty `unreachable` is the `("retry", srcs)`
        signal. When same-node shared memory is enabled it falls back to the serial,
        shm-aware path (the concurrent native gather streams same-node buckets over
        Flight rather than mmap).
        """
        if self._shm:
            return self._gather_combine_serial(group_keys_json, aggregates_json, sources, finalize)
        payload, unreachable = self._server.gather_combine(
            _process_client(),
            group_keys_json,
            aggregates_json,
            sources,
            fan_in,
            finalize,
            credits=self._window(),
            token=self._token,
        )
        self._fetches += len(sources)
        self._observe_backpressure()
        return payload, unreachable

    def gather_concat(
        self,
        sources: list[tuple[str, ShuffleTicket]],
        *,
        fan_in: int = _DEFAULT_FAN_IN,
    ) -> tuple[list[pa.RecordBatch], list[int]]:
        """Concurrently fetch every mapper's raw bucket into one list (window/sort/join).

        Like `gather`, but fetches concurrently (bounded by `fan_in`) and returns the
        lost-source indices instead of raising, so the reducer can report `("retry",
        srcs)`. Falls back to the serial, shm-aware path when shared memory is enabled.
        """
        if self._shm:
            return self._gather_concat_serial(sources)
        rows, unreachable = self._server.gather_concat(
            _process_client(), sources, fan_in, credits=self._window(), token=self._token
        )
        self._fetches += len(sources)
        self._observe_backpressure()
        return rows, unreachable

    def _gather_combine_serial(
        self, gk: str, aj: str, sources: list[tuple[str, ShuffleTicket]], finalize: bool
    ) -> tuple[pa.RecordBatch | None, list[int]]:
        """Serial shm-aware fold: fetch each source through the locality selector
        (honoring same-node shared memory) and `combine` incrementally, tracking lost
        sources. Equivalent result to the concurrent native gather (combine is
        associative), only slower — used when shm is on."""
        # `from batcher._native import …` (not `import batcher._native`) so the
        # dependency is on the compiled submodule — which Carbonite may drive to govern
        # the data plane — not the `batcher` root package (which would pull in `api` and
        # break the kyber/carbonite/core independence contract).
        from batcher._native import RetryableShuffleError, combine, combine_finalize

        running, unreachable = None, []
        for idx, (addr, ticket) in enumerate(sources):
            try:
                batches = self.fetch(addr, ticket)
            except RetryableShuffleError:
                unreachable.append(idx)
                continue
            if batches:
                merged = batches if running is None else [running, *batches]
                running = combine(gk, aj, merged)
        if unreachable:
            return (None, unreachable)
        if running is None:
            return (None, [])
        return (combine_finalize(gk, aj, [running]) if finalize else running, [])

    def _gather_concat_serial(
        self, sources: list[tuple[str, ShuffleTicket]]
    ) -> tuple[list[pa.RecordBatch], list[int]]:
        """Serial shm-aware concat: fetch each source's bucket, tracking lost ones."""
        from batcher._native import RetryableShuffleError

        rows: list[pa.RecordBatch] = []
        unreachable = []
        for idx, (addr, ticket) in enumerate(sources):
            try:
                rows += self.fetch(addr, ticket)
            except RetryableShuffleError:
                unreachable.append(idx)
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
