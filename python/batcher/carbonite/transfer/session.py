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
    ) -> None:
        self._server = FlightShuffleServer(advertise_host, token)
        self._credits = credits
        self._token = token
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
        """Expose `batches` under `ticket` for reducers to fetch."""
        self._server.publish(ticket, batches)

    def fetch(self, addr: str, ticket: ShuffleTicket) -> list[pa.RecordBatch]:
        """Fetch one partition from `addr`, choosing the cheapest transfer mode.

        Same address as this session ⇒ `DIRECT_MEMORY` (read the local store, no
        serialization). Otherwise stream over credit-bounded Flight (`NETWORK`).
        The chosen mode is recorded for `locality_ratio`.
        """
        mode = select_mode(addr, self.addr)
        self._fetches += 1
        if mode is not TransferMode.NETWORK:
            self._off_network += 1
        if mode is TransferMode.DIRECT_MEMORY:
            local = self._server.local_fetch(ticket)
            return local if local is not None else []
        # SHARED_MEMORY (same-node mmap) is not yet a distinct execution path, so it
        # streams over Flight like NETWORK until the Rust shared-memory path lands.
        # The process-wide pooled client reuses one channel per peer across every
        # session's fetches. The window is adaptive when a flow controller is set.
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

    @property
    def partition_count(self) -> int:
        """Partitions currently retained by this session's server (leak tests)."""
        return self._server.partition_count

    def max_inflight(self, ticket: ShuffleTicket) -> int | None:
        """Peak in-flight batches for a locally published `ticket` (test hook)."""
        return self._server.max_inflight(ticket)
