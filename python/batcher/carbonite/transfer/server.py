"""The node-local Arrow Flight shuffle server — Carbonite's transfer endpoint.

One server per worker process hosts that worker's shuffle output partitions and
serves them to reducers over credit-bounded Flight, **without the Ray object store
ever holding a `RecordBatch`** — only the small `(addr, ticket)` strings transit
Ray's control path. This is the byte-moving endpoint the `ShuffleSession` drives;
it wraps the Rust `bc-transport` engine (`batcher._native`).

`local_fetch` is the same-process fast path: a reducer co-located with a mapper
reads the partition straight from this server's store with no socket hop.
"""

from __future__ import annotations

from dataclasses import dataclass

import pyarrow as pa

from batcher._native import FlightShuffleServer as _Server
from batcher._native import ShuffleClient as _Client
from batcher._native import flight_fetch as _fetch
from batcher._native import gather_combine as _gather_combine
from batcher._native import gather_concat as _gather_concat

__all__ = ["FlightShuffleServer", "ShuffleClient", "ShuffleTicket", "fetch"]


@dataclass(frozen=True, slots=True)
class ShuffleTicket:
    """Identifies one shuffle output partition: `plan/stage/src/dst/epoch`."""

    plan_id: int
    stage_id: int
    src_partition: int
    dst_partition: int
    epoch: int = 0

    def __str__(self) -> str:
        return (
            f"{self.plan_id}/{self.stage_id}/{self.src_partition}/{self.dst_partition}/{self.epoch}"
        )


class FlightShuffleServer:
    """A node-local Flight server hosting this worker's shuffle outputs.

    `advertise_host` is the node's routable address (the Ray node IP): when set the
    server binds all interfaces and advertises `{advertise_host}:{port}` so reducers
    on other nodes can reach it. Omitted/empty keeps single-host loopback behavior.
    """

    def __init__(self, advertise_host: str | None = None, token: str | None = None) -> None:
        self._srv = _Server(advertise_host, token)

    @property
    def addr(self) -> str:
        """The `host:port` to advertise to reducers."""
        return self._srv.addr

    def publish(self, ticket: ShuffleTicket, batches: list[pa.RecordBatch]) -> None:
        """Expose `batches` under `ticket` for reducers to fetch."""
        self._srv.publish(str(ticket), list(batches))

    def local_fetch(self, ticket: ShuffleTicket) -> list[pa.RecordBatch] | None:
        """Read a partition this server published, with no network hop.

        The `DIRECT_MEMORY` path for a same-process reducer. `None` if `ticket`
        was never published here, so the caller falls back to a network fetch.
        """
        return self._srv.local_fetch(str(ticket))

    def publish_shared(self, ticket: ShuffleTicket, batches: list[pa.RecordBatch]) -> None:
        """Mirror `ticket`'s batches to a same-node shared-memory file (Arrow IPC over
        mmap) under this server's address, so a same-node reducer in another process
        reads them without a gRPC/loopback hop. Best-effort (write errors are ignored)."""
        self._srv.publish_shared(str(ticket), list(batches))

    def shm_fetch(self, source_addr: str, ticket: ShuffleTicket) -> list[pa.RecordBatch] | None:
        """Read a partition a same-node peer published under `(source_addr, ticket)` from
        shared memory, or `None` if absent — the `SHARED_MEMORY` path. `None` means the
        caller falls back to Flight (empty bucket / un-shm'd peer / shm off)."""
        return self._srv.shm_fetch(source_addr, str(ticket))

    def clear_shared(self) -> None:
        """Remove every shared-memory file this server published (plan teardown)."""
        self._srv.clear_shared()

    def max_inflight(self, ticket: ShuffleTicket) -> int | None:
        """Peak number of batches the producer ever had in flight for `ticket`.

        `None` if the ticket was never published. Lets a test assert the
        credit-flow-control bound: this never exceeds the granted window.
        """
        return self._srv.max_inflight(str(ticket))

    def release(self, ticket: ShuffleTicket) -> None:
        """Evict one published partition once its reducers have fetched it."""
        self._srv.release(str(ticket))

    def clear_plan(self, plan_id: int) -> None:
        """Evict every partition for `plan_id` at plan teardown."""
        self._srv.clear_plan(plan_id)

    def clear(self) -> None:
        """Evict every published partition on this server."""
        self._srv.clear()

    @property
    def partition_count(self) -> int:
        """Partitions currently retained (telemetry / leak tests)."""
        return self._srv.partition_count

    def gather_combine(
        self,
        client: ShuffleClient,
        group_keys_json: str,
        aggregates_json: str,
        sources: list[tuple[str, ShuffleTicket]],
        fan_in: int,
        finalize: bool,
        credits: int | None = None,
        token: str | None = None,
    ) -> tuple[pa.RecordBatch | None, list[int]]:
        """Concurrently fetch + `combine` the aggregate partials from every source.

        Fetches every `(addr, ticket)` at once (bounded by `fan_in`), folding each into
        one running partial in Rust — so peak memory is `fan_in` in-flight fetches plus
        the running state, independent of the source count. Returns
        `(payload, unreachable)`: `payload` is the finalized batch (or the merged
        partial when `finalize` is false), or `None` when `unreachable` is non-empty
        (those sources hit a retryable fault → the driver recomputes and retries) or
        every bucket was empty. `combine` is associative, so the concurrent fold equals
        a serial one.
        """
        src = [(addr, str(ticket)) for addr, ticket in sources]
        if credits is None:
            return _gather_combine(
                self._srv, client._client, group_keys_json, aggregates_json, src, fan_in, finalize
            )
        return _gather_combine(
            self._srv,
            client._client,
            group_keys_json,
            aggregates_json,
            src,
            fan_in,
            finalize,
            credits,
            token,
        )

    def gather_concat(
        self,
        client: ShuffleClient,
        sources: list[tuple[str, ShuffleTicket]],
        fan_in: int,
        credits: int | None = None,
        token: str | None = None,
    ) -> tuple[list[pa.RecordBatch], list[int]]:
        """Concurrently fetch every source's raw batches into one list (window/sort/join).

        Like `gather_combine` but without a fold — the reducer needs the whole bucket
        and re-orders it downstream. Returns `(batches, unreachable)`; a non-empty
        `unreachable` leaves the batches partial (the driver recomputes and retries).
        """
        src = [(addr, str(ticket)) for addr, ticket in sources]
        if credits is None:
            return _gather_concat(self._srv, client._client, src, fan_in)
        return _gather_concat(self._srv, client._client, src, fan_in, credits, token)


class ShuffleClient:
    """A pooled, persistent shuffle consumer.

    Holds one tokio runtime and a gRPC channel pool for its lifetime, so a
    reducer's many fetches reuse connections (one per peer) instead of
    reconnecting per partition. Hold one per reducer and fetch through it; the
    connection cost is then O(peers), not O(partitions) — what makes an all-to-all
    shuffle scale to a large cluster.
    """

    def __init__(self) -> None:
        self._client = _Client()

    def fetch(
        self,
        addr: str,
        ticket: ShuffleTicket,
        credits: int | None = None,
        token: str | None = None,
    ) -> list[pa.RecordBatch]:
        """Fetch a remote partition over a credit-bounded stream on a pooled channel.

        `token` is the shuffle auth secret presented to an auth-gated peer (N5).
        """
        if credits is None:
            return self._client.fetch(addr, str(ticket), token=token)
        return self._client.fetch(addr, str(ticket), credits, token)

    @property
    def connection_count(self) -> int:
        """Number of peers with a live cached channel (telemetry/tests)."""
        return self._client.connection_count


def fetch(addr: str, ticket: ShuffleTicket, credits: int | None = None) -> list[pa.RecordBatch]:
    """Fetch a remote shuffle partition over credit-bounded Flight streaming.

    A one-shot fetch (fresh connection). Prefer a `ShuffleClient` for repeated
    fetches so the gRPC channel is reused. `credits` is the flow-control window
    Carbonite grants; `None` uses the engine's conservative default window.
    """
    if credits is None:
        return _fetch(addr, str(ticket))
    return _fetch(addr, str(ticket), credits)
