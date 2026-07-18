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
from typing import TYPE_CHECKING

import pyarrow as pa

from batcher._native import FlightShuffleServer as _Server
from batcher._native import ShuffleClient as _Client
from batcher._native import flight_fetch as _fetch
from batcher._native import gather_combine as _gather_combine
from batcher._native import gather_concat as _gather_concat
from batcher._native import gather_to_files as _gather_to_files

if TYPE_CHECKING:
    from batcher.carbonite.transfer.tls import ShuffleTlsMaterial

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

    def __init__(
        self,
        advertise_host: str | None = None,
        token: str | None = None,
        tls: ShuffleTlsMaterial | None = None,
    ) -> None:
        if tls is None:
            self._srv = _Server(advertise_host, token)
        else:
            # TLS-secured server: present this node's certificate, and (under mTLS)
            # require a client certificate the cluster CA signed.
            self._srv = _Server(
                advertise_host,
                token,
                tls.server_cert_pem,
                tls.server_key_pem,
                tls.client_ca_pem,
            )
        # Shuffle output volume this server has made available for reducers to fetch — the
        # network-egress magnitude a `spilled: bool` / credit window cannot show. Measurement
        # only (Carbonite/Kyber consume it to reason about shuffle cost); never a result.
        self._bytes_published = 0
        # Bytes served over the same-node fast paths (DIRECT_MEMORY / SHARED_MEMORY) — no
        # network hop. Against `bytes_published` this is the shuffle *locality* ratio: how
        # much data placement kept local (cheap) vs forced over the wire (the reducer fetches).
        self._bytes_served_locally = 0

    @property
    def addr(self) -> str:
        """The `host:port` to advertise to reducers."""
        return self._srv.addr

    def publish(self, ticket: ShuffleTicket, batches: list[pa.RecordBatch]) -> None:
        """Expose `batches` under `ticket` for reducers to fetch."""
        batches = list(batches)
        self._bytes_published += sum(b.nbytes for b in batches)
        self._srv.publish(str(ticket), batches)

    @property
    def bytes_published(self) -> int:
        """Total bytes this server has exposed for shuffle fetches — the measured shuffle
        output (network-egress) volume of this worker for the plan's lifetime."""
        return self._bytes_published

    def local_fetch(self, ticket: ShuffleTicket) -> list[pa.RecordBatch] | None:
        """Read a partition this server published, with no network hop.

        The `DIRECT_MEMORY` path for a same-process reducer. `None` if `ticket`
        was never published here, so the caller falls back to a network fetch.
        """
        batches = self._srv.local_fetch(str(ticket))
        if batches is not None:
            self._bytes_served_locally += sum(b.nbytes for b in batches)
        return batches

    @property
    def bytes_served_locally(self) -> int:
        """Bytes served over the same-node fast paths (no network) — vs `bytes_published`,
        the shuffle locality win from co-locating producer and reducer."""
        return self._bytes_served_locally

    def publish_shared(self, ticket: ShuffleTicket, batches: list[pa.RecordBatch]) -> None:
        """Mirror `ticket`'s batches to a same-node shared-memory file (Arrow IPC over
        mmap) under this server's address, so a same-node reducer in another process
        reads them without a gRPC/loopback hop. Best-effort (write errors are ignored)."""
        self._srv.publish_shared(str(ticket), list(batches))

    def shm_fetch(self, source_addr: str, ticket: ShuffleTicket) -> list[pa.RecordBatch] | None:
        """Read a partition a same-node peer published under `(source_addr, ticket)` from
        shared memory, or `None` if absent — the `SHARED_MEMORY` path. `None` means the
        caller falls back to Flight (empty bucket / un-shm'd peer / shm off)."""
        batches = self._srv.shm_fetch(source_addr, str(ticket))
        if batches is not None:
            self._bytes_served_locally += sum(b.nbytes for b in batches)
        return batches

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
        shm: bool = False,
        replicas: list[list[str]] | None = None,
    ) -> tuple[pa.RecordBatch | None, list[int]]:
        """Concurrently fetch + `combine` the aggregate partials from every source.

        Fetches every `(addr, ticket)` at once (bounded by `fan_in`), folding each into
        one running partial in Rust — so peak memory is `fan_in` in-flight fetches plus
        the running state, independent of the source count. Returns
        `(payload, unreachable)`: `payload` is the finalized batch (or the merged
        partial when `finalize` is false), or `None` when `unreachable` is non-empty
        (those sources hit a retryable fault → the driver recomputes and retries) or
        every bucket was empty. `combine` is associative, so the concurrent fold equals
        a serial one. When `shm` is set, same-node sources are read zero-copy from shared
        memory (with a Flight fallback) *inside* the concurrent set, so cross-node fetches
        still fan out.

        `replicas[i]` are the fallback addresses holding a copy of source `i`'s bucket:
        a retryable fault against one address falls over to the next, so a lost mapper is
        served from a survivor instead of recomputed. `None` ⇒ no replicas.
        """
        src = [(addr, str(ticket)) for addr, ticket in sources]
        if credits is None:
            return _gather_combine(
                self._srv,
                client._client,
                group_keys_json,
                aggregates_json,
                src,
                fan_in,
                finalize,
                shm=shm,
                replicas=replicas or [],
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
            shm,
            replicas or [],
        )

    def gather_to_files(
        self,
        client: ShuffleClient,
        sources: list[tuple[str, ShuffleTicket]],
        spill_dir: str,
        fan_in: int,
        credits: int | None = None,
        token: str | None = None,
        shm: bool = False,
        replicas: list[list[str]] | None = None,
    ) -> tuple[list[str], list[int]]:
        """Concurrently fetch every source's bucket and spill each to an IPC file under
        `spill_dir`, returning `(paths, unreachable)`.

        The bounded-memory sibling of `gather_concat`: it holds only `fan_in` in-flight
        fetches at once and lands each on disk, so a reducer whose bucket exceeds RAM
        stages it out of core for the spilling reduce (`combine_finalize_spilling`).
        """
        src = [(addr, str(ticket)) for addr, ticket in sources]
        if credits is None:
            return _gather_to_files(
                self._srv,
                client._client,
                src,
                spill_dir,
                fan_in,
                shm=shm,
                replicas=replicas or [],
            )
        return _gather_to_files(
            self._srv,
            client._client,
            src,
            spill_dir,
            fan_in,
            credits,
            token,
            shm,
            replicas or [],
        )

    def gather_concat(
        self,
        client: ShuffleClient,
        sources: list[tuple[str, ShuffleTicket]],
        fan_in: int,
        credits: int | None = None,
        token: str | None = None,
        shm: bool = False,
        replicas: list[list[str]] | None = None,
    ) -> tuple[list[pa.RecordBatch], list[int]]:
        """Concurrently fetch every source's raw batches into one list (window/sort/join).

        Like `gather_combine` but without a fold — the reducer needs the whole bucket
        and re-orders it downstream. Returns `(batches, unreachable)`; a non-empty
        `unreachable` leaves the batches partial (the driver recomputes and retries). When
        `shm` is set, same-node sources are read zero-copy from shared memory (Flight
        fallback) within the concurrent set.

        `replicas[i]` are the fallback addresses holding a copy of source `i`'s bucket, so
        a lost mapper is served from a survivor instead of recomputed. `None` ⇒ no replicas.
        """
        src = [(addr, str(ticket)) for addr, ticket in sources]
        if credits is None:
            return _gather_concat(
                self._srv, client._client, src, fan_in, shm=shm, replicas=replicas or []
            )
        return _gather_concat(
            self._srv, client._client, src, fan_in, credits, token, shm, replicas or []
        )


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
        global _BYTES_FETCHED
        if credits is None:
            batches = self._client.fetch(addr, str(ticket), token=token)
        else:
            batches = self._client.fetch(addr, str(ticket), credits, token)
        _BYTES_FETCHED += sum(b.nbytes for b in batches)
        return batches

    @property
    def connection_count(self) -> int:
        """Number of peers with a live cached channel (telemetry/tests)."""
        return self._client.connection_count


_BYTES_FETCHED = 0


def fetch(addr: str, ticket: ShuffleTicket, credits: int | None = None) -> list[pa.RecordBatch]:
    """Fetch a remote shuffle partition over credit-bounded Flight streaming.

    A one-shot fetch (fresh connection). Prefer a `ShuffleClient` for repeated
    fetches so the gRPC channel is reused. `credits` is the flow-control window
    Carbonite grants; `None` uses the engine's conservative default window.
    """
    global _BYTES_FETCHED
    batches = _fetch(addr, str(ticket)) if credits is None else _fetch(addr, str(ticket), credits)
    _BYTES_FETCHED += sum(b.nbytes for b in batches)
    return batches


def bytes_fetched() -> int:
    """Total bytes this process has fetched over the shuffle network — its reducer-side
    ingress volume, the counterpart to `FlightShuffleServer.bytes_published` (egress).
    Measurement only; Carbonite/Kyber reason about shuffle network cost from the pair."""
    return _BYTES_FETCHED
