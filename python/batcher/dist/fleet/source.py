"""A relation whose batches stay partitioned on the shuffle fleet between stages.

A Flight stage run with ``materialize=False`` leaves each reducer's finalized bucket
published on its host actor's Flight server and returns a `FlightMaterializedSource`
over `(addr, ticket, rows)` handles. The next adaptive stage scans it shared-nothing
via `FlightFetchSplit`s — each next-stage worker fetches its bucket straight from the
holding actor — instead of the driver collecting every reducer's output. Its exact
`row_count` (summed from the per-bucket Arrow `num_rows`, never a Python row scan)
feeds the optimizer with EXACT provenance.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import pyarrow as pa

from batcher.carbonite.transfer import ShuffleTicket

__all__ = ["FlightFetchSplit", "FlightMaterializedSource"]


#: Bounded wait for the per-stage bucket release. Short on purpose: freeing an
#: intermediate is memory hygiene, and a slow worker must not hold up the next stage.
_RELEASE_TIMEOUT_S = 2.0


@dataclass(frozen=True, slots=True)
class FlightFetchSplit:
    """One reducer's result bucket, read locator-only over Arrow Flight from the
    worker actor that still hosts it — the shared-nothing unit a
    [`FlightMaterializedSource`] advertises (each next-stage worker fetches its bucket
    straight from the holding actor, never through the driver). `schema_` is carried
    so an empty bucket still yields a schema."""

    addr: str
    ticket: ShuffleTicket
    rows: int
    schema_: pa.Schema

    def schema(self) -> pa.Schema:
        return self.schema_

    def affinity(self) -> str:
        """The shuffle address holding this bucket — the worker that reads it for free.

        Consumed by the locality-aware split assignment
        (`partition_io.affinity.balance_with_affinity`), which routes a bucket to the
        worker already holding it so `read` takes the `DIRECT_MEMORY` path below.
        """
        return self.addr

    def read(self, projection: list[str] | None = None) -> list[pa.RecordBatch]:
        from batcher.carbonite.transfer.lifecycle import local_session, process_client

        # Fetch through this process's shuffle session when it has one, so the transfer
        # mode is *selected* rather than assumed to be remote: a next-stage worker reading
        # the bucket its own actor published gets it zero-copy out of the local store
        # (`DIRECT_MEMORY`), a same-node peer's bucket comes over mmap'd shared memory, and
        # only a genuinely remote one crosses the network. Going straight to the client
        # made every cross-stage read a loopback gRPC round-trip, serializing bytes that
        # were already in the reader's heap. The session also carries the shuffle token,
        # so a secured (`shuffle_token`) fleet's intermediate is now readable from a
        # worker at all.
        session = local_session(self.addr)
        if session is not None:
            batches = session.fetch(self.addr, str(self.ticket))
        else:
            # No session in this process (the driver collecting an intermediate itself).
            # Pooled, so an adaptive stage re-reading this intermediate reuses the channel
            # it already has to the holding actor rather than dialling it again per read.
            batches = process_client().fetch(self.addr, str(self.ticket))
        if projection is not None:
            batches = [b.select(projection) for b in batches]
        return batches

    def iter_batches(self, projection: list[str] | None = None) -> Iterator[pa.RecordBatch]:
        yield from self.read(projection)

    def row_count(self) -> int | None:
        return self.rows

    def identity(self) -> str:
        return f"flight:{self.addr}:{self.ticket}"


class FlightMaterializedSource:
    """A relation whose batches live on persistent worker Flight servers (one bucket
    per reducer), produced by a Flight stage run with `materialize=False`. The next
    stage scans it in place via shared-nothing `FlightFetchSplit`s and its exact
    `row_count` feeds the optimizer (EXACT provenance); `cleanup()` tears down the
    actors + placement group holding the data once the query no longer needs it.

    When the producing stage *borrowed* a query-lifetime `ShuffleFleet` (the adaptive
    persistent-fleet path), `actors`/`pg` are `None`: the fleet owns those resources
    and is freed once by the adaptive loop, so this source's `cleanup()` no-ops."""

    __slots__ = ("_actors", "_handles", "_pg", "_schema")
    bounded = True

    def __init__(self, handles, schema: pa.Schema, actors, pg) -> None:
        self._handles = handles  # [(addr, ticket, rows)] per non-empty reducer bucket
        self._schema = schema
        self._actors = actors
        self._pg = pg

    def _split(self, handle) -> FlightFetchSplit:
        addr, ticket, rows = handle
        return FlightFetchSplit(addr, ticket, rows, self._schema)

    def _release_buckets(self) -> None:
        """Tell each holding worker to drop the buckets this intermediate advertised.

        Resolved through the live fleet's cached `addrs`, which is a local list parallel to
        `actors` — so mapping an address back to its actor costs nothing and needs no
        remote call. Fire-and-forget: a bucket that fails to free is wasted memory, never a
        wrong answer, and this runs on a teardown path where raising would replace the
        query's real outcome.
        """
        if not self._handles:
            return
        try:
            import ray

            from batcher.dist.fleet import _fleet

            fleet = _fleet.current_fleet()
            if fleet is None or not getattr(fleet, "addrs", None):
                return
            by_addr = dict(zip(fleet.addrs, fleet.actors, strict=False))
            refs = [
                by_addr[addr].release_ticket.remote(str(ticket))
                for addr, ticket, _rows in self._handles
                if addr in by_addr
            ]
            if refs:
                ray.wait(refs, num_returns=len(refs), timeout=_RELEASE_TIMEOUT_S)
        except Exception as exc:  # pragma: no cover - teardown must never raise
            from batcher._internal.logging import note_suppressed

            note_suppressed("dist", "release materialized shuffle buckets", exc)

    def schema(self) -> pa.Schema:
        return self._schema

    def read(self, projection: list[str] | None = None) -> list[pa.RecordBatch]:
        out: list[pa.RecordBatch] = []
        for h in self._handles:
            out.extend(self._split(h).read(projection))
        return out

    def iter_batches(self, projection: list[str] | None = None) -> Iterator[pa.RecordBatch]:
        for h in self._handles:
            yield from self._split(h).iter_batches(projection)

    def row_count(self) -> int | None:
        return sum(rows for _addr, _ticket, rows in self._handles)

    def identity(self) -> str:
        return f"flight-materialized:{self._schema}:{self.row_count()}"

    def splits(self, target_size: int | None = None):  # noqa: ARG002
        return [self._split(h) for h in self._handles]

    def cleanup(self) -> None:
        """Release this intermediate's buckets, then tear down the actors that held them.

        Called once per intermediate by the adaptive loop's `finally`, at the point the
        next stage has finished reading it.

        The bucket release happens **before** the `_actors is None` early return, and that
        ordering is the whole fix. `_actors is None` is the *borrowed-fleet* case — the
        stage ran on a query-lifetime fleet that outlives it — which is exactly the case
        that leaks: without an explicit release, stage `k`'s buckets stay resident on the
        workers through stages `k+1..n`, so a deep adaptive query holds every stage's
        intermediate at once. Killing the actors, the old behaviour, only frees memory in
        the case where the actors were about to die anyway.
        """
        self._release_buckets()
        if self._actors is None:
            return

        import contextlib

        import ray

        from batcher.dist.executors.ray_runtime import release_placement

        for a in self._actors:
            with contextlib.suppress(Exception):
                ray.kill(a)
        release_placement(self._pg)
