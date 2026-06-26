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

    def read(self, projection: list[str] | None = None) -> list[pa.RecordBatch]:
        from batcher.carbonite.transfer.server import fetch

        batches = fetch(self.addr, self.ticket)
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
        """Kill the actors holding the buckets and release their placement group.

        No-ops when the producing stage borrowed a query-lifetime fleet (`_actors`
        is `None`) — that fleet is owned and freed by the adaptive loop instead.
        """
        if self._actors is None:
            return

        import contextlib

        import ray

        from batcher.dist.executors.ray_runtime import release_placement

        for a in self._actors:
            with contextlib.suppress(Exception):
                ray.kill(a)
        release_placement(self._pg)
