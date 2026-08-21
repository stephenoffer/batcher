"""`BrokerSplit` — one partition of a broker, read one epoch at a time on a worker.

Also owns the per-process cache of live partition consumers that keeps a distributed
streaming query from rebuilding a client on every trigger.
"""

from __future__ import annotations

import atexit
from collections.abc import Iterator
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

import pyarrow as pa

from batcher.io.formats.streaming.broker.schema import broker_schema
from batcher.io.formats.streaming.broker.source import BrokerSource, _options_fingerprint

__all__ = ["BrokerSplit"]


_EPOCH_READERS: dict[str, tuple[BrokerSource, Any]] = {}

#: How many partition consumers one worker process keeps warm. A worker owns a shard of the
#: topic's partitions, so this only needs to cover a shard; the cap is a leak-stop, not a
#: tuning knob. The oldest entry is closed when a new partition needs a slot.
_EPOCH_READER_LIMIT = 32


def _close_epoch_readers() -> None:
    """Close every cached partition consumer. Registered at exit; safe to call twice."""
    while _EPOCH_READERS:
        _, (source, _) = _EPOCH_READERS.popitem()
        with suppress(Exception):
            source.close()


atexit.register(_close_epoch_readers)


@dataclass(frozen=True, slots=True)
class BrokerSplit:
    """One partition/shard of a broker, reconstructed on the worker.

    Carries only picklable offset *locators* (format name, topic, partition id,
    poll size, client options) — never live client handles or data. ``read``
    rebuilds the concrete broker source from the format registry, scoped to this
    single partition.
    """

    format_name: str
    topic: str
    partition: int
    poll_size: int
    poll_bytes: int = BrokerSource.DEFAULT_POLL_BYTES
    #: Whether this partition's reader produces the `headers` column. It has to travel with
    #: the split: a worker that rebuilt the source without it would return a batch one
    #: column narrower than its siblings, and the epoch's concat would fail on the schema.
    include_headers: bool = False
    #: The payload wire-format options (`value_format`, `value_schema`, `schema_registry`,
    #: …) exactly as the source received them. They travel as *config*, not as built codec
    #: objects: a live `SchemaRegistry` holds a lock and a socket, neither of which pickles,
    #: and a worker that rebuilt the codec from config gets its own cache anyway. A split
    #: that dropped them would return an undecoded `binary` column while its siblings
    #: returned a struct, and the epoch's concat would fail on the schema.
    codecs: dict[str, Any] = field(default_factory=dict, repr=False)
    #: `repr=False` because these are the *client* options — they carry `sasl.password`,
    #: `sasl_plain_password`, and Event Hubs' `connection_str` (a SAS key). A split is
    #: pickled to every worker and appears verbatim in any traceback that mentions it, so
    #: the generated dataclass `repr` was printing broker credentials into logs.
    options: dict[str, Any] = field(default_factory=dict, repr=False)

    def _reader(self) -> BrokerSource:
        from batcher.io.formats.base import SOURCES

        cls = SOURCES.get(self.format_name)
        return cls(  # type: ignore[no-any-return]
            self.topic,
            poll_size=self.poll_size,
            poll_bytes=self.poll_bytes,
            include_headers=self.include_headers,
            partitions=[self.partition],
            **self.codecs,
            **self.options,
        )

    def schema(self) -> pa.Schema:
        # Asked of the rebuilt reader rather than of `broker_schema` directly: a split with
        # a payload codec advertises the *decoded* column types, and answering the raw
        # schema here would disagree with the batches this very split produces.
        if any(self.codecs.get(f"{side}_format") for side in ("value", "key")):
            return self._reader().schema()
        return broker_schema(self.include_headers)

    def read(self, projection: list[str] | None = None) -> list[pa.RecordBatch]:
        return self._reader().read(projection)

    def iter_batches(self, projection: list[str] | None = None) -> Iterator[pa.RecordBatch]:
        yield from self._reader().iter_batches(projection)

    def read_epoch(
        self, start_offset: Any | None = None, projection: list[str] | None = None
    ) -> tuple[list[pa.RecordBatch], Any | None]:
        """Read **one** micro-batch of this partition, resuming after `start_offset`.

        The distributed streaming path needs a partition read that *ends*: `read` and
        `iter_batches` poll an unbounded broker forever, which is right for a consumer and
        useless for an epoch. This polls once and reports the offset it stopped at, so the
        driver can write-ahead the position and hand it back for the next epoch — the same
        resume-strictly-after contract the single-node checkpoint uses, evaluated on the
        worker that owns the partition.

        Returns the batch (empty when the poll had nothing) and the new resume position.
        """
        src = self._epoch_reader(start_offset)
        messages = src._poll()
        if not messages:  # None (end of stream) or an empty poll
            return [], start_offset
        src._track_positions(messages)
        batch = src._make_batch(messages, projection)
        # Fall back to `start_offset` rather than `None` when the poll carried nothing for
        # this partition. `_read_epoch` drops a `None` position from the epoch's offset map
        # entirely, so returning one does not mean "unchanged" — it means the driver loses
        # the partition's checkpoint and the next restart replays from `auto.offset.reset`.
        position = src.snapshot_position()["offsets"].get(str(self.partition), start_offset)
        # Remember where this consumer now stands so the next epoch can reuse it in place.
        _EPOCH_READERS[self.identity()] = (src, position)
        # Deliberately does *not* `_commit_delivered()`. This returns *before* the driver
        # has published the epoch, so committing here would be the same at-most-once bug
        # the single-node path was fixed for: a crash between this return and the publish
        # would leave the broker believing the messages were handled. On this path the
        # driver's write-ahead log is the source of truth for position, not the broker's
        # own offsets.
        return [batch], position

    def _epoch_reader(self, start_offset: Any | None) -> BrokerSource:
        """A consumer for this partition, kept alive across micro-batches where it is safe.

        Building a fresh client per epoch is correct and ruinously slow. A streaming query
        runs one epoch per trigger, so a 200ms trigger meant tearing down and rebuilding a
        Kafka consumer five times a second per partition: a TCP connect, a metadata fetch, a
        group join, and a seek, all serialized *ahead* of the poll that actually reads data.
        The connection setup dominated the epoch, and the broker saw a consumer churn that
        looks like a crash-loop. Worse, a client that lives only for one poll can never
        prefetch, so each epoch started with an empty local queue and paid a full broker
        round-trip for its first record.

        Reuse is conditional on *contiguous* resumption: the cached consumer is handed back
        only when the driver asks to resume from exactly the position that consumer last
        delivered. Any other request (a restart from an older checkpoint, a different query
        landing on this worker, a rebalance) closes it and builds a fresh one positioned by
        `seek`. That keeps the fast path fast while making a stale cursor impossible to read
        from — the cache can cost a reconnection, never a wrong or skipped row.

        Args:
            start_offset: The position the driver wants this epoch to resume strictly after.

        Returns:
            A broker source scoped to this split's partition, positioned at ``start_offset``.
        """
        key = self.identity()
        cached = _EPOCH_READERS.pop(key, None)
        if cached is not None:
            source, last_position = cached
            if last_position == start_offset:
                # The driver came back for the position this consumer stopped at, so the
                # epoch it delivered is behind us and the client may release it. This is the
                # only place on the distributed path where that is safe, and it is not merely
                # bookkeeping: a broker that holds delivered-but-unreleased messages (Pulsar
                # keeps every un-acked handle so `_commit_delivered` can ack them) grew that
                # buffer without bound once the consumer started outliving the epoch.
                # Recovery here is driven by the driver's write-ahead log via `seek`, never by
                # the broker's own cursor, so releasing early cannot skip a row.
                source._commit_delivered()
                _EPOCH_READERS[key] = cached  # re-insert: most-recently-used goes last
                return source
            with suppress(Exception):
                source.close()
        while len(_EPOCH_READERS) >= _EPOCH_READER_LIMIT:
            evicted, _ = _EPOCH_READERS.pop(next(iter(_EPOCH_READERS)))
            with suppress(Exception):
                evicted.close()
        source = self._reader()
        if start_offset is not None:
            source.seek({"offsets": {str(self.partition): start_offset}})
        _EPOCH_READERS[key] = (source, start_offset)
        return source

    def row_count(self) -> int | None:
        return None

    def identity(self) -> str:
        """Partition-scoped stats key, carrying the same connection fingerprint as the source.

        A split's identity had the same cluster-blindness as the source's: partition 3 of
        ``events`` on production and on staging were one key. It is fingerprinted from the
        redacted options so the split and the source it came from agree, and so the persisted
        key never carries a credential.
        """
        return (
            f"{self.format_name}:{self.topic}:p{self.partition}"
            f":{_options_fingerprint(self.options)}"
        )
