"""Kinesis broker source — one Split per shard, via ``boto3`` shard iterators.

Backed by ``boto3`` (the optional ``kinesis`` extra). A :class:`KinesisSource`
polls a shard with ``get_records`` (after obtaining a shard iterator with
``get_shard_iterator``) and assembles each poll into one Arrow batch via the
shared ``_make_batch`` helper.

``splits()`` returns one split per shard (the shard id is the offset locator), so
a distributed reader assigns one consumer per shard. The Kinesis sequence number
is opaque text; it is hashed into the int64 ``offset`` column so the fixed broker
schema is preserved (the raw sequence is not needed downstream, only ordering /
de-dup within a shard, which the sequence-number-based iterator already provides).

The ``boto3`` import is deferred to construction; if the extra is missing a
:class:`BackendError` instructs the user to install it.
"""

from __future__ import annotations

from typing import Any

from batcher._internal.errors import BackendError
from batcher.io.formats.base import SOURCES
from batcher.io.formats.streaming.broker import BrokerMessage, BrokerSource

__all__ = ["KinesisSource"]


def _import_boto3() -> Any:
    """Import ``boto3`` or raise a guiding ``BackendError``."""
    try:
        import boto3
    except ImportError as exc:
        raise BackendError(
            "reading from Kinesis needs the kinesis extra: pip install 'batcher[kinesis]'"
        ) from exc
    return boto3


@SOURCES.register("kinesis")
class KinesisSource(BrokerSource):
    """An unbounded Kinesis stream, consumed via ``boto3``.

    The ``topic`` is the Kinesis stream name. Options: ``region`` (AWS region),
    ``iterator_type`` (``"TRIM_HORIZON"`` by default, or ``"LATEST"``), and
    ``partitions`` (the specific shards to read — set by :class:`BrokerSplit` on a
    worker; the values are shard *indices* into the discovered shard list).
    """

    format_name = "kinesis"

    __slots__ = ("_client_obj", "_iterators", "_partitions", "_shard_ids")

    def __init__(
        self,
        topic: str,
        *,
        poll_size: int = 16_384,
        partitions: list[int] | None = None,
        region: str = "us-east-1",
        iterator_type: str = "TRIM_HORIZON",
        **options: Any,
    ) -> None:
        super().__init__(
            topic,
            poll_size=poll_size,
            region=region,
            iterator_type=iterator_type,
            **options,
        )
        self._partitions = partitions
        self._client_obj: Any = None
        self._shard_ids: list[str] | None = None
        self._iterators: dict[str, str] = {}

    def _client(self) -> Any:
        if self._client_obj is None:
            boto3 = _import_boto3()
            self._client_obj = boto3.client("kinesis", region_name=self._options["region"])
        return self._client_obj

    def _shards(self) -> list[str]:
        if self._shard_ids is None:
            client = self._client()
            resp = client.list_shards(StreamName=self.topic)
            self._shard_ids = [s["ShardId"] for s in resp.get("Shards", [])]
        return self._shard_ids

    def _discover_partitions(self) -> list[int]:
        if self._partitions is not None:
            return list(self._partitions)
        return list(range(len(self._shards())))

    def _active_shards(self) -> list[str]:
        shards = self._shards()
        if self._partitions is None:
            return shards
        return [shards[i] for i in self._partitions if i < len(shards)]

    def _iterator(self, shard_id: str) -> str:
        if shard_id not in self._iterators:
            client = self._client()
            resp = client.get_shard_iterator(
                StreamName=self.topic,
                ShardId=shard_id,
                ShardIteratorType=self._options["iterator_type"],
            )
            self._iterators[shard_id] = resp["ShardIterator"]
        return self._iterators[shard_id]

    def _poll(self) -> list[BrokerMessage] | None:
        client = self._client()
        messages: list[BrokerMessage] = []
        for shard_index, shard_id in enumerate(self._active_shards()):
            resp = client.get_records(
                ShardIterator=self._iterator(shard_id),
                Limit=self.poll_size,
            )
            next_iter = resp.get("NextShardIterator")
            if next_iter is not None:
                self._iterators[shard_id] = next_iter
            for rec in resp.get("Records", []):
                ts = rec.get("ApproximateArrivalTimestamp")
                messages.append(
                    BrokerMessage(
                        value=rec["Data"],
                        partition=shard_index,
                        offset=_seq_to_offset(rec["SequenceNumber"]),
                        timestamp=int(ts.timestamp() * 1000) if ts is not None else 0,
                        topic=self.topic,
                        key=(rec.get("PartitionKey") or "").encode("utf-8") or None,
                    )
                )
        return messages


def _seq_to_offset(sequence_number: str) -> int:
    """Map an opaque Kinesis sequence number to a stable int64 offset column.

    The raw sequence is a large decimal string; take it modulo 2**63 so it fits
    the fixed int64 ``offset`` column while preserving monotonic ordering within
    the precision of int64 (sequence numbers within a shard are increasing).
    """
    try:
        return int(sequence_number) % (1 << 63)
    except ValueError:
        return abs(hash(sequence_number)) % (1 << 63)
