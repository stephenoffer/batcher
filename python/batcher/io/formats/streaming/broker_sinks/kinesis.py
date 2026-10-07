"""Amazon Kinesis Data Streams sink — publish each micro-batch through ``boto3``.

The write side of `kinesis.KinesisSource`, on the shared `BrokerStreamSink` contract.

**Batching.** Records go out through ``PutRecords`` in requests of at most 500 records and
5 MiB, the API's documented limits. ``PutRecords`` is not all-or-nothing: the response
names each record that failed (``ErrorCode``, typically
``ProvisionedThroughputExceededException``), and only those are resent, with exponential
backoff, up to ``max_attempts`` attempts. A whole request rejected for throughput is retried
the same way. Anything still failing after that fails the micro-batch.

**Ordering** is per partition key within a shard, and only when ``ordered=True``. AWS
documents that ``PutRecords`` does not guarantee order, and resending the failed subset of
a request can put a record after one written later. ``ordered=True`` switches to one
``PutRecord`` per record, chaining ``SequenceNumberForOrdering`` per partition key, which is
the API's ordering mechanism, at one request per record.

**Partition key.** A row's ``key`` (string or UTF-8 binary) is the ``PartitionKey``. A row
with no key gets ``<batch id>-<row>``, which spreads records across shards and implies no
order.

**Retry identity.** Kinesis has no producer-side deduplication and no per-record metadata
to carry an id, so a replayed epoch republishes and ``dedup_ids=`` is refused. Put a record
id in the payload and deduplicate on it downstream.

Not yet verified against a live Kinesis; see tests/PENDING_VERIFICATION.md.
"""

from __future__ import annotations

import time
from typing import Any

from batcher._internal.errors import IOError, PlanError
from batcher.io.formats.streaming.broker_sinks.contract import (
    BrokerRecords,
    BrokerStreamSink,
    SinkCapabilities,
)
from batcher.io.formats.streaming.kinesis import _CLIENT_OPTIONS, _import_boto3, _is_throttle
from batcher.io.formats.streaming.sinks import STREAM_SINKS

__all__ = ["KinesisStreamSink"]

#: ``PutRecords`` limits: records per request, and bytes per request (data plus keys).
_MAX_RECORDS = 500
_MAX_REQUEST_BYTES = 5 * 1024 * 1024

#: First backoff between attempts, doubled each time.
_BACKOFF_SECONDS = 0.1


def _chunks(entries: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Split request entries into ``PutRecords`` calls within both limits, in order."""
    out: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    size = 0
    for entry in entries:
        entry_size = len(entry["Data"]) + len(entry["PartitionKey"].encode())
        if current and (len(current) >= _MAX_RECORDS or size + entry_size > _MAX_REQUEST_BYTES):
            out.append(current)
            current, size = [], 0
        current.append(entry)
        size += entry_size
    if current:
        out.append(current)
    return out


@STREAM_SINKS.register("kinesis")
class KinesisStreamSink(BrokerStreamSink):
    """Publish each micro-batch's rows to a Kinesis data stream, one record per row.

    Args:
        topic: The stream name, for rows whose ``topic`` column is null or absent.
        region: The AWS region.
        ordered: Keep per-partition-key order with one ``PutRecord`` per record.
        max_attempts: Attempts per record before the micro-batch fails.
        flush_timeout: Seconds a micro-batch may spend retrying before it fails.
        dedup_ids: Refused: Kinesis records carry no metadata to hold an id.
        options: Codec options (``value_format=`` ...) plus ``endpoint_url`` and AWS
            credentials (``aws_access_key_id`` ..., secret references accepted), as for the
            source.
    """

    capabilities = SinkCapabilities(
        broker="kinesis",
        display="Kinesis",
        columns=("value", "key", "topic"),
        ordering="per partition key within a shard, only with ordered=True (PutRecord with "
        "SequenceNumberForOrdering); PutRecords does not guarantee order",
        batching="PutRecords requests of at most 500 records / 5 MiB; failed records resent",
        retry_identity="none: Kinesis has no producer dedup or record metadata; put an id "
        "in the payload",
        max_attempts_option="max_attempts",
    )

    def __init__(
        self,
        *,
        topic: str | None = None,
        region: str = "us-east-1",
        ordered: bool = False,
        max_attempts: int = 5,
        flush_timeout: float = 30.0,
        dedup_ids: str | None = None,
        **options: Any,
    ) -> None:
        if max_attempts < 1:
            raise PlanError(f"kinesis sink max_attempts must be >= 1, got {max_attempts}")
        codecs = self.split_codec_options(options)
        super().__init__(
            topic=topic, flush_timeout=flush_timeout, dedup_ids=dedup_ids, codec_options=codecs
        )
        unknown = sorted(set(options) - set(_CLIENT_OPTIONS))
        if unknown:
            raise PlanError(
                f"kinesis sink: unknown option(s) {unknown}; accepted: {list(_CLIENT_OPTIONS)}"
            )
        self._region = region
        self._ordered = ordered
        self._max_attempts = max_attempts
        self._client_options = options
        self._client: Any = None

    def _connect(self) -> None:
        from batcher.io.credentials import resolve_client_secrets
        from batcher.io.formats.streaming.broker.schema import _BROKER_SECRET_HINTS

        boto3 = _import_boto3()
        extra = {k: v for k, v in self._client_options.items() if v}
        self._client = boto3.client(
            "kinesis",
            region_name=self._region,
            **resolve_client_secrets(extra, what="kinesis", hints=_BROKER_SECRET_HINTS),
        )

    def _publish(self, records: BrokerRecords, batch_id: int) -> None:
        by_stream: dict[str, list[dict[str, Any]]] = {}
        for i, value in enumerate(records.values):
            key = records.text_key(i, broker="Kinesis") or f"{batch_id}-{i}"
            entry = {"Data": value, "PartitionKey": key}
            by_stream.setdefault(records.destinations[i], []).append(entry)
        deadline = time.monotonic() + self._flush_timeout
        for stream, entries in by_stream.items():
            if self._ordered:
                self._put_ordered(stream, entries, batch_id, deadline)
            else:
                for chunk in _chunks(entries):
                    self._put_records(stream, chunk, batch_id, deadline)

    def _put_records(
        self, stream: str, entries: list[dict[str, Any]], batch_id: int, deadline: float
    ) -> None:
        """One ``PutRecords`` call, resending only the records the response says failed."""
        pending = entries
        last_error = ""
        for attempt in range(self._max_attempts):
            if attempt:
                self._backoff(attempt, deadline, batch_id)
            try:
                response = self._client.put_records(StreamName=stream, Records=pending)
            except Exception as exc:
                if not _is_throttle(exc):
                    raise IOError(f"kinesis sink: PutRecords to {stream!r} failed: {exc}") from exc
                last_error = type(exc).__name__
                continue
            if not response.get("FailedRecordCount"):
                return
            results = response.get("Records", [])
            failed = [e for e, r in zip(pending, results, strict=True) if r.get("ErrorCode")]
            last_error = next(r["ErrorCode"] for r in results if r.get("ErrorCode"))
            pending = failed
        raise IOError(
            f"kinesis sink: {len(pending)} record(s) of micro-batch {batch_id} to {stream!r} "
            f"still failed after {self._max_attempts} attempt(s); last error: {last_error}. "
            "The epoch is not durable; raise max_attempts or the stream's shard count."
        )

    def _put_ordered(
        self, stream: str, entries: list[dict[str, Any]], batch_id: int, deadline: float
    ) -> None:
        """One ``PutRecord`` per record, chaining sequence numbers per partition key."""
        last: dict[str, str] = {}
        for entry in entries:
            key = entry["PartitionKey"]
            request = dict(entry, StreamName=stream)
            if key in last:
                request["SequenceNumberForOrdering"] = last[key]
            for attempt in range(self._max_attempts):
                if attempt:
                    self._backoff(attempt, deadline, batch_id)
                try:
                    last[key] = self._client.put_record(**request)["SequenceNumber"]
                    break
                except Exception as exc:
                    if not _is_throttle(exc) or attempt + 1 == self._max_attempts:
                        raise IOError(
                            f"kinesis sink: PutRecord to {stream!r} failed for micro-batch "
                            f"{batch_id}: {exc}"
                        ) from exc

    def _backoff(self, attempt: int, deadline: float, batch_id: int) -> None:
        delay = _BACKOFF_SECONDS * (2 ** (attempt - 1))
        if time.monotonic() + delay > deadline:
            raise IOError(
                f"kinesis sink: micro-batch {batch_id} ran past flush_timeout="
                f"{self._flush_timeout}s while retrying throttled records"
            )
        time.sleep(delay)

    def _disconnect(self) -> None:
        self._client = None
