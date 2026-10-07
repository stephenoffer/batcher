"""Azure Event Hubs streaming sink — publish each micro-batch through ``azure-eventhub``.

The write side of `eventhubs.EventHubsSource`, on the shared `BrokerStreamSink` contract.
An ``EventHubProducerClient`` bound to one hub sends ``EventDataBatch`` objects: rows are
grouped by their routing (an explicit ``partition`` column, else the ``key`` as the
partition key, else neither) and each group is packed into as many batches as the
service's maximum batch size requires. ``send_batch`` returns once the service has
accepted the batch, so the micro-batch is reported written only after every batch was.

**Ordering** is per partition. A row's ``key`` is the batch's ``partition_key``, which the
service hashes to a partition, so rows sharing a key keep their order; a ``partition``
column (the partition id) pins rows to a partition directly. A producer client is bound to
one Event Hub, so a per-row ``topic`` column is not read.

**Retry identity.** The Python SDK's producer has no idempotent publishing, so a replayed
epoch republishes. ``dedup_ids=`` adds a ``batcher-dedup-id`` application property to
deduplicate on downstream. Headers become application properties, as bytes.

Not yet verified against a live Event Hubs namespace; see tests/PENDING_VERIFICATION.md.
"""

from __future__ import annotations

from typing import Any

from batcher._internal.errors import IOError, PlanError
from batcher._internal.optional import require
from batcher.io.formats.streaming.broker_sinks.contract import (
    BrokerRecords,
    BrokerStreamSink,
    SinkCapabilities,
)
from batcher.io.formats.streaming.sinks import STREAM_SINKS

__all__ = ["EventHubsStreamSink"]


def _import_eventhub() -> Any:
    """Import ``azure.eventhub`` or raise a guiding ``BackendError``."""
    return require(
        "azure.eventhub",
        feature="Event Hubs support",
        provides="azure-eventhub",
        extra="eventhubs",
    )


@STREAM_SINKS.register("eventhubs")
class EventHubsStreamSink(BrokerStreamSink):
    """Publish each micro-batch's rows to an Event Hub, one event per row.

    Args:
        topic: The Event Hub name.
        connection_str: The namespace connection string, or a secret reference to it.
        flush_timeout: Seconds each ``send_batch`` may take.
        dedup_ids: A stable writer name; each event then carries a ``batcher-dedup-id``
            application property.
        options: Codec options (``value_format=`` ...).
    """

    capabilities = SinkCapabilities(
        broker="eventhubs",
        display="Event Hubs",
        columns=("value", "key", "partition", "headers"),
        ordering="per partition; the key is the partition_key, a partition column pins "
        "the partition id",
        batching="EventDataBatch per routing group, split at the service's maximum batch size",
        retry_identity="none on publish (no idempotent producer in the Python SDK); "
        "dedup_ids= application property",
        per_row_destination=False,
    )

    def __init__(
        self,
        *,
        topic: str | None = None,
        connection_str: str = "",
        flush_timeout: float = 60.0,
        dedup_ids: str | None = None,
        **options: Any,
    ) -> None:
        codecs = self.split_codec_options(options)
        if options:
            raise PlanError(f"eventhubs sink: unknown option(s) {sorted(options)}")
        super().__init__(
            topic=topic, flush_timeout=flush_timeout, dedup_ids=dedup_ids, codec_options=codecs
        )
        self._connection_str = connection_str
        self._client: Any = None
        self._event_cls: Any = None

    def _connect(self) -> None:
        from batcher.io.credentials import resolve_secret

        if not self._connection_str:
            raise PlanError("the Event Hubs sink needs connection_str=... (or a secret reference)")
        eventhub = _import_eventhub()
        self._event_cls = eventhub.EventData
        self._client = eventhub.EventHubProducerClient.from_connection_string(
            conn_str=resolve_secret(self._connection_str, what="Event Hubs connection_str"),
            eventhub_name=self._topic,
        )

    def _publish(self, records: BrokerRecords, batch_id: int) -> None:
        groups: dict[tuple[str | None, str | None], list[int]] = {}
        for i in range(len(records)):
            partition = records.partitions[i] if records.partitions is not None else None
            if partition is not None:
                route = (str(partition), None)
            else:
                route = (None, records.text_key(i, broker="Event Hubs"))
            groups.setdefault(route, []).append(i)
        for (partition_id, partition_key), rows in groups.items():
            self._send_group(records, rows, partition_id, partition_key, batch_id)

    def _send_group(
        self,
        records: BrokerRecords,
        rows: list[int],
        partition_id: str | None,
        partition_key: str | None,
        batch_id: int,
    ) -> None:
        """Pack one routing group's events into batches and send each one."""
        batch = self._new_batch(partition_id, partition_key)
        for i in rows:
            event = self._event_cls(records.values[i])
            headers = records.header_pairs(i)
            if headers:
                event.properties = dict(headers)
            try:
                batch.add(event)
                continue
            except ValueError:
                # The batch is at the service's size limit: send it and start another.
                if len(batch) == 0:
                    raise IOError(
                        f"eventhubs sink: row {i} of micro-batch {batch_id} is larger than "
                        "the hub's maximum event size"
                    ) from None
            self._send(batch, batch_id)
            batch = self._new_batch(partition_id, partition_key)
            batch.add(event)
        if len(batch):
            self._send(batch, batch_id)

    def _new_batch(self, partition_id: str | None, partition_key: str | None) -> Any:
        kwargs: dict[str, Any] = {}
        if partition_id is not None:
            kwargs["partition_id"] = partition_id
        if partition_key is not None:
            kwargs["partition_key"] = partition_key
        return self._client.create_batch(**kwargs)

    def _send(self, batch: Any, batch_id: int) -> None:
        try:
            self._client.send_batch(batch, timeout=self._flush_timeout)
        except Exception as exc:
            raise IOError(
                f"eventhubs sink: a batch of micro-batch {batch_id} was not accepted: {exc}. "
                "The epoch is not durable."
            ) from exc

    def _disconnect(self) -> None:
        if self._client is None:
            return
        client, self._client = self._client, None
        client.close()
