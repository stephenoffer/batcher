"""The one capability contract every message-broker streaming sink implements.

Kafka, Pulsar, Kinesis, Pub/Sub and Event Hubs differ in their clients and agree on almost
everything else, so the part they agree on lives here once:

- **The column contract.** ``value`` is required; ``key``, ``topic`` (a per-row
  destination) and ``headers`` (``array<struct<key, value>>`` or a map) are read where the
  broker has somewhere to put them, and ``partition`` where it can be addressed. Other
  columns are ignored, so the broker schema a source delivers can be written straight back.
  A column the broker cannot carry is refused at the first micro-batch with its name and
  type, never half-way through a publish.
- **Payload codecs.** ``value_format=`` / ``key_format=`` serialize a typed column through
  the same codec vocabulary the sources decode with.
- **Acknowledgement before report.** `write_batch` returns only once every record of the
  micro-batch is acknowledged by the broker, and raises `IOError` otherwise. A failed epoch
  is replayed from the checkpoint, so delivery is at-least-once on every broker here.
- **Retry identity.** ``dedup_ids=<writer name>`` attaches a ``batcher-dedup-id`` header
  ``<writer name>:<batch id>:<row>`` to every record on brokers with a per-message metadata
  slot. A replayed epoch carries the same ids for the same rows, as long as the pipeline
  emits the epoch's rows in the same order, so a consumer can drop a replayed duplicate.
  Broker-native deduplication, where the service has it, is described per sink by its
  `SinkCapabilities`.

Each sink states what it does with these through a class-level `SinkCapabilities`, which the
docs table is checked against, so "what ordering does the Pulsar sink give me" has one
answer in code rather than one per page.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import pyarrow as pa

from batcher._internal.errors import PlanError
from batcher.io.formats.streaming.codecs.base import build_payload_codecs

__all__ = [
    "DEDUP_ID_HEADER",
    "BrokerRecords",
    "BrokerStreamSink",
    "SinkCapabilities",
]

#: The header / attribute / property name a ``dedup_ids=`` record id travels under.
DEDUP_ID_HEADER = "batcher-dedup-id"

#: Arrow types a payload column (`key` / `value`) may have. Brokers carry bytes; a string
#: column is UTF-8 encoded on the way out, which is what every client does too.
_PAYLOAD_TYPES = ("binary", "large_binary", "string", "large_string")

#: The codec options every broker sink accepts, forwarded to `build_payload_codecs`.
_CODEC_OPTIONS = (
    "value_format",
    "value_schema",
    "value_subject",
    "key_format",
    "key_schema",
    "key_subject",
    "schema_registry",
    "schema_registry_auth",
    "value_codec_options",
    "key_codec_options",
)


@dataclass(frozen=True)
class SinkCapabilities:
    """What one broker sink guarantees, stated once so code, tests and docs agree.

    Attributes:
        broker: The sink's registry name (``"kafka"``, ``"pulsar"``, ...).
        display: The broker's name as an error message should spell it.
        columns: The input columns the sink reads; every other column is ignored.
        ordering: The scope within which records keep the order they were written in.
        batching: How records are grouped into requests to the broker.
        retry_identity: What the broker offers to recognize a retried record.
        null_values: Whether a null ``value`` is publishable (a Kafka tombstone).
        per_row_destination: Whether a ``topic`` column may route rows elsewhere.
        max_attempts_option: The sink option bounding broker-side retries, if any.
    """

    broker: str
    display: str
    columns: tuple[str, ...]
    ordering: str
    batching: str
    retry_identity: str
    null_values: bool = False
    per_row_destination: bool = True
    max_attempts_option: str | None = None


@dataclass
class BrokerRecords:
    """One micro-batch's records as the per-row values a broker client takes.

    Every list is as long as the batch; an optional list is `None` when the column is
    absent. The conversion is one vectorized `to_pylist` per column: the clients take one
    record at a time regardless, and the encode is the user's chosen wire format.
    """

    values: list[bytes | None]
    destinations: list[str]
    keys: list[bytes | None] | None = None
    partitions: list[Any] | None = None
    headers: list[list[tuple[str, bytes | None]]] | None = None

    def __len__(self) -> int:
        return len(self.values)

    def header_pairs(self, i: int) -> list[tuple[str, bytes | None]]:
        """Row `i`'s headers, or an empty list when it carries none."""
        if self.headers is None:
            return []
        return self.headers[i]

    def text_properties(self, i: int, *, broker: str) -> dict[str, str]:
        """Row `i`'s headers as the ``str -> str`` map Pulsar and Pub/Sub carry.

        Raises:
            PlanError: If a header value is not UTF-8, which a text-only property map
                cannot carry without corrupting it.
        """
        out: dict[str, str] = {}
        for key, payload in self.header_pairs(i):
            if payload is None:
                out[key] = ""
                continue
            try:
                out[key] = payload.decode("utf-8")
            except UnicodeDecodeError:
                raise PlanError(
                    f"the {broker} sink carries headers as text, and header {key!r} is not "
                    "UTF-8; hex-encode it first, e.g. with col(...).str.hex()"
                ) from None
        return out

    def text_key(self, i: int, *, broker: str) -> str | None:
        """Row `i`'s key decoded as UTF-8 text, for brokers whose key is a string.

        Raises:
            PlanError: If the key is not UTF-8, which a string key cannot carry.
        """
        if self.keys is None or self.keys[i] is None:
            return None
        try:
            return self.keys[i].decode("utf-8")
        except UnicodeDecodeError:
            raise PlanError(
                f"the {broker} sink's key is text, and row {i}'s key is not UTF-8; "
                "project a string key, e.g. col('key').str.hex()"
            ) from None


def payload_column(table: pa.Table, name: str) -> list[bytes | None]:
    """A payload column as Python bytes, UTF-8 encoding a string column on the way."""
    column = table.column(name)
    if pa.types.is_string(column.type) or pa.types.is_large_string(column.type):
        return [None if v is None else v.encode("utf-8") for v in column.to_pylist()]
    return column.to_pylist()


def record_headers(value: Any) -> list[tuple[str, bytes | None]]:
    """One row's `headers` column as ``[(key, bytes)]`` pairs.

    Two shapes are accepted because both are natural to produce with expressions: a map
    column arrives as a list of ``(key, value)`` pairs, and a ``list<struct<key, value>>``
    column arrives as a list of dicts.
    """
    headers: list[tuple[str, bytes | None]] = []
    for item in value or ():
        if isinstance(item, dict):
            key, payload = item.get("key"), item.get("value")
        else:
            key, payload = item
        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        headers.append((str(key), payload))
    return headers


def _optional(table: pa.Table, name: str) -> list[Any] | None:
    """Column `name` as a Python list, or `None` when the table has no such column."""
    if table.schema.get_field_index(name) < 0:
        return None
    return table.column(name).to_pylist()


class BrokerStreamSink:
    """The shared `StreamSink` template: encode, validate, extract, publish, report.

    A subclass declares its `capabilities` and implements three hooks: `_connect` builds
    the client, `_publish` sends one micro-batch's `BrokerRecords` and returns only once
    every record is acknowledged (raising `IOError` otherwise), and `_disconnect` releases
    the client. Everything a user can observe about the column contract is here, so the
    five brokers cannot drift apart on it.

    Args:
        topic: Destination for rows whose ``topic`` column is null or absent.
        flush_timeout: Seconds a micro-batch may wait for acknowledgements.
        dedup_ids: A stable writer name; when set, each record carries a
            ``batcher-dedup-id`` of ``<name>:<batch id>:<row>`` where the broker has a
            metadata slot for it.
        codec_options: The ``value_format=`` / ``key_format=`` family, as for the sources.
    """

    capabilities: ClassVar[SinkCapabilities]

    def __init__(
        self,
        *,
        topic: str | None,
        flush_timeout: float,
        dedup_ids: str | None = None,
        codec_options: dict[str, Any] | None = None,
    ) -> None:
        broker = self.capabilities.broker
        if flush_timeout <= 0:
            raise PlanError(f"{broker} sink flush_timeout must be > 0, got {flush_timeout}")
        if dedup_ids is not None and "headers" not in self.capabilities.columns:
            raise PlanError(
                f"the {broker} sink has no per-record metadata to carry dedup_ids; embed a "
                "record id in the payload instead"
            )
        self._topic = topic
        self._flush_timeout = flush_timeout
        self._dedup_ids = dedup_ids
        options = {k: v for k, v in (codec_options or {}).items() if k in _CODEC_OPTIONS}
        self._value_codec, self._key_codec = build_payload_codecs(topic or "", options)

    @staticmethod
    def split_codec_options(options: dict[str, Any]) -> dict[str, Any]:
        """Pop the payload-codec options out of a sink's ``**options``, in place.

        Returns:
            The codec options, for the `codec_options` argument.
        """
        return {k: options.pop(k) for k in _CODEC_OPTIONS if k in options}

    # --- the StreamSink protocol -------------------------------------------
    def open(self) -> None:
        """Build the client. Deferred to here so a plan can be built without the extra."""
        self._connect()

    def write_batch(self, batch_id: int, table: pa.Table) -> str | None:
        """Publish every row, wait for the acknowledgements, and report what was written.

        Args:
            batch_id: The micro-batch's id, used in the receipt and in dedup ids.
            table: The micro-batch's output.

        Returns:
            A ``<broker>:<topic>:<batch_id>:<rows>`` receipt for the commit log.

        Raises:
            PlanError: If the table cannot be carried by this broker.
            IOError: If any record was rejected or not acknowledged in time.
        """
        table = self._encode(table)
        self._validate(table)
        broker = self.capabilities.broker
        if table.num_rows == 0:
            return f"{broker}:{self._topic}:{batch_id}:0"
        records = self._records(table, batch_id)
        self._publish(records, batch_id)
        return f"{broker}:{self._topic}:{batch_id}:{table.num_rows}"

    def close(self) -> None:
        """Release the client. Idempotent."""
        self._disconnect()

    # --- hooks ------------------------------------------------------------
    def _connect(self) -> None:
        raise NotImplementedError

    def _publish(self, records: BrokerRecords, batch_id: int) -> None:
        raise NotImplementedError

    def _disconnect(self) -> None:
        raise NotImplementedError

    def _validate_extra(self, table: pa.Table) -> None:
        """Broker-specific refusals beyond the shared contract; none by default."""

    # --- the shared contract ----------------------------------------------
    def _encode(self, table: pa.Table) -> pa.Table:
        """Serialize the payload columns through this sink's codecs, if it has any.

        Before `_validate`, deliberately: with a codec the incoming `value` is a struct,
        which the payload-type check exists to reject when there is *no* codec to turn it
        into bytes. Validating first would refuse exactly the shape a codec accepts.
        """
        for name, codec in (("value", self._value_codec), ("key", self._key_codec)):
            if codec is None:
                continue
            index = table.schema.get_field_index(name)
            if index < 0:
                continue
            encoded = codec.encode(table.column(name).combine_chunks())
            table = table.set_column(index, pa.field(name, pa.binary()), encoded)
        return table

    def _validate(self, table: pa.Table) -> None:
        """Refuse a table this broker cannot carry, before a single record is sent."""
        caps = self.capabilities
        label = caps.display
        schema = table.schema
        if schema.get_field_index("value") < 0:
            raise PlanError(
                f"the {label} sink needs a 'value' column; the write schema is "
                f"{schema.names}. Project one, e.g. "
                ".select(value=col('payload').cast('string'))"
            )
        has_topic = schema.get_field_index("topic") >= 0
        if self._topic is None and not (caps.per_row_destination and has_topic):
            raise PlanError(
                f"the {label} sink needs a destination: pass topic=... to "
                f"write.{caps.broker}()"
                + (", or project a 'topic' column" if caps.per_row_destination else "")
            )
        for name in ("key", "value"):
            _check_payload_type(schema, name, label)
        if not caps.null_values and table.column("value").null_count:
            raise PlanError(
                f"the {label} sink cannot publish a null 'value' ({label} has no "
                "tombstone record); filter them out, e.g. .filter(col('value').is_not_null())"
            )
        self._validate_extra(table)

    def _records(self, table: pa.Table, batch_id: int) -> BrokerRecords:
        """Extract one micro-batch's per-row record fields for the client."""
        caps = self.capabilities
        topics = _optional(table, "topic") if caps.per_row_destination else None
        default = self._topic or ""
        destinations = (
            [default] * table.num_rows
            if topics is None
            else [t if t is not None else default for t in topics]
        )
        keys = _optional_payload(table, "key") if "key" in caps.columns else None
        raw_headers = _optional(table, "headers") if "headers" in caps.columns else None
        headers = None if raw_headers is None else [record_headers(h) for h in raw_headers]
        if self._dedup_ids is not None:
            prefix = f"{self._dedup_ids}:{batch_id}:"
            if headers is None:
                headers = [[] for _ in range(table.num_rows)]
            for i, pairs in enumerate(headers):
                pairs.append((DEDUP_ID_HEADER, f"{prefix}{i}".encode()))
        return BrokerRecords(
            values=payload_column(table, "value"),
            destinations=destinations,
            keys=keys,
            partitions=_optional(table, "partition") if "partition" in caps.columns else None,
            headers=headers,
        )


def _optional_payload(table: pa.Table, name: str) -> list[bytes | None] | None:
    """Payload column `name` as bytes, or `None` when the table has no such column."""
    if table.schema.get_field_index(name) < 0:
        return None
    return payload_column(table, name)


def _check_payload_type(schema: pa.Schema, name: str, label: str) -> None:
    """Reject a `key`/`value` column the broker cannot carry, naming it and its type."""
    index = schema.get_field_index(name)
    if index < 0:
        return
    field_type = schema.field(index).type
    if str(field_type) not in _PAYLOAD_TYPES:
        raise PlanError(
            f"the {label} sink's {name!r} column must be binary or string, not {field_type}; "
            f"serialize it first, e.g. .with_columns({name}=col({name}).cast('string'))"
        )
