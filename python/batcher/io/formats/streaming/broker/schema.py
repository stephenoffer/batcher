"""The fixed broker message schema, the message record, and option redaction.

Everything here is shared by every concrete broker and is deliberately free of any client:
the schema is a constant, `BrokerMessage` is a plain record, and redaction is a pure
function over an option mapping. Keeping them apart from `BrokerSource` is what lets the
split module import them without pulling in the poll loop.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pyarrow as pa

from batcher.io.formats.sql._common import connection_fingerprint

__all__ = [
    "HEADERS_TYPE",
    "BrokerMessage",
    "as_header_pairs",
    "broker_schema",
    "normalize_starting_position",
    "opaque_offset",
    "redact_broker_options",
]


def as_header_pairs(properties: Any) -> list[tuple[str, bytes | None]] | None:
    """One broker's per-message metadata as the ``[(name, bytes)]`` the schema holds.

    Kafka calls them headers, Pulsar calls them *properties*, Pub/Sub calls them
    *attributes* and Event Hubs calls them *properties* again — and all four are the same
    idea: a small string-keyed map riding with the payload, carrying a trace id, a schema
    reference, a routing hint or a tenant. Only Kafka's ever reached the `headers` column,
    so ``include_headers=True`` on the other three produced a column of nulls: the option
    was accepted, the work was done, and nothing said the data was not there.

    The value is normalized to bytes, because that is what the column's type is and because
    the four clients disagree — Pulsar and Pub/Sub hand back `str`, Event Hubs hands back
    whatever was published. A key that is bytes is decoded, since the column's key is
    `string`.

    Args:
        properties: The client's metadata mapping, or None/empty when there is none.

    Returns:
        The pairs, or None when the message carried no metadata — the same null-versus-empty
        distinction Kafka's headers already draw, so "this message had none" stays
        distinguishable from "this broker does not carry them".
    """
    if not properties:
        return None
    pairs: list[tuple[str, bytes | None]] = []
    for name, value in dict(properties).items():
        key = name.decode("utf-8", "replace") if isinstance(name, bytes) else str(name)
        if value is None or isinstance(value, bytes):
            payload = value
        elif isinstance(value, str):
            payload = value.encode("utf-8")
        else:
            payload = str(value).encode("utf-8")
        pairs.append((key, payload))
    return pairs


def opaque_offset(position: str) -> int:
    """Map a broker's opaque native position onto the fixed int64 ``offset`` column.

    Kinesis sequence numbers and Event Hubs offsets are *text*, and the broker schema's
    ``offset`` is ``int64``. A numeric position is taken modulo ``2**63``, which preserves
    the within-partition ordering the column exists for. Anything else falls back to a
    `sha256` digest rather than `hash()`, because Python salts `str` hashing per process:
    with `hash()` the same record got a different ``offset`` on every run and on every
    worker, silently breaking the ordering and de-duplication the column is for, across
    exactly the restart and distributed boundaries that matter.

    The *native* position still travels separately as `BrokerMessage.resume_token`; this
    is the lossy projection onto a schema column, never what a client seeks with.

    Args:
        position: The broker's native position, as text.

    Returns:
        A stable, non-negative int64 offset.
    """
    import hashlib

    try:
        return int(position) % (1 << 63)
    except ValueError:
        digest = hashlib.sha256(position.encode("utf-8")).digest()[:8]
        return int.from_bytes(digest, "big") % (1 << 63)


#: The two whole-stream starting positions every broker understands, under the names Spark
#: uses. Each concrete broker maps them onto its own vocabulary (Kafka's
#: ``auto.offset.reset``, Kinesis's ``ShardIteratorType``, Event Hubs' offset sentinel).
STARTING_POSITIONS = ("earliest", "latest")


def normalize_starting_position(value: object, *, aliases: dict[str, str]) -> str:
    """Map a `starting_position` onto one concrete broker's own vocabulary.

    Five brokers, five spellings of the same two ideas: Kafka says
    ``auto.offset.reset=earliest``, Kinesis says ``ShardIteratorType=TRIM_HORIZON``, Event
    Hubs says offset ``-1``, Pulsar says ``MessageId.earliest``. A reader who knows one has
    to look up the other four, and a job ported from Spark knows none of them — it knows
    ``startingOffsets`` / ``startingPosition``, which is ``"earliest"`` or ``"latest"``.

    So the *option* is the same everywhere and the mapping lives with the broker that needs
    it. A broker's native spelling still works, because refusing it would break the readers
    that already pass one.

    Args:
        value: What the user asked for — a shared name, or this broker's native spelling.
        aliases: This broker's ``{shared_name: native_value}`` mapping.

    Returns:
        The native value to hand the client.

    Raises:
        PlanError: If `value` is neither a shared name nor a recognized native one.
    """
    from batcher._internal.errors import PlanError, suggestion

    if not isinstance(value, str):
        raise PlanError(
            f"starting_position must be a string, not {type(value).__name__} ({value!r}); "
            f"use one of {list(STARTING_POSITIONS)}"
        )
    if value in aliases:
        return aliases[value]
    native = set(aliases.values())
    if value in native:
        return value  # this broker's own spelling, passed through unchanged
    known = [*STARTING_POSITIONS, *sorted(native)]
    hint = suggestion(value, known)
    raise PlanError(
        f"unknown starting_position {value!r}; use one of {known}." + (f" {hint}" if hint else "")
    )


#: Two forces pull on this list. Too narrow and a credential reaches a log line and the
#: persisted stats key; too broad and unrelated options collapse to one fingerprint, so two
#: genuinely different clusters share a learned-statistics entry. A bare ``"sas"`` was too
#: broad in exactly that way — it matched every ``sasl.*`` key, including the non-secret
#: ``sasl.mechanism`` and ``sasl.username`` — while still missing an inline PEM private key,
#: which librdkafka takes as ``ssl.key.pem`` and no hint here matched.
_BROKER_SECRET_HINTS = (
    "password",
    "passwd",
    "pwd",
    "secret",
    "token",
    "credential",
    "api_key",
    "apikey",
    "connection_str",
    "conn_str",
    "sas_key",
    "saskey",
    "shared_access",
    "sharedaccess",
    "private_key",
    "privatekey",
    "key.pem",
    "key_pem",
    "keystore",
    "truststore",
    "certificate",
)


def redact_broker_options(options: Mapping[str, Any]) -> dict[str, Any]:
    """Broker client options with every credential-bearing value masked.

    Two call sites need this and they fail differently. A `repr` leak prints a SASL password
    into a traceback or a log line. An `identity()` leak is worse: identity is the key learned
    statistics are *persisted* under, so the credential is written to the metadata store and
    outlives the process that held it.

    Masking rather than dropping is what keeps the key stable across a credential rotation —
    a rotated password maps to the same ``"***"``, so the topic's accumulated statistics are
    not orphaned on every rotation. This mirrors
    `batcher.io.formats.sql.odbc.redact_connection_string`, which exists for the same reason.

    Args:
        options: Broker client options, as passed through to the concrete client.

    Returns:
        A new mapping with credential-bearing values replaced by ``"***"``.
    """
    return {
        key: ("***" if any(hint in key.lower() for hint in _BROKER_SECRET_HINTS) else value)
        for key, value in options.items()
    }


def _options_fingerprint(options: Mapping[str, Any]) -> str:
    """The connection's contribution to an identity — redacted, then fingerprinted."""
    return connection_fingerprint(redact_broker_options(options))


#: Built once. `pa.Schema` is immutable, so every broker, split, and assembled batch can share
#: the one instance — and they should: `_make_batch` asked for a fresh schema on *every*
#: micro-batch, allocating six `pa.Field` objects per poll on the latency-critical path, and
#: two batches built from separately-constructed (equal) schemas do not share a schema pointer,
#: so downstream concatenation had to fall back to a field-by-field comparison.
_BROKER_FIELDS = [
    pa.field("key", pa.binary()),
    pa.field("value", pa.binary()),
    pa.field("partition", pa.int64()),
    pa.field("offset", pa.int64()),
    pa.field("timestamp", pa.int64()),
    pa.field("topic", pa.string()),
]

#: The `headers` column's type, matching Spark's Kafka source exactly
#: (``array<struct<key:string,value:binary>>``) so a ported job's accessors keep working.
HEADERS_TYPE = pa.list_(pa.struct([pa.field("key", pa.string()), pa.field("value", pa.binary())]))

_BROKER_SCHEMA = pa.schema(_BROKER_FIELDS)
_BROKER_SCHEMA_WITH_HEADERS = pa.schema([*_BROKER_FIELDS, pa.field("headers", HEADERS_TYPE)])


def broker_schema(
    include_headers: bool = False,
    *,
    value_type: pa.DataType | None = None,
    key_type: pa.DataType | None = None,
) -> pa.Schema:
    """The broker message schema shared by every broker source.

    `include_headers` adds the `headers` column, off by default and opt-in for the reason
    Spark's ``includeHeaders`` is: headers are per-message metadata most pipelines never
    read, and decoding them into a nested Arrow column costs on every message of every
    poll. A stream that does want them — for a trace id, a schema-registry id, a routing
    hint — could not reach them at all before.

    `value_type` / `key_type` retype the payload columns when the source was given a wire
    format (``value_format="avro"``, …). The retype has to happen *here* rather than after
    the batch is built, because this schema is what `Dataset.schema` answers from and what
    the optimizer plans against: a source that decodes to a struct but advertises `binary`
    type-checks every downstream expression against the wrong type and only fails once rows
    arrive.

    Args:
        include_headers: Whether to add the `headers` column.
        value_type: The decoded type of the `value` column, or None for raw bytes.
        key_type: The decoded type of the `key` column, or None for raw bytes.

    Returns:
        The schema for that choice. The undecoded cases return the *shared* immutable
        instance, so the common path allocates nothing per poll.
    """
    if value_type is None and key_type is None:
        return _BROKER_SCHEMA_WITH_HEADERS if include_headers else _BROKER_SCHEMA
    base = _BROKER_SCHEMA_WITH_HEADERS if include_headers else _BROKER_SCHEMA
    retyped = {"value": value_type, "key": key_type}
    return pa.schema(
        [pa.field(f.name, retyped[f.name]) if retyped.get(f.name) is not None else f for f in base]
    )


@dataclass(frozen=True, slots=True)
class BrokerMessage:
    """One polled message: raw bytes plus its broker coordinates.

    ``key`` may be ``None`` (an unkeyed message); all other fields are required.
    ``timestamp`` is milliseconds since the Unix epoch.

    ``resume_token`` is the *native* position a client seeks strictly after to
    replay from this message on recovery (a Kinesis sequence number, a Pulsar
    message id, …). It is checkpoint bookkeeping only — never a schema column —
    and defaults to ``None``, in which case the int64 ``offset`` is the token.
    """

    value: bytes
    partition: int
    offset: int
    timestamp: int
    topic: str
    key: bytes | None = None
    resume_token: Any = None
    #: Per-message headers as ``[(name, value)]``, or None when the source was not asked
    #: for them. Only populated when the broker supports headers *and* the reader opted in.
    headers: list[tuple[str, bytes | None]] | None = None
