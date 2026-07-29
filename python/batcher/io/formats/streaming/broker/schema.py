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

__all__ = ["BrokerMessage", "broker_schema", "redact_broker_options"]


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
_BROKER_SCHEMA = pa.schema(
    [
        pa.field("key", pa.binary()),
        pa.field("value", pa.binary()),
        pa.field("partition", pa.int64()),
        pa.field("offset", pa.int64()),
        pa.field("timestamp", pa.int64()),
        pa.field("topic", pa.string()),
    ]
)


def broker_schema() -> pa.Schema:
    """The fixed broker message schema shared by every broker source."""
    return _BROKER_SCHEMA


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
