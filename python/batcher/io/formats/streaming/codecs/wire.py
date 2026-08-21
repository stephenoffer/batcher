"""Confluent Schema Registry framing and the registry client that resolves schema ids.

A Kafka payload written by a Confluent serializer is not a bare Avro record. It carries a
five-byte prefix — a zero magic byte and a big-endian 4-byte schema id — and, for Protobuf,
a further varint array naming which message inside the descriptor was written. Decoding
such a payload as if it were bare fails in the least helpful way available: the first five
bytes are consumed as field data, so the reader does not error, it returns *plausible
garbage*. That is the whole reason this framing is handled here rather than left to a user's
`map_batches`.

The client is deliberately stdlib-only (`urllib.request`). A schema registry lookup is one
cached GET per distinct schema id for the lifetime of a source, so a dependency on an HTTP
library would put a package into every Batcher install to serve a request that is made a
handful of times per query.

Layer: `io`, neutral.
"""

from __future__ import annotations

import base64
import json
import threading
from dataclasses import dataclass
from typing import Any

from batcher._internal.errors import BackendError, PlanError

__all__ = [
    "CONFLUENT_MAGIC",
    "FramedPayload",
    "SchemaRegistry",
    "frame_confluent",
    "unframe_confluent",
]

#: The leading byte of every Confluent-framed payload. A payload that starts with anything
#: else was not written by a Confluent serializer, and decoding it as if it were would strip
#: five bytes of real data.
CONFLUENT_MAGIC = 0

#: Bytes of framing before the payload proper: the magic byte plus a 4-byte schema id.
_HEADER_LEN = 5


@dataclass(frozen=True, slots=True)
class FramedPayload:
    """One payload split into its Confluent framing and its body.

    ``message_indexes`` is populated only for Protobuf, where the framing additionally names
    the path to the message within the descriptor. It is an empty tuple for Avro and JSON,
    and for a Protobuf payload using the default first-message index (the wire encoding for
    which is the single byte ``0``, not an empty array).
    """

    schema_id: int
    body: bytes
    message_indexes: tuple[int, ...] = ()


def _read_zigzag_varint(data: bytes, pos: int) -> tuple[int, int]:
    """Read one zigzag-encoded base-128 varint, returning the value and the new position."""
    result = 0
    shift = 0
    while True:
        if pos >= len(data):
            raise BackendError("truncated Confluent Protobuf message-index varint")
        byte = data[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            break
        shift += 7
        if shift > 63:
            raise BackendError("Confluent Protobuf message-index varint is too long")
    return (result >> 1) ^ -(result & 1), pos


def _write_zigzag_varint(value: int) -> bytes:
    """Encode one integer as a zigzag base-128 varint."""
    zig = (value << 1) ^ (value >> 63)
    out = bytearray()
    while True:
        byte = zig & 0x7F
        zig >>= 7
        if zig:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def unframe_confluent(payload: bytes, *, protobuf: bool = False) -> FramedPayload:
    """Split a Confluent-framed payload into its schema id and body.

    Args:
        payload: The raw message bytes as delivered by the broker.
        protobuf: Whether to also consume the Protobuf message-index array that follows
            the schema id.

    Returns:
        The parsed framing and the remaining body.

    Raises:
        BackendError: If the payload is too short or does not begin with the magic byte.
            Both are reported rather than tolerated, because the failure this framing
            exists to prevent is a *silent* misread, and a payload that is not framed
            cannot be told from one that is except by this check.
    """
    if len(payload) < _HEADER_LEN:
        raise BackendError(
            f"payload is {len(payload)} bytes, too short to carry Confluent framing "
            f"({_HEADER_LEN} bytes of magic byte + schema id). Set schema_registry=None "
            "if these messages are not registry-framed."
        )
    if payload[0] != CONFLUENT_MAGIC:
        raise BackendError(
            f"payload does not start with the Confluent magic byte 0 (got {payload[0]}); "
            "it was not written by a Schema Registry serializer. Set schema_registry=None "
            "to decode it as a bare payload."
        )
    schema_id = int.from_bytes(payload[1:_HEADER_LEN], "big")
    pos = _HEADER_LEN
    indexes: tuple[int, ...] = ()
    if protobuf:
        count, pos = _read_zigzag_varint(payload, pos)
        if count == 0:
            # The single-byte `0` is the wire encoding of "the first message", not of an
            # empty index array. Reading it as empty picks the wrong message on any
            # descriptor with more than one, which decodes without error into wrong fields.
            indexes = (0,)
        else:
            values = []
            for _ in range(count):
                value, pos = _read_zigzag_varint(payload, pos)
                values.append(value)
            indexes = tuple(values)
    return FramedPayload(schema_id=schema_id, body=payload[pos:], message_indexes=indexes)


def frame_confluent(
    schema_id: int, body: bytes, *, message_indexes: tuple[int, ...] | None = None
) -> bytes:
    """Prefix `body` with Confluent framing for `schema_id`.

    Args:
        schema_id: The registered id of the writer schema.
        body: The serialized payload.
        message_indexes: The Protobuf message-index path, or None for Avro/JSON.

    Returns:
        The framed payload a Confluent deserializer will accept.
    """
    header = bytes([CONFLUENT_MAGIC]) + int(schema_id).to_bytes(4, "big")
    if message_indexes is None:
        return header + body
    if tuple(message_indexes) == (0,):
        return header + _write_zigzag_varint(0) + body
    prefix = _write_zigzag_varint(len(message_indexes))
    for index in message_indexes:
        prefix += _write_zigzag_varint(index)
    return header + prefix + body


class SchemaRegistry:
    """A Confluent Schema Registry client: schema id (or subject) to schema text.

    Two lookups, both cached for the client's lifetime. A schema id is immutable by the
    registry's own contract, so caching it is not a staleness trade — it is the only way a
    per-message id lookup can be affordable at all. A *subject's latest version* is not
    immutable, so it is resolved once at source construction and not re-polled: a reader
    schema that changed underneath a running query would silently change the stream's Arrow
    schema mid-flight, which no downstream operator can absorb.

    Examples:
        .. doctest::

            >>> from batcher.io.formats.streaming.codecs import SchemaRegistry
            >>> registry = SchemaRegistry("http://localhost:8081")  # doctest: +SKIP
            >>> registry.schema_by_id(1)  # doctest: +SKIP
            '{"type": "record", ...}'
    """

    __slots__ = ("_auth", "_by_id", "_lock", "_timeout", "_url")

    def __init__(
        self,
        url: str,
        *,
        username: str | None = None,
        password: str | None = None,
        basic_auth_user_info: str | None = None,
        timeout: float = 10.0,
    ) -> None:
        """Create a registry client for `url`.

        Args:
            url: The registry base URL, e.g. ``"http://localhost:8081"``.
            username: Basic-auth user, when the registry requires one.
            password: Basic-auth password.
            basic_auth_user_info: The Confluent spelling of the same credential,
                ``"user:password"``, accepted so a config copied from a Kafka client
                carries over verbatim.
            timeout: Seconds any one HTTP request may take. Bounded on purpose: an
                unreachable registry otherwise hangs source construction with no error,
                which reads as a stuck query rather than as the misconfiguration it is.
        """
        if not isinstance(url, str) or not url.strip():
            raise PlanError("schema_registry url must be a non-empty string")
        if not url.startswith(("http://", "https://")):
            # Checked here rather than at request time so a typo is reported where it was
            # written, and so `urlopen` is never handed a `file://` or `ftp://` URL.
            raise PlanError(f"schema_registry url must be http:// or https://, got {url!r}")
        self._url = url.rstrip("/")
        if basic_auth_user_info and username is None:
            username, _, password = basic_auth_user_info.partition(":")
        self._auth = None
        if username is not None:
            raw = f"{username}:{password or ''}".encode()
            self._auth = "Basic " + base64.b64encode(raw).decode("ascii")
        self._timeout = float(timeout)
        self._by_id: dict[int, str] = {}
        # A distributed source decodes on several worker threads against one client, and
        # two threads missing the same id would otherwise both issue the GET.
        self._lock = threading.Lock()

    @property
    def url(self) -> str:
        """The registry base URL this client was built for."""
        return self._url

    def _get(self, path: str) -> dict[str, Any]:
        """One GET against the registry, returning the decoded JSON body."""
        import urllib.error
        import urllib.request

        # The scheme was pinned to http(s) in `__init__`, so `urlopen` cannot be steered
        # at a local file or another handler by the option value.
        request = urllib.request.Request(f"{self._url}{path}")
        request.add_header("Accept", "application/vnd.schemaregistry.v1+json, application/json")
        if self._auth is not None:
            request.add_header("Authorization", self._auth)
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:  # pragma: no cover - needs a live registry
            detail = exc.read().decode("utf-8", "replace")[:200]
            raise BackendError(
                f"schema registry {self._url}{path} returned {exc.code}: {detail}"
            ) from exc
        except OSError as exc:  # pragma: no cover - needs a live registry
            raise BackendError(f"schema registry {self._url} is unreachable: {exc}") from exc

    def schema_by_id(self, schema_id: int) -> str:
        """The registered schema text for `schema_id`, cached.

        Args:
            schema_id: The id read from a payload's Confluent framing.

        Returns:
            The schema as registered, as text (Avro JSON, a ``.proto``, or JSON Schema).
        """
        cached = self._by_id.get(schema_id)
        if cached is not None:
            return cached
        with self._lock:
            cached = self._by_id.get(schema_id)
            if cached is not None:
                return cached
            body = self._get(f"/schemas/ids/{schema_id}")
            schema = body.get("schema")
            if not isinstance(schema, str):
                raise BackendError(
                    f"schema registry returned no schema text for id {schema_id}: {body!r}"
                )
            self._by_id[schema_id] = schema
            return schema

    def latest(self, subject: str) -> tuple[int, str]:
        """The id and schema text of `subject`'s latest version.

        Args:
            subject: The registry subject, conventionally ``"{topic}-value"``.

        Returns:
            ``(schema_id, schema_text)``.
        """
        body = self._get(f"/subjects/{subject}/versions/latest")
        schema_id = body.get("id")
        schema = body.get("schema")
        if not isinstance(schema_id, int) or not isinstance(schema, str):
            raise BackendError(
                f"schema registry returned no usable latest version for subject "
                f"{subject!r}: {body!r}"
            )
        self._by_id[schema_id] = schema
        return schema_id, schema

    def __repr__(self) -> str:
        """Name the registry without printing its credential."""
        return f"SchemaRegistry(url={self._url!r}, authenticated={self._auth is not None})"
