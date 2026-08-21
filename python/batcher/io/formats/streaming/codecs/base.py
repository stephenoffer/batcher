"""The payload-codec contract: a binary message column in, a typed Arrow column out.

A broker delivers opaque bytes. Every real streaming pipeline's first act is therefore to
turn those bytes into columns, and until now Batcher had no answer but `map_batches` and a
hand-written decoder — which is not a small inconvenience: it puts the wire format outside
the plan, so the engine cannot report the stream's real schema, cannot push a projection
into the decode, and cannot tell a malformed record from a bug in user code.

A `PayloadCodec` closes that. It is deliberately **column-at-a-time**, not row-at-a-time:
`decode` receives the whole `value` (or `key`) column of one poll and returns one Arrow
array. That keeps the boundary the same one `io.formats.structured.avro` and
`io.formats.semistructured.protobuf` already sit on — the unavoidable deserialization an
Arrow-less wire format requires, paid once per batch — rather than adding a per-row Python
call to the hot path, which `.claude/rules/architecture.md` forbids.

Codecs are registered by name in `CODECS`, so `value_format="avro"` reaches one the same
way `format="parquet"` reaches a source, and a third-party wire format plugs in without
forking the engine.

Layer: `io`, neutral. A codec knows about bytes and Arrow types, never about plans.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import pyarrow as pa
import pyarrow.compute as pc

from batcher._internal.errors import PlanError, suggestion
from batcher._internal.registry import Registry

__all__ = [
    "CODECS",
    "DecodeMode",
    "PayloadCodec",
    "build_payload_codecs",
    "null_mask_from",
    "resolve_codec",
    "scatter",
]

#: How a codec answers a payload it cannot decode.
#:
#: ``"fail"`` raises (Spark's ``FAILFAST``) and is the default, because a stream that
#: silently nulls every record when a producer changes format is a stream that reports
#: success while delivering nothing. ``"permissive"`` (Spark's ``PERMISSIVE``) yields a
#: null for the offending row and keeps going, which is what a topic with a known tail of
#: legacy records needs.
DecodeMode = str

_MODES = ("fail", "permissive")

#: Wire-format codecs, keyed by the name a user writes as ``value_format=``.
#:
#: Deferred registration (`on_miss`): each codec module imports its own optional driver
#: guard, and a base install should not pay for five of them to look up ``"json"``.
CODECS: Registry[Any] = Registry(
    "payload codec",
    doc="integrations/streams/payload-formats",
    on_miss=lambda: _register_builtin_codecs(),
)


def _register_builtin_codecs() -> None:
    """Import the built-in codec modules so each registers itself."""
    from batcher.io.formats.streaming.codecs import (  # noqa: F401
        avro as _avro,
    )
    from batcher.io.formats.streaming.codecs import (
        json as _json,  # noqa: F401
    )
    from batcher.io.formats.streaming.codecs import (
        protobuf as _protobuf,  # noqa: F401
    )
    from batcher.io.formats.streaming.codecs import (
        text as _text,  # noqa: F401
    )


@runtime_checkable
class PayloadCodec(Protocol):
    """Turns one column of message payloads into one typed Arrow column, and back.

    Implementations are constructed once per source (or sink) and reused for every
    micro-batch, so any schema resolution, registry lookup, or descriptor parsing belongs in
    `__init__` rather than in `decode`.
    """

    #: The registry name a user writes as ``value_format=``.
    name: str

    def arrow_type(self) -> pa.DataType:
        """The Arrow type `decode` produces, known before any message is read."""
        ...

    def decode(self, column: pa.Array) -> pa.Array:
        """Decode a binary payload column into `arrow_type()`."""
        ...

    def encode(self, column: pa.Array) -> pa.Array:
        """Encode a typed column back into binary payloads."""
        ...


def null_mask_from(column: pa.Array) -> list[bool]:
    """Which rows of a payload column are null, as a Python list of flags.

    Every codec needs this and each would otherwise recompute it differently. A null
    payload is not a decode failure — Kafka's tombstone record is a null `value` and means
    "this key is deleted" — so it must survive the round trip as a null rather than raising
    or becoming an empty struct.

    Args:
        column: The binary payload column.

    Returns:
        One flag per row, True where the payload is null.
    """
    if column.null_count == 0:
        return [False] * len(column)
    return [not valid for valid in column.is_valid().to_pylist()]


def scatter(decoded: pa.Array, keep: list[int], total: int, struct_type: pa.DataType) -> pa.Array:
    """Place `decoded` back at the row positions it came from, nulls elsewhere.

    Two codecs need this and for the same reason: a decoder that parses a *batch* (pyarrow's
    JSON reader, protarrow's message table) can only be handed the rows it can parse, so the
    result comes back dense and has to be re-expanded to the poll's row count. Getting that
    wrong shifts every later row onto the wrong message, which is a wrong answer rather than
    an error.

    Done with a take against an index array rather than a Python row loop, so the cost is one
    Arrow kernel over the batch instead of per-row work in the control plane.

    Args:
        decoded: The successfully decoded rows, densely packed.
        keep: The original row position of each entry of `decoded`, ascending.
        total: The row count of the batch being rebuilt.
        struct_type: The decoded column's type, used to type the null rows.

    Returns:
        An array of `total` rows: `decoded` at the `keep` positions, null elsewhere.
    """
    import numpy as np

    indices = np.full(total, -1, dtype=np.int64)
    indices[np.asarray(keep, dtype=np.int64)] = np.arange(len(keep), dtype=np.int64)
    mask = indices >= 0
    taken = decoded.take(pa.array(np.where(mask, indices, 0)))
    return pc.if_else(pa.array(mask), taken, pa.nulls(total, type=struct_type))


def _check_mode(mode: str) -> str:
    """Validate a decode mode at construction, where the option was written."""
    if mode not in _MODES:
        hint = suggestion(mode, _MODES)
        raise PlanError(
            f"unknown decode mode {mode!r}; use 'fail' or 'permissive'."
            + (f" {hint}" if hint else "")
        )
    return mode


def resolve_codec(
    spec: Any,
    *,
    schema: Any = None,
    registry: Any = None,
    subject: str | None = None,
    mode: str = "fail",
    **options: Any,
) -> Any:
    """Build the codec named by `spec`, or return it unchanged if it already is one.

    Accepting an already-built codec is what lets a user who needs a wire format Batcher
    does not ship pass their own object to ``value_format=`` with no registration step,
    while the common case stays a string.

    Args:
        spec: A registered codec name (``"avro"``, ``"json"``, ``"protobuf"``,
            ``"string"``, ``"bytes"``), or a `PayloadCodec` instance to use as-is.
        schema: The reader schema, in whatever form the codec accepts (an Avro schema
            dict/JSON string, an Arrow schema, a protobuf message class).
        registry: A `SchemaRegistry` when payloads carry Confluent framing, else None.
        subject: The registry subject to resolve the reader schema from. Defaults to the
            standard ``"{topic}-value"`` / ``"{topic}-key"`` naming, supplied by the caller.
        mode: `DecodeMode` — ``"fail"`` or ``"permissive"``.
        options: Passed through to the concrete codec.

    Returns:
        A ready `PayloadCodec`, or None when `spec` is None.

    Raises:
        PlanError: If `spec` names no registered codec, or `mode` is not a decode mode.
    """
    if spec is None:
        return None
    if not isinstance(spec, str):
        if isinstance(spec, PayloadCodec):
            return spec
        raise PlanError(
            f"value_format/key_format must be a codec name or a PayloadCodec, not "
            f"{type(spec).__name__}; known names are {sorted(CODECS.names())}"
        )
    codec_cls = CODECS.get(spec)
    return codec_cls(
        schema=schema,
        registry=registry,
        subject=subject,
        mode=_check_mode(mode),
        **options,
    )


def _as_registry(spec: Any, auth: str | None) -> Any:
    """Accept a registry as a URL string or an already-built `SchemaRegistry`."""
    if spec is None:
        return None
    from batcher.io.formats.streaming.codecs.wire import SchemaRegistry

    if isinstance(spec, SchemaRegistry):
        return spec
    if isinstance(spec, str):
        return SchemaRegistry(spec, basic_auth_user_info=auth)
    raise PlanError(
        f"schema_registry must be a URL string or a SchemaRegistry, not {type(spec).__name__}"
    )


def build_payload_codecs(topic: str, config: dict[str, Any]) -> tuple[Any, Any]:
    """Build the ``(value, key)`` codecs a broker source was configured with.

    One function rather than two call sites per broker, because the two payload columns are
    configured identically and differ only in the registry subject they default to. That
    default is the Confluent **TopicNameStrategy** — ``"{topic}-value"`` and
    ``"{topic}-key"`` — which is what every Confluent serializer registers under, so a topic
    written by a standard producer resolves with no subject option at all.

    A single registry client is shared by both codecs so its schema cache is shared too: a
    key and a value schema fetched separately would otherwise open two connections and hold
    two caches for one topic.

    Args:
        topic: The topic being read, for the default subject names.
        config: The raw codec options as the source received them.

    Returns:
        ``(value_codec, key_codec)``, either of which is None when no format was named.
    """
    registry = _as_registry(config.get("schema_registry"), config.get("schema_registry_auth"))
    codecs = []
    for side in ("value", "key"):
        spec = config.get(f"{side}_format")
        if spec is None:
            codecs.append(None)
            continue
        codecs.append(
            resolve_codec(
                spec,
                schema=config.get(f"{side}_schema"),
                registry=registry,
                subject=config.get(f"{side}_subject") or f"{topic}-{side}",
                mode=config.get(f"{side}_decode_mode", "fail"),
            )
        )
    return codecs[0], codecs[1]
