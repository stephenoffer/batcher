"""The few Flight SQL protobuf messages the service reads and writes, encoded by hand.

Flight SQL wraps every command in a ``google.protobuf.Any`` whose ``type_url`` names an
``arrow.flight.protocol.sql`` message. pyarrow ships Flight but not those message classes,
and every message the service needs carries only ``bytes``, ``string``, ``int64`` and enum
fields, so the wire format here is the protobuf encoding of exactly those: a varint key
(field number and wire type), then a varint or a length-prefixed value.

Field numbers are the ones in Apache Arrow's ``format/FlightSql.proto`` and ``Flight.proto``.
`tests/unit/test_flightsql_proto.py` holds this codec against the protobuf runtime's own
parser over descriptors built from those definitions, so a wrong number fails there rather
than against a client.

This is the `integrations` layer.
"""

from __future__ import annotations

__all__ = ["decode", "encode", "pack_any", "unpack_any"]

#: The package every Flight SQL message lives in.
SQL_PREFIX = "type.googleapis.com/arrow.flight.protocol.sql."

_VARINT, _LEN = 0, 2


def _varint(value: int) -> bytes:
    value &= (1 << 64) - 1  # protobuf encodes a negative int64 as its two's complement
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _read_varint(data: bytes, pos: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        if pos >= len(data):
            raise ValueError("truncated protobuf varint")
        byte = data[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
        shift += 7


def encode(fields: dict[int, bytes | str | int | None]) -> bytes:
    """Encode `fields` (field number to value) as one protobuf message.

    `bytes` and `str` are length-delimited, `int` is a varint, and None or an empty value
    is omitted, as proto3 omits a field at its default.

    Args:
        fields: Field number to value.

    Returns:
        The encoded message.
    """
    out = bytearray()
    for number, value in sorted(fields.items()):
        if value is None or value == b"" or value == "" or value == 0:
            continue
        if isinstance(value, int):
            out += _varint(number << 3 | _VARINT) + _varint(value)
        else:
            raw = value.encode() if isinstance(value, str) else value
            out += _varint(number << 3 | _LEN) + _varint(len(raw)) + raw
    return bytes(out)


def decode(data: bytes) -> dict[int, bytes | int]:
    """Decode one protobuf message into field number to raw value; the last repeat wins.

    Args:
        data: The encoded message.

    Returns:
        Field number to `bytes` (length-delimited) or `int` (varint).

    Raises:
        ValueError: A wire type other than varint or length-delimited, or a truncation.
    """
    fields: dict[int, bytes | int] = {}
    pos = 0
    while pos < len(data):
        key, pos = _read_varint(data, pos)
        number, wire = key >> 3, key & 7
        if wire == _VARINT:
            fields[number], pos = _read_varint(data, pos)
        elif wire == _LEN:
            size, pos = _read_varint(data, pos)
            if pos + size > len(data):
                raise ValueError("truncated protobuf field")
            fields[number] = data[pos : pos + size]
            pos += size
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")
    return fields


def pack_any(message: str, fields: dict[int, bytes | str | int | None]) -> bytes:
    """Encode a Flight SQL message wrapped in ``google.protobuf.Any``.

    Args:
        message: The message name, such as ``"TicketStatementQuery"``.
        fields: Its fields.

    Returns:
        The encoded ``Any``.
    """
    return encode({1: SQL_PREFIX + message, 2: encode(fields)})


def unpack_any(data: bytes) -> tuple[str, dict[int, bytes | int]]:
    """Decode a ``google.protobuf.Any`` holding a Flight SQL message.

    Args:
        data: The encoded ``Any``.

    Returns:
        The message name (the part of the type URL after the package) and its fields.

    Raises:
        ValueError: `data` is not an ``Any`` naming an ``arrow.flight.protocol.sql`` message.
    """
    outer = decode(data)
    url = outer.get(1, b"")
    url = url.decode() if isinstance(url, bytes) else ""
    if not url.startswith(SQL_PREFIX):
        raise ValueError(f"not a Flight SQL command (type_url {url!r})")
    inner = outer.get(2, b"")
    return url[len(SQL_PREFIX) :], decode(inner if isinstance(inner, bytes) else b"")
