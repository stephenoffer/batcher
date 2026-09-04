"""Arrow tables to bytes and back, for anything that stores a result outside the process.

The one place the engine's in-memory ↔ over-the-wire conversion is written. Arrow IPC is
already the engine's serialized form, so this adds no representation: it frames the
buffers a `Table` already holds, optionally compressed by the same codec the spill path
uses. Round-tripping preserves the schema exactly, which is the property that matters —
invariant #7 makes a column *type* change a wrong answer, not a formatting difference,
and a codec that widened an int on the way home would be exactly that.

It lives in `plan.types` beside `logical_bytes`/`retained_bytes` because a shared result
cache (`carbonite.cache_shared`) needs it and so would any future caller in `io` or
`dist`, none of which may import each other. The existing IPC writers in those two
packages write to *files* rather than to a buffer and are not this in disguise.
"""

from __future__ import annotations

import pyarrow as pa
from pyarrow import ipc

__all__ = ["table_from_ipc", "table_to_ipc"]

#: Codecs `pa.ipc.IpcWriteOptions` accepts. `lz4` is the default because it is what the
#: spill path already writes: on Arrow buffers it runs at roughly memory bandwidth, so on
#: any store reached over a socket the compression is free relative to the transfer it
#: saves. `zstd` trades CPU for ratio and suits a slow or metered link.
COMPRESSIONS: tuple[str, ...] = ("lz4", "zstd")


def table_to_ipc(table: pa.Table, compression: str | None = "lz4") -> bytes:
    """Serialize `table` to a self-describing Arrow IPC stream.

    Args:
        table: The table to serialize. An empty table is still framed, carrying its
            schema, so a cached empty result round-trips as an empty table of the right
            type rather than as nothing.
        compression: One of `COMPRESSIONS`, or `None` for uncompressed. An unavailable
            codec (a pyarrow built without it) falls back to uncompressed rather than
            raising: the bytes are a cache entry, and refusing to write one because a
            codec is missing trades a working slow path for a failure.

    Returns:
        The serialized stream.

    Examples:
        .. doctest::

            >>> import pyarrow as pa
            >>> from batcher.plan.types.ipc import table_from_ipc, table_to_ipc
            >>> t = pa.table({"x": pa.array([1, 2, 3], pa.int32())})
            >>> table_from_ipc(table_to_ipc(t)).equals(t)
            True
    """
    sink = pa.BufferOutputStream()
    options = _write_options(compression)
    with ipc.new_stream(sink, table.schema, options=options) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


def table_from_ipc(data: bytes) -> pa.Table:
    """Deserialize an Arrow IPC stream produced by `table_to_ipc`.

    Args:
        data: The serialized stream.

    Returns:
        The table, with the schema it was written with.

    Raises:
        pyarrow.ArrowInvalid: If `data` is not a readable Arrow IPC stream. Callers
            storing these bytes somewhere mutable are expected to catch this and treat a
            corrupt entry as a miss.

    Examples:
        .. doctest::

            >>> import pyarrow as pa
            >>> from batcher.plan.types.ipc import table_from_ipc, table_to_ipc
            >>> table_from_ipc(table_to_ipc(pa.table({"x": []}))).num_rows
            0
    """
    with ipc.open_stream(pa.BufferReader(data)) as reader:
        return reader.read_all()


def _write_options(compression: str | None) -> ipc.IpcWriteOptions | None:
    """Write options for `compression`, or `None` when it is unavailable or not wanted."""
    if compression is None or compression not in COMPRESSIONS:
        return None
    try:
        return ipc.IpcWriteOptions(compression=compression)
    except (pa.ArrowNotImplementedError, pa.ArrowInvalid, ValueError):
        return None  # a pyarrow built without this codec; write it plain
