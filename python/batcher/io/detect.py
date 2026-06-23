"""Format auto-detection for the generic `read(path, format=None)` entry point.

Resolves a format name (a `SOURCES` registry key) from an explicit override, the
path's URI scheme, or its file extension. Table/database sources (delta, iceberg,
sql, …) are addressed by their explicit `read_*` functions, not by extension.
"""

from __future__ import annotations

import os

from batcher._internal.errors import FormatError

__all__ = ["detect_format"]

_EXT_TO_FORMAT: dict[str, str] = {
    ".parquet": "parquet",
    ".pq": "parquet",
    ".csv": "csv",
    ".tsv": "csv",
    ".json": "json",
    ".ndjson": "json",
    ".jsonl": "json",
    ".orc": "orc",
    ".arrow": "arrow",
    ".feather": "feather",
    ".ipc": "ipc",
    ".avro": "avro",
    ".xlsx": "excel",
    ".xls": "excel",
    ".lance": "lance",
    ".xml": "xml",
    ".log": "logs",
    ".pb": "protobuf",
    ".msgpack": "msgpack",
    ".mp": "msgpack",
    ".txt": "text",
    ".text": "text",
    ".pdf": "documents",
    ".npy": "numpy",
    ".npz": "numpy",
    ".tfrecord": "tfrecord",
    ".tfrecords": "tfrecord",
    ".tar": "webdataset",
    ".h5": "hdf5",
    ".hdf5": "hdf5",
    ".zarr": "zarr",
}

# URI schemes that name a source type directly (e.g. delta://, iceberg://).
_SCHEME_TO_FORMAT: dict[str, str] = {
    "delta": "delta",
    "iceberg": "iceberg",
    "hudi": "hudi",
}


def _scheme(path: str) -> str:
    idx = path.find("://")
    return path[:idx].lower() if idx > 0 else ""


def _ext(path: str) -> str:
    # Strip a trailing slash (directory) and any glob suffix before taking the ext.
    base = path.rstrip("/").split("*", 1)[0]
    return os.path.splitext(base)[1].lower()


def detect_format(path: str, explicit: str | None = None) -> str:
    """Resolve the format name for `path`, preferring an `explicit` override.

    Order: explicit → URI scheme (delta/iceberg/…) → file extension. For a
    directory or glob with no extension, the caller should pass `format=`.
    Raises `FormatError` if the format cannot be inferred.
    """
    if explicit:
        return explicit
    scheme = _scheme(path)
    if scheme in _SCHEME_TO_FORMAT:
        return _SCHEME_TO_FORMAT[scheme]
    ext = _ext(path)
    if ext in _EXT_TO_FORMAT:
        return _EXT_TO_FORMAT[ext]
    raise FormatError(
        f"could not infer a format for {path!r}; pass format=... "
        f"(e.g. read({path!r}, format='parquet'))"
    )
