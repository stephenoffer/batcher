"""Format auto-detection for the generic `read(path, format=None)` entry point.

Resolves a format name (a `SOURCES` registry key) from an explicit override, the
path's URI scheme, or its file extension. Table/database sources (delta, iceberg,
sql, …) are addressed by their explicit `read_*` functions, not by extension.
"""

from __future__ import annotations

import os

from batcher._internal.errors import FormatError

__all__ = ["DATA_SUFFIXES", "detect_format", "format_for_extension"]

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
    ".pcd": "point_cloud",
    ".ply": "point_cloud",
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


#: A transactional table announces itself by the metadata directory at its root. Detecting
#: it matters for more than convenience: a Delta table *is* a directory of Parquet files,
#: so a caller who cannot name it is one `format="parquet"` away from treating the table as
#: a plain directory — and a maintenance rewrite that does so deletes the data files the
#: older versions still reference, destroying time travel.
_TABLE_MARKERS: tuple[tuple[str, str], ...] = (
    ("_delta_log", "delta"),
    ("metadata", "iceberg"),
    (".hoodie", "hudi"),
)


def _table_at(path: str) -> str | None:
    """The table format rooted at `path`, from its metadata directory, or None.

    Local-filesystem check only, and best-effort: a remote or unreadable path simply
    yields None and the caller falls back to naming the format explicitly.
    """
    if "://" in path:
        return None
    root = path.rstrip("/")
    if not os.path.isdir(root):
        return None
    for marker, fmt in _TABLE_MARKERS:
        if os.path.isdir(os.path.join(root, marker)):
            return fmt
    return None


def detect_format(path: str, explicit: str | None = None) -> str:
    """Resolve the format name for `path`, preferring an `explicit` override.

    Order: explicit → URI scheme (delta/iceberg/…) → a table's metadata directory
    (``_delta_log`` / ``metadata`` / ``.hoodie``) → file extension. Raises `FormatError`
    if the format cannot be inferred.
    """
    if explicit:
        return explicit
    scheme = _scheme(path)
    if scheme in _SCHEME_TO_FORMAT:
        return _SCHEME_TO_FORMAT[scheme]
    table = _table_at(path)
    if table is not None:
        return table
    ext = _ext(path)
    if ext in _EXT_TO_FORMAT:
        return _EXT_TO_FORMAT[ext]
    raise FormatError(
        f"could not infer a format for {path!r}; pass format=... "
        f"(e.g. read({path!r}, format='parquet'))"
    )


#: Every data-file extension the registry knows, for a caller that must list a directory
#: and work out what is *in* it (`expand` takes the whole tuple in one listing pass).
DATA_SUFFIXES: tuple[str, ...] = tuple(_EXT_TO_FORMAT)


def format_for_extension(ext: str) -> str | None:
    """The registered format for a file extension (``".parquet"`` → ``"parquet"``), or None.

    The inverse of the extension rule `detect_format` applies to a path. It exists for the
    one caller that cannot use `detect_format`: a table stored as a **directory** has no
    extension of its own, but its data files do — and a `MERGE` always has an existing
    target to look at, so it can infer the format from the files rather than demanding
    `format=` for the layout every warehouse actually uses.

    Args:
        ext: A file extension, with the leading dot.

    Returns:
        The format name, or None if the extension is not a known data format.
    """
    return _EXT_TO_FORMAT.get(ext.lower())
