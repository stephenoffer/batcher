"""Format auto-detection for the generic `read(path, format=None)` entry point.

Resolves a format name (a `SOURCES` registry key) from an explicit override, the
path's URI scheme, or its file extension. Table/database sources (delta, iceberg,
sql, …) are addressed by their explicit `read_*` functions, not by extension.
"""

from __future__ import annotations

import os
from typing import Any

from batcher._internal.errors import FormatError, unknown_value

__all__ = [
    "COMPRESSION_SUFFIXES",
    "DATA_SUFFIXES",
    "compression_for_path",
    "detect_format",
    "format_for_extension",
]

#: Compression suffixes that wrap another format rather than being one. ``events.csv.gz``
#: is a CSV, and every engine users come from treats it that way, so the suffix is stripped
#: before the format is read off the name — and reported separately by
#: `compression_for_path` so the reader can decompress the stream.
COMPRESSION_SUFFIXES: dict[str, str] = {
    ".gz": "gzip",
    ".gzip": "gzip",
    ".bz2": "bz2",
    ".zst": "zstd",
    ".zstd": "zstd",
    ".xz": "lzma",
    ".lzma": "lzma",
    ".lz4": "lz4",
    ".br": "brotli",
}

_EXT_TO_FORMAT: dict[str, str] = {
    ".parquet": "parquet",
    ".pq": "parquet",
    ".parq": "parquet",
    ".csv": "csv",
    ".tsv": "csv",
    ".tab": "csv",
    ".psv": "csv",
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
    ".mcap": "mcap",
    ".mf4": "mdf",
    ".arrows": "arrow",
    ".lnc": "lance",
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
    """The format-bearing extension of `path`, with any compression suffix stripped.

    ``events.csv.gz`` is a CSV: the ``.gz`` says how the bytes are packed, not what they
    mean. Taking `splitext` alone reported ``.gz``, so every compressed file — the shape
    an export pipeline produces by default — failed detection and had to name `format=`.
    """
    # Strip a trailing slash (directory) and any glob suffix before taking the ext.
    base = path.rstrip("/").split("*", 1)[0]
    stem, ext = os.path.splitext(base)
    if ext.lower() in COMPRESSION_SUFFIXES:
        ext = os.path.splitext(stem)[1]
    return ext.lower()


def compression_for_path(path: str) -> str | None:
    """The compression codec named by `path`'s suffix, or None if it names none.

    Args:
        path: A file path or URI.

    Returns:
        A `pyarrow.CompressedInputStream` codec name (``"gzip"``, ``"zstd"``, …), or
        None when the path carries no compression suffix.

    Examples:
        .. doctest::

            >>> from batcher.io.detect import compression_for_path
            >>> compression_for_path("events.csv.gz")
            'gzip'
            >>> compression_for_path("events.csv") is None
            True
    """
    base = path.rstrip("/").split("*", 1)[0].split("?", 1)[0]
    return COMPRESSION_SUFFIXES.get(os.path.splitext(base)[1].lower())


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


def _registered_sources() -> tuple[str, ...]:
    """Every registered source name, for a suggestion or an alternatives list.

    Imported lazily: `io.formats` imports every format module, and `detect` is imported
    from inside that graph, so a module-level import would be circular.
    """
    from batcher.io.formats.base import SOURCES

    return tuple(sorted(SOURCES.names()))


def _validate_explicit(explicit: str) -> str:
    """Check an explicitly-named format against the registry, suggesting a near miss.

    Without this the name travelled to `SOURCES.get`, whose registry-level error names no
    format vocabulary a user thinks in and offers no suggestion — so ``format="parquett"``
    read as a bare "unknown source" rather than the one-character typo it is.
    """
    if explicit in _registered_sources():
        return explicit
    raise unknown_value(
        FormatError,
        "format",
        explicit,
        _registered_sources(),
        hint="omit format= to infer it from the path's extension or URI scheme.",
    )


def detect_format(path: Any, explicit: str | None = None) -> str:
    """Resolve the format name for `path`, preferring an `explicit` override.

    Order: explicit → URI scheme (delta/iceberg/…) → a table's metadata directory
    (``_delta_log`` / ``metadata`` / ``.hoodie``) → file extension, with any compression
    suffix stripped so ``events.csv.gz`` resolves as CSV. Raises `FormatError` if the
    format cannot be inferred, naming what it *could* have inferred.

    Args:
        path: The path, URI, `pathlib.Path`, or list of paths to infer from.
        explicit: A format name that overrides inference. Checked against the registry.

    Returns:
        The registered source name to read `path` with.

    Examples:
        .. doctest::

            >>> from batcher.io.detect import detect_format
            >>> detect_format("s3://bucket/events.csv.gz")
            'csv'
            >>> detect_format("data/", explicit="parquet")
            'parquet'
    """
    if explicit:
        return _validate_explicit(explicit)
    from batcher.io.base._paths import normalize_source_path

    # A list of files is one relation; the first names the format they all share.
    root, files = normalize_source_path(path)
    path = files[0] if files else root
    scheme = _scheme(path)
    if scheme in _SCHEME_TO_FORMAT:
        return _SCHEME_TO_FORMAT[scheme]
    table = _table_at(path)
    if table is not None:
        return table
    ext = _ext(path)
    if ext in _EXT_TO_FORMAT:
        return _EXT_TO_FORMAT[ext]
    # Suggest over the *extensions*, not the format names: the user wrote a filename, so
    # ``.parquett`` should point at ``.parquet`` rather than at a format vocabulary they
    # never typed. A directory or extension-less path gets no suggestion and the plain
    # "name the format" instruction, which is the only fix available to it.
    raise unknown_value(
        FormatError,
        "file extension",
        ext or path,
        tuple(_EXT_TO_FORMAT),
        label="Recognized extensions",
        hint=(
            f"pass format= to name it explicitly, e.g. read({path!r}, format='parquet'). "
            "A directory or extension-less path always needs format=."
        ),
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
