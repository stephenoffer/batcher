"""Format and layout detection for the generic `read(path, format=None)` entry point.

Resolves a format name (a `SOURCES` registry key) from an explicit override, the
path's URI scheme, its file extension, or the files inside a directory. Table/database
sources (delta, iceberg, sql, …) are addressed by their explicit `read_*` functions, not
by extension.

It also answers the *layout* question a path carries, which is a sibling of the format
one and needs the same cheap listing: whether a directory is a Hive tree
(`partition_aware_format`) and which columns it is partitioned by
(`hive_partition_keys`). Both exist because a partitioned directory read or rewritten as
a flat one loses the columns it is organized by, silently.
"""

from __future__ import annotations

import contextlib
import os
from typing import Any

from batcher._internal.errors import FormatError, unknown_value

__all__ = [
    "COMPRESSION_SUFFIXES",
    "DATA_SUFFIXES",
    "compression_for_path",
    "detect_format",
    "format_for_extension",
    "hive_partition_keys",
    "partition_aware_format",
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
    # FASTA and FASTQ each carry several conventional suffixes, and a corpus mixes them
    # freely: `.fa`/`.fasta` for nucleotides, `.faa`/`.fna`/`.ffn` for the amino-acid and
    # nucleotide splits NCBI publishes, `.fq`/`.fastq` for reads. Mapping only the long
    # forms would leave `bt.read("reads.fq")` guessing.
    ".fasta": "fasta",
    ".fa": "fasta",
    ".faa": "fasta",
    ".fna": "fasta",
    ".ffn": "fasta",
    ".fastq": "fastq",
    ".fq": "fastq",
    # Intervals, annotations, and variants. GFF3 and GTF share a reader — they differ
    # only in how the ninth column encodes its attributes, which this engine keeps as
    # text rather than guessing the dialect from the extension.
    ".bed": "bed",
    ".bedgraph": "bed",
    ".gff": "gff",
    ".gff3": "gff",
    ".gtf": "gff",
    ".vcf": "vcf",
    ".log": "logs",
    ".pb": "protobuf",
    ".msgpack": "msgpack",
    ".mp": "msgpack",
    ".txt": "text",
    ".text": "text",
    ".pdf": "documents",
    # `.warc.gz` resolves here too: `_ext` strips the compression suffix first.
    ".warc": "warc",
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

    Read off the **last path segment**, after dropping any URI query string. Truncating the
    path at the first ``*`` instead threw the extension away with it, so *every* glob failed
    detection — ``read("data/*.parquet")`` raised `FormatError` even though it is the
    documented spelling for reading many files, and so did
    ``s3://bucket/*.parquet?endpoint_override=…``, which is how an on-prem S3 is addressed.
    The last segment is the one that carries the extension; a wildcard inside it
    (``part-*.parquet``) leaves the suffix perfectly readable, and a wildcard *directory*
    (``data/2024-*/``) has no extension either way and falls through as before.
    """
    base = path.split("?", 1)[0].rstrip("/")
    segment = base.rsplit("/", 1)[-1]
    stem, ext = os.path.splitext(segment)
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
    return _training_shards_at(root)


def _training_shards_at(root: str) -> str | None:
    """``"training_shards"`` if `root` is a corpus written by `write_shards`, else None.

    Checked last, and on *two* signals rather than one. The manifest is called
    ``index.json``, which is a name an unrelated directory can plausibly carry, so a
    directory only claims the format when it also holds a shard the manifest would be
    describing. A single generic marker here would hijack a plain directory read.
    """
    if not os.path.isfile(os.path.join(root, "index.json")):
        return None
    with contextlib.suppress(OSError):
        if any(name.startswith("shard-") and name.endswith(".arrow") for name in os.listdir(root)):
            return "training_shards"
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
    suffix stripped so ``events.csv.gz`` resolves as CSV → the extensions of the files
    *inside* a directory, which is how a sharded or partitioned output written earlier
    reads back without naming a format nobody chose. A directory holding two data formats
    is not one relation, so it declines rather than picking one. Raises `FormatError` if
    the format cannot be inferred, naming what it *could* have inferred.

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
    # A directory has no extension of its own, but the files in it do. Reading back a
    # directory Batcher itself wrote (`write.parquet(dir, partition_by=…)` produces one)
    # used to demand `format="parquet"` for a layout the writer chose, which is the least
    # guessable argument in the API — the caller never named a format on the way in.
    inferred = _format_from_directory(path)
    if inferred is not None:
        return inferred
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
            "A directory of files with no recognized extension always needs format=."
        ),
    )


def _format_from_directory(path: str) -> str | None:
    """The format of the data files under directory `path`, or None if it is not one.

    One listing, and only where the alternative is raising. A directory holding more than
    one data format is *not* one relation, so a mixed listing declines rather than picking
    the first — the caller either meant a glob or has to name the format.
    """
    from batcher.io.filesystem import resolve_filesystem

    try:
        files = resolve_filesystem(path).expand(path, suffix=DATA_SUFFIXES)
    except Exception:
        return None
    formats = {fmt for f in files if (fmt := _EXT_TO_FORMAT.get(_ext(f))) is not None}
    return formats.pop() if len(formats) == 1 else None


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


#: Reader options the Hive-aware Parquet source takes. Deliberately an allowlist rather
#: than a list of the ones it rejects: the flat reader's keyword surface is open-ended
#: (`**opts` forwarded to the format), so a denylist would silently go stale and the
#: upgrade would start raising `TypeError` on a keyword that used to work. A caller who
#: passed anything else keeps the reader they asked for — trading a missing column for a
#: silently ignored `on_error=` is the worse of the two failures, because nothing warns.
_PARTITIONED_OPTIONS: frozenset[str] = frozenset({"partitioning", "schema_mode"})

_GLOB_CHARS = ("*", "?", "[")


def partition_aware_format(path: Any, fmt: str, opts: dict[str, Any] | None = None) -> str:
    """`fmt`, upgraded to its partition-aware reader when `path` is a Hive tree.

    ``write.parquet(dir, partition_by=["g"])`` stores `g` in the directory *names*, not in
    the files, so reading that directory back with the flat reader returns every row minus
    the column the data is organized by — a lossy round trip through Batcher's own writer,
    reported only as a warning. Every engine users arrive from (Spark, DuckDB, Polars,
    pandas) recovers those columns, so this routes the read to `ParquetDatasetSource`,
    which already does exactly that and prunes partitions besides.

    Deliberately narrow. It upgrades only when the path is a plain directory whose
    immediate children are ``col=value`` directories, only for Parquet (the one format with
    a partition-aware reader), and only when every option the caller passed is one the
    partitioned source takes. Anything the partitioned reader then refuses to open falls
    back to the flat reader, so this can add a recovered column but never take a working
    read away.

    Args:
        path: The path about to be read.
        fmt: The format name resolved so far.
        opts: The reader options the caller passed.

    Returns:
        The format name to construct the source with.
    """
    if fmt != "parquet" or not _PARTITIONED_OPTIONS.issuperset(opts or {}):
        return fmt
    if not isinstance(path, str) or any(c in path for c in _GLOB_CHARS):
        return fmt
    if "?" in path:
        # A query string carries connection config (`?endpoint_override=…` for an on-prem
        # S3) that `resolve_filesystem` understands and the partitioned source — which
        # builds its own dataset straight from the path — does not. Upgrading here would
        # turn a working read of a MinIO/Ceph bucket into a connection failure.
        return fmt
    if not _has_hive_children(path):
        return fmt
    try:  # the partitioned reader must be able to open it, or nothing changes
        from batcher.io.formats.base import SOURCES

        SOURCES.get("parquet_dataset")(path).schema()
    except Exception:
        return fmt
    return "parquet_dataset"


def hive_partition_keys(path: str) -> list[str]:
    """The Hive partition column names of the tree at `path`, outermost first.

    Reads the layout off the directory names rather than off any file, because that is
    where a Hive partition column lives. Used by anything that must *preserve* an existing
    layout while rewriting the data under it: a compaction that read a partitioned tree and
    wrote it back flat would leave the rows intact and the organization destroyed, and the
    next partition-pruned query would read the whole table.

    Descends one branch, which is enough: every branch of a Hive tree carries the same keys
    in the same order, and walking them all would be an O(partitions) listing on the driver
    to learn a fact the first branch already answers.

    Args:
        path: The dataset root.

    Returns:
        The partition column names, outermost first, or an empty list if `path` is not a
        partitioned tree.

    Examples:
        .. doctest::

            >>> from batcher.io.detect import hive_partition_keys
            >>> hive_partition_keys("/nonexistent")
            []
    """
    from batcher.io.base._paths import hive_segment
    from batcher.io.filesystem import resolve_filesystem

    keys: list[str] = []
    try:
        fs = resolve_filesystem(path)
        current = path
        while dirs := fs.list_dirs(current):
            segments = [hive_segment(d) for d in dirs]
            if not all(segments) or len({s[0] for s in segments if s}) != 1:
                break
            keys.append(segments[0][0])  # type: ignore[index]
            current = dirs[0]
    except Exception:
        return keys
    return keys


def _has_hive_children(path: str) -> bool:
    """Whether `path` is a directory whose immediate children are all ``col=value`` dirs.

    One *non-recursive* listing, deliberately: it is the cheapest question that separates
    a Hive root from a flat directory of part files, and it stays cheap on a tree holding
    a million files, where a recursive listing on the driver is the thing that must not
    happen just to pick a reader. A flat directory has no subdirectories at all, so it
    answers False without a second thought.

    Every child must name the *same* partition column. A tree showing ``dt=x`` beside
    ``g=a`` is not partitioned by either — it is a rewrite that changed partitioning and
    left the old directories behind — and handing that to a partition-aware reader gets a
    schema conflict where the flat reader would at least return the rows.
    """
    return bool(hive_partition_keys(path))
