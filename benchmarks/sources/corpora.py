"""File corpora — the benchmarks that measure reading, not querying.

Where :mod:`sources.tables` materializes a benchmark's tables into shared Arrow, the
corpus benchmarks measure the read path *itself*, so nothing here loads data. Each
corpus is a lazily-resolved set of files that a case opens inside its own timed call:

- **Scan (file layout)** — the same 16-column ``int64`` data laid out three ways in the
  Ray bucket (one big file / ~132 MiB files / many small files), isolating scan-planning
  cost from the bytes read. See :class:`ScanCorpus`.
- **Images** — a bounded slice of the 211,742-JPEG profile-picture corpus, for the
  multimodal read/decode benchmark. See :class:`ImageCorpus`.

Both expose the same two forms so every engine reads the identical file set: ``glob``
for readers that expand a pattern themselves, and ``open()`` for those (PyArrow, Ray
Data) whose readers take an explicit path list. Bases are overridable via
``BENCH_SCAN_BASE`` / ``BENCH_IMAGES_BASE`` or the ``--source`` CLI flag.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pyarrow.fs as pafs

# --------------------------------------------------------------------------- #
# Scan corpus — one logical table, three physical file layouts
# --------------------------------------------------------------------------- #
SCAN_BASE = os.environ.get("BENCH_SCAN_BASE", "s3://ray-benchmark-data")

# Benchmark ``--scale`` -> the bucket's size directory, mirroring the TPC-H convention
# that scale 1 is ~1 GiB of data.
SCAN_SIZES: dict[int, str] = {
    1: "1GiB",
    10: "10GiB",
    100: "100GiB",
    1000: "1TiB",
    10000: "10TiB",
}

# The three layouts, in increasing file count. Every layout holds the *same* schema
# (``column0..column15``, all ``int64``, uniformly random over ``[0, 2^63)``), so the
# only variable across them is how the bytes are split into files.
SCAN_LAYOUTS = ("one_big", "ideal", "many_small")

_SCAN_DIRS = {
    "one_big": "{base}/parquet/{size}",  # ~1 GiB files (8.4M rows, 8 row groups each)
    "ideal": "{base}/parquet/128MiB-file/{size}",  # ~132 MiB files — the recommended size
    "many_small": "{base}/small-parquet/{size}",  # ~1.2 MiB files (8,192 rows each)
}

# `small-parquet/{1GiB,100GiB}` mixes a few ~133 MiB files (named `75_*`) in among the
# 1.2 MiB ones (`80_*`). Reading them would make the "many small files" layout not
# actually many-small, and at 1GiB it also breaks row-count parity with the other two
# layouts (9,445,376 rows rather than 8,388,608). Restrict those two dirs to `80_*`.
# The other sizes are homogeneous and need no filter.
_SCAN_FILE_PREFIX = {("many_small", "1GiB"): "80_", ("many_small", "100GiB"): "80_"}


def _list_corpus(
    directory: str, suffix: str, file_prefix: str
) -> tuple[pafs.FileSystem, list[str]]:
    """List ``directory`` now, returning its filesystem and sorted member paths.

    Filters to files ending in ``suffix`` whose basename starts with ``file_prefix``. The
    listing is on demand (not at corpus construction) so the corpus benchmarks time it as
    part of the workload.

    When a ``file_prefix`` selects a subset of a large directory, the listing is
    **prefix-scoped** through fsspec (one object-store LIST of the matching keys) rather
    than paging the whole directory — so the non-Batcher engines, which read this file
    list, are timed against the same efficient listing Batcher's own reader now does
    (a fair comparison, not a handicap from the harness's listing code). Falls back to the
    full pyarrow directory listing when fsspec / its backend is absent.
    """
    filesystem, base = pafs.FileSystem.from_uri(directory)
    scoped = _prefix_scoped_paths(directory, base, suffix, file_prefix)
    if scoped is not None:
        return filesystem, scoped
    entries = filesystem.get_file_info(pafs.FileSelector(base, recursive=False))
    paths = [
        entry.path
        for entry in entries
        if entry.path.endswith(suffix) and entry.base_name.startswith(file_prefix)
    ]
    return filesystem, sorted(paths)


def _prefix_scoped_paths(
    directory: str, base: str, suffix: str, file_prefix: str
) -> list[str] | None:
    """Filesystem-native paths under ``base`` matching ``file_prefix``, via a prefix LIST.

    Uses fsspec's prefix-scoped glob (so a subset of a huge bucket lists in one call) and
    maps the results back to the filesystem-native form pyarrow readers expect. Returns
    ``None`` (fall back to a full listing) for a local path, a missing fsspec backend, or
    any listing error — the fast path is purely an optimization, never the correctness path.
    """
    if "://" not in directory or file_prefix == "":
        return None
    try:
        import fsspec

        backend = fsspec.filesystem(directory.split("://", 1)[0])
        matches = backend.glob(f"{base}/{file_prefix}*{suffix}")
    except Exception:
        return None
    paths = sorted(m for m in matches if m.endswith(suffix))
    return paths or None


@dataclass(frozen=True)
class ScanCorpus:
    """One (layout, size) corpus of parquet files, resolved lazily.

    Exposes the two forms the engines need. ``glob`` is for the engines that expand a
    glob themselves (every SQL engine); ``open()`` performs the listing explicitly, for
    PyArrow and Ray Data, whose readers take a file list rather than a pattern.
    """

    layout: str
    size: str
    directory: str
    file_prefix: str = ""

    @property
    def glob(self) -> str:
        """The ``*.parquet`` pattern covering this corpus's files."""
        return f"{self.directory}/{self.file_prefix}*.parquet"

    def open(self) -> tuple[pafs.FileSystem, list[str]]:
        """List the corpus now, returning its filesystem and sorted member paths."""
        return _list_corpus(self.directory, ".parquet", self.file_prefix)


def scan_corpora(scale: float, source: str | None = None) -> dict[str, ScanCorpus]:
    """The three file layouts at ``scale``, keyed by layout name.

    ``scale`` selects the size directory via :data:`SCAN_SIZES` (1 -> ``1GiB``,
    10 -> ``10GiB``, ...); ``source`` overrides the bucket base.
    """
    size = SCAN_SIZES.get(int(scale))
    if size is None:
        valid = ", ".join(str(s) for s in SCAN_SIZES)
        raise ValueError(f"scan scale must be one of {valid} (got {scale})")
    base = source or SCAN_BASE
    return {
        layout: ScanCorpus(
            layout=layout,
            size=size,
            directory=_SCAN_DIRS[layout].format(base=base, size=size),
            file_prefix=_SCAN_FILE_PREFIX.get((layout, size), ""),
        )
        for layout in SCAN_LAYOUTS
    }


# --------------------------------------------------------------------------- #
# Image corpus — an unstructured (JPEG) multimodal read/decode workload
# --------------------------------------------------------------------------- #
IMAGES_BASE = os.environ.get("BENCH_IMAGES_BASE", "s3://ray-benchmark-data/profile-pictures/1GiB")

# Benchmark ``--scale`` -> a filename prefix selecting how many images to read. The
# corpus is 211,742 JPEGs named ``000000.jpg``..``211741.jpg`` (all 110x110 RGB, ~5 KiB),
# so a zero-padded prefix bounds the count by a power of ten. The default is deliberately
# small: image reads over S3 are per-file (one object open each), so thousands of tiny
# files is minutes of wall-clock even before any engine's overhead (see the README note).
IMAGE_COUNTS: dict[int, tuple[str, int]] = {
    1: ("00000", 10),
    10: ("0000", 100),
    100: ("000", 1_000),
    1000: ("00", 10_000),
    10000: ("0", 100_000),
}
IMAGE_SUFFIX = ".jpg"
# The corpus's native pixel dimensions (H, W). The decode benchmark targets these so all
# engines produce identically-sized tensors — and because Batcher's ``read.images`` reader
# requires an explicit decode size (it has no native-resolution decode mode), a fact this
# makes the decode shape express uniformly rather than special-case.
IMAGE_NATIVE_SIZE = (110, 110)


@dataclass(frozen=True)
class ImageCorpus:
    """A bounded set of JPEG files for the multimodal read/decode benchmark.

    Like :class:`ScanCorpus`, exposes ``glob`` (for readers that expand a pattern, e.g.
    Batcher's ``read.images``) and ``open()`` (an explicit path list, for Ray Data / Daft
    / PyArrow, whose image/binary readers take a list). Resolution is lazy so that file
    listing is timed as part of the workload.
    """

    directory: str
    file_prefix: str
    count: int
    native_size: tuple[int, int] = IMAGE_NATIVE_SIZE
    suffix: str = IMAGE_SUFFIX

    @property
    def glob(self) -> str:
        """The pattern covering this corpus's image files."""
        return f"{self.directory}/{self.file_prefix}*{self.suffix}"

    def open(self) -> tuple[pafs.FileSystem, list[str]]:
        """List the corpus now, returning its filesystem and sorted member paths.

        The paths are filesystem-native (bucket-relative for S3, absolute for local) —
        the form Ray Data and PyArrow want alongside the filesystem object.
        """
        return _list_corpus(self.directory, self.suffix, self.file_prefix)

    def uris(self) -> list[str]:
        """Scheme-qualified URIs (``s3://...`` / ``file://...``) — the form Daft downloads.

        Derived from the corpus directory's scheme so a Daft ``url.download`` gets a URI
        it can resolve, whether the base is an S3 bucket or a local directory.
        """
        _, paths = self.open()
        if "://" in self.directory:
            scheme = self.directory.split("://", 1)[0]
            return [f"{scheme}://{p}" for p in paths]
        return [f"file://{p}" for p in paths]


def image_corpus(scale: float, source: str | None = None) -> ImageCorpus:
    """The image corpus at ``scale`` (1 -> 10 files, 10 -> 100, ... via :data:`IMAGE_COUNTS`)."""
    entry = IMAGE_COUNTS.get(int(scale))
    if entry is None:
        valid = ", ".join(str(s) for s in IMAGE_COUNTS)
        raise ValueError(f"images scale must be one of {valid} (got {scale})")
    prefix, count = entry
    return ImageCorpus(directory=source or IMAGES_BASE, file_prefix=prefix, count=count)
