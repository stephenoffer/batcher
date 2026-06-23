"""Multimodal media source base — list files, never decode pixels/frames.

A `MediaSource` turns a directory/glob of media files (images, audio, video, …)
into a relation whose rows are *references* to the files plus cheap, header-only
metadata. Every media source emits the same common columns —
``uri:string, bytes:binary, size:int64, mime:string`` — and each concrete
subclass adds a handful of format-specific metadata columns (an image's
width/height, an audio file's sample rate, …) extracted from the file *header*
only. No pixel buffer, audio sample array, or video frame is ever decoded at read
time; that work belongs downstream in a Rust operator over the ``bytes`` column.

The unit of work is a *file-batch*: ``batch_files`` files are assembled into one
Arrow `RecordBatch`, so the Python control plane never touches a row — it builds
whole batches. Splits are one `MediaSplit` per file-batch, each carrying only the
list of file paths (picklable locators), so a distributed read fans file-batches
out to workers that read their own files directly from storage.

Concrete sources live one-per-file alongside this module (`images.py`,
`audio.py`, `video.py`, `embeddings.py`) and register into the shared `SOURCES`
registry; a new media kind is one new file overriding `_meta_fields` /
`_extract_meta`.
"""

from __future__ import annotations

import mimetypes
from collections.abc import Iterator
from dataclasses import dataclass
from typing import IO, Any, ClassVar

import pyarrow as pa

from batcher._internal.errors import IOError as BatcherIOError
from batcher.io.filesystem import resolve_filesystem

__all__ = ["MediaSource", "MediaSplit", "read_blob_bytes"]

# How many leading bytes are enough to sniff a media type by magic number and to
# read a format header. Kept small so metadata extraction stays header-only.
_MAGIC_PEEK_BYTES = 4096

# In reference mode (no full-byte materialization) we still read a header chunk so
# MIME sniffing and header-only metadata work; large enough for image/audio/video
# container headers, tiny next to a multi-GB payload.
_HEADER_BYTES = 1 << 16  # 64 KiB

# Common columns every media source emits, in order.
_COMMON_FIELDS: tuple[tuple[str, pa.DataType], ...] = (
    ("uri", pa.string()),
    ("bytes", pa.binary()),
    ("size", pa.int64()),
    ("mime", pa.string()),
)

# Magic-number prefixes for media types stdlib `mimetypes` may miss by extension.
_MAGIC_PREFIXES: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"BM", "image/bmp"),
    (b"RIFF", "image/webp"),  # disambiguated below by the WEBP/WAVE tag at [8:12]
    (b"OggS", "audio/ogg"),
    (b"fLaC", "audio/flac"),
    (b"\x1aE\xdf\xa3", "video/x-matroska"),
)


class MediaSource:
    """Base for a media source that lists files and emits references + header meta.

    Subclasses set `suffixes` (extensions used to list files) and `format_name`
    (the registry key used to rebuild a split on a worker), and override
    `_meta_fields` and `_extract_meta`. The base owns filesystem resolution, file
    listing, file-batch assembly, MIME sniffing, splits, and row counting.

    Construction is uniform across every media kind::

        ImageSource("s3://bucket/imgs/", batch_files=128, with_meta=True)

    `batch_files` controls how many files become one Arrow `RecordBatch` (and one
    split); `with_meta` toggles the format-specific metadata columns (set it
    False to skip even header reads when only the raw bytes are needed).

    `materialize_bytes` controls whether the full file payload is loaded into the
    ``bytes`` column. The default `True` reads every file whole (fine for small
    media). Set it `False` for **reference mode**: only the file header (for MIME +
    metadata) and the size (a stat, not a read) are touched, and ``bytes`` is left
    null. This is what keeps GB-per-row media (video/audio) from OOM-ing — the
    relation is a table of lightweight handles, so a query can filter/sample on
    ``size``/``mime``/dimensions *before* any payload is read; materialize the
    bytes for the rows that survive with `read_blob_bytes`.
    """

    suffixes: ClassVar[tuple[str, ...]] = ()
    format_name: ClassVar[str] = ""

    __slots__ = ("_batch_files", "_files_cache", "_fs", "_materialize_bytes", "_path", "_with_meta")

    def __init__(
        self,
        path: str,
        *,
        batch_files: int = 64,
        with_meta: bool = True,
        materialize_bytes: bool = True,
    ) -> None:
        if batch_files < 1:
            raise ValueError("batch_files must be >= 1")
        self._path = path
        self._batch_files = batch_files
        self._with_meta = with_meta
        self._materialize_bytes = materialize_bytes
        self._fs = resolve_filesystem(path)
        self._files_cache: list[str] | None = None

    # ---- shared, do-not-override ------------------------------------------
    def _files(self) -> list[str]:
        """List every media file under the path (sorted, deduped across suffixes).

        Each suffix is expanded independently; a suffix that matches nothing is
        skipped (a directory of mixed media legitimately lacks some extensions).
        An empty overall listing is an error.
        """
        if self._files_cache is None:
            seen: dict[str, None] = {}
            for suffix in self.suffixes:
                try:
                    matches = self._fs.expand(self._path, suffix=suffix)
                except BatcherIOError:
                    continue  # this extension matched no files; try the next.
                for f in matches:
                    seen.setdefault(f, None)
            if not seen:
                raise BatcherIOError(
                    f"no {self.format_name} files ({', '.join(self.suffixes)}) under {self._path!r}"
                )
            self._files_cache = sorted(seen)
        return self._files_cache

    def schema(self) -> pa.Schema:
        """The output schema: common columns plus (if enabled) metadata columns."""
        fields = [pa.field(n, t) for n, t in _COMMON_FIELDS]
        if self._with_meta:
            fields += [pa.field(n, t) for n, t in self._meta_fields()]
        return pa.schema(fields)

    def read(self, projection: list[str] | None = None) -> list[pa.RecordBatch]:
        return list(self.iter_batches(projection))

    def iter_batches(self, projection: list[str] | None = None) -> Iterator[pa.RecordBatch]:
        files = self._files()
        for start in range(0, len(files), self._batch_files):
            chunk = files[start : start + self._batch_files]
            batch = self._build_batch(chunk)
            yield batch.select(projection) if projection is not None else batch

    def row_count(self) -> int | None:
        """The number of media files — known from listing, without reading data."""
        return len(self._files())

    def identity(self) -> str:
        return f"{self.format_name}:{self._path}"

    def splits(self, target_size: int | None = None) -> list[MediaSplit]:  # noqa: ARG002
        """One split per file-batch; each carries only its file-path locators."""
        files = self._files()
        return [
            MediaSplit(
                self.format_name,
                tuple(files[s : s + self._batch_files]),
                self._with_meta,
                self._materialize_bytes,
            )
            for s in range(0, len(files), self._batch_files)
        ]

    # ---- batch assembly ---------------------------------------------------
    def _build_batch(self, chunk: list[str]) -> pa.RecordBatch:
        """Assemble one Arrow `RecordBatch` from a chunk of files (no decode).

        In reference mode (`materialize_bytes=False`) the ``bytes`` column is null
        and only the header + size are touched per file — so a chunk of GB videos
        costs kilobytes, not gigabytes.
        """
        uris: list[str] = []
        blobs: list[bytes | None] = []
        sizes: list[int] = []
        mimes: list[str] = []
        meta_rows: list[dict[str, Any]] = []
        meta_fields = self._meta_fields() if self._with_meta else []
        for path in chunk:
            header, payload, size = self._read_payload(path)
            uris.append(path)
            blobs.append(payload)  # None in reference mode
            sizes.append(size)
            mimes.append(_sniff_mime(path, header))
            if self._with_meta:
                meta_rows.append(self._safe_extract(header, meta_fields))
        arrays: list[pa.Array] = [
            pa.array(uris, pa.string()),
            pa.array(blobs, pa.binary()),
            pa.array(sizes, pa.int64()),
            pa.array(mimes, pa.string()),
        ]
        names = [n for n, _ in _COMMON_FIELDS]
        for name, dtype in meta_fields:
            arrays.append(pa.array([row.get(name) for row in meta_rows], dtype))
            names.append(name)
        return pa.RecordBatch.from_arrays(arrays, names=names)

    def _read_payload(self, path: str) -> tuple[bytes, bytes | None, int]:
        """Return ``(header_bytes, payload_or_None, size)`` for one file.

        Full mode reads the whole file (header == payload, size == len). Reference
        mode reads only a header chunk (for MIME + metadata) and stats the size,
        leaving the payload `None` — so no GB payload is ever resident.
        """
        if self._materialize_bytes:
            with self._fs.open(path) as fh:
                data = fh.read()
            return data, data, len(data)
        with self._fs.open(path) as fh:
            header = fh.read(_HEADER_BYTES)
        return header, None, self._fs.size(path)

    def _safe_extract(
        self, data: bytes, meta_fields: list[tuple[str, pa.DataType]]
    ) -> dict[str, Any]:
        """Extract header metadata, tolerating an unreadable/corrupt header.

        A file whose header cannot be parsed yields nulls for its metadata
        columns rather than failing the whole batch — a partial-listing read must
        not be derailed by one bad file.
        """
        try:
            return self._extract_meta(data)
        except Exception:  # header parse errors are per-file, non-fatal
            return dict.fromkeys((n for n, _ in meta_fields))

    # ---- override points --------------------------------------------------
    def _meta_fields(self) -> list[tuple[str, pa.DataType]]:
        """The format-specific metadata columns this source adds (name, type)."""
        return []

    def _extract_meta(self, data: bytes) -> dict[str, Any]:  # noqa: ARG002
        """Extract header-only metadata from a file's raw bytes (no full decode).

        Implementations MUST read only the file header (e.g. an image's
        dimensions, an audio file's sample rate) — never decode the full payload.
        Returns a dict keyed by the names in `_meta_fields`.
        """
        return {}


@dataclass(frozen=True, slots=True)
class MediaSplit:
    """One file-batch of a media source, reconstructed on a worker via `SOURCES`.

    Carries only ``(format_name, files, with_meta)`` — a tuple of file-path
    locators, never data — so it pickles cheaply to a remote worker that then
    reads just its files directly from storage. Mirrors the `Split` read surface
    so a worker treats a split exactly like a source.
    """

    format_name: str
    files: tuple[str, ...]
    with_meta: bool
    materialize_bytes: bool = True

    def _source(self) -> MediaSource:
        """Rebuild a source restricted to this split's files (no re-listing)."""
        from batcher.io.formats.base import SOURCES

        cls = SOURCES.get(self.format_name)
        # Reuse the source's batch assembly but pin its file list to this split's
        # files; batch_files is set so the whole split assembles as one batch.
        src: MediaSource = cls(
            self.files[0],
            batch_files=len(self.files),
            with_meta=self.with_meta,
            materialize_bytes=self.materialize_bytes,
        )
        src._files_cache = list(self.files)
        return src

    def schema(self) -> pa.Schema:
        return self._source().schema()

    def read(self, projection: list[str] | None = None) -> list[pa.RecordBatch]:
        return self._source().read(projection)

    def iter_batches(self, projection: list[str] | None = None) -> Iterator[pa.RecordBatch]:
        yield from self._source().iter_batches(projection)

    def row_count(self) -> int | None:
        return len(self.files)

    def identity(self) -> str:
        return f"{self.format_name}:{self.files[0]}+{len(self.files)}"


def read_blob_bytes(
    batch: pa.RecordBatch, *, uri_col: str = "uri", into: str = "bytes"
) -> pa.RecordBatch:
    """Materialize file payloads for a batch of reference handles.

    Reads each row's ``uri_col`` file and writes its bytes into the ``into``
    column (replacing it if present, else appending). Intended to run inside
    `map_batches` *after* filtering/sampling reference-mode handles, so only the
    surviving rows' payloads are ever read — and with a small `batch_size`, only a
    few payloads are resident at once::

        ds = bt.read.video("s3://clips/", materialize_bytes=False)
        big = ds.filter(col("size") < 500_000_000)        # prune on metadata first
        decoded = big.map_batches(read_blob_bytes, batch_size=4)

    Bounds memory by `batch_size`; the GB payloads never all co-reside.
    """
    uris = batch.column(uri_col).to_pylist()
    blobs: list[bytes | None] = []
    for u in uris:
        if u is None:
            blobs.append(None)
            continue
        with resolve_filesystem(u).open(u) as fh:
            blobs.append(fh.read())
    # `large_binary` (64-bit offsets) so a batch of GB payloads can't overflow the
    # 2 GB limit of 32-bit `binary` — the whole point is large per-row payloads.
    arr = pa.array(blobs, pa.large_binary())
    if into in batch.schema.names:
        return batch.set_column(batch.schema.get_field_index(into), into, arr)
    return batch.append_column(into, arr)


def _sniff_mime(path: str, data: bytes) -> str:
    """Best-effort MIME type from magic bytes, falling back to the extension.

    Magic bytes win (a file is what its bytes say it is); the stdlib
    `mimetypes` extension guess is the fallback, and an unknown type is
    ``application/octet-stream``.
    """
    head = data[:_MAGIC_PEEK_BYTES]
    for prefix, mime in _MAGIC_PREFIXES:
        if head.startswith(prefix):
            if prefix == b"RIFF":
                tag = head[8:12]
                if tag == b"WEBP":
                    return "image/webp"
                if tag == b"WAVE":
                    return "audio/x-wav"
                if tag == b"AVI ":
                    return "video/x-msvideo"
                continue
            return mime
    guessed, _ = mimetypes.guess_type(path)
    return guessed or "application/octet-stream"


def _read_header(fh: IO[Any]) -> bytes:  # pragma: no cover - convenience for subclasses
    """Read just the leading header bytes from an open handle."""
    return fh.read(_MAGIC_PEEK_BYTES)
