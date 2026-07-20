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

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, ClassVar

import pyarrow as pa

from batcher._internal.errors import IOError as BatcherIOError
from batcher._internal.hardware import available_cpu_count
from batcher.io.base._tolerance import ErrorPolicy
from batcher.io.filesystem import resolve_filesystem
from batcher.io.formats.multimodal._batching import pack_by_count_and_bytes, probe_sizes
from batcher.io.formats.multimodal._mime import sniff_mime
from batcher.io.formats.multimodal._pruning import prune_files
from batcher.plan.source_stats import SourceStatistics
from batcher.plan.stats import ColumnStat, Provenance

__all__ = ["MediaSource", "MediaSplit"]

# In reference mode (no full-byte materialization) we still read a header chunk so
# MIME sniffing and header-only metadata work; large enough for image/audio/video
# container headers, tiny next to a multi-GB payload.
_HEADER_BYTES = 1 << 16  # 64 KiB
# Default byte ceiling on one file-batch. Media file sizes span orders of magnitude, so a
# batch bounded only by file count is unbounded in memory; 256 MiB keeps a batch inside a
# worker's envelope while staying large enough that per-batch overhead stays negligible.
_DEFAULT_BATCH_BYTES = 256 << 20
# Common columns every media source emits, in order.
_COMMON_FIELDS: tuple[tuple[str, pa.DataType], ...] = (
    ("uri", pa.string()),
    # `large_binary` (64-bit offsets), not `binary` (32-bit): this column exists to hold
    # large per-row payloads, and a batch of them overflows a 32-bit offset array at 2 GB
    # *total*. At the default 64 files per batch that is one 32 MB average file — well
    # inside ordinary video and point-cloud sizes — and the overflow raises mid-read
    # rather than being caught at plan time. `read_blob_bytes` already got this right.
    ("bytes", pa.large_binary()),
    ("size", pa.int64()),
    ("mime", pa.string()),
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

    __slots__ = (
        "_batch_bytes",
        "_batch_files",
        "_chunk_cache",
        "_errors",
        "_files_cache",
        "_fs",
        "_materialize_bytes",
        "_path",
        "_with_meta",
    )

    def __init__(
        self,
        path: str,
        *,
        batch_files: int = 64,
        batch_bytes: int = _DEFAULT_BATCH_BYTES,
        with_meta: bool = True,
        materialize_bytes: bool = True,
        on_error: str = "raise",
    ) -> None:
        if batch_files < 1:
            raise ValueError("batch_files must be >= 1")
        if batch_bytes < 1:
            raise ValueError("batch_bytes must be >= 1")
        self._path = path
        self._batch_files = batch_files
        self._batch_bytes = batch_bytes
        self._with_meta = with_meta
        self._materialize_bytes = materialize_bytes
        self._fs = resolve_filesystem(path)
        # A media corpus at scale always holds a few unreadable members — a truncated
        # upload, a zero-byte object, a JPEG whose trailer never arrived. `on_error`
        # decides whether one of them costs the whole read.
        self._errors = ErrorPolicy(on_error)
        self._files_cache: list[str] | None = None
        self._chunk_cache: list[list[str]] | None = None

    # ---- shared, do-not-override ------------------------------------------
    def _files(self) -> list[str]:
        """List every media file under the path (sorted, matching any accepted suffix).

        All accepted extensions are resolved in a *single* listing pass — a directory of
        many files must never be re-listed once per extension (that turned one read into
        one full object-store listing per suffix). A directory legitimately lacking some
        extensions is fine; only an empty overall listing is an error.
        """
        if self._files_cache is None:
            try:
                matches = self._fs.expand(self._path, suffix=self.suffixes)
            except BatcherIOError as exc:
                raise BatcherIOError(
                    f"no {self.format_name} files ({', '.join(self.suffixes)}) under {self._path!r}"
                ) from exc
            self._files_cache = sorted(matches)
        return self._files_cache

    def _chunks(self) -> list[list[str]]:
        """The file-batches this source reads, bounded by **both** count and bytes.

        A fixed file count is the wrong unit for media. A directory mixing 4 KB thumbnails
        with 200 MB videos batched 64-at-a-time yields batches spanning four orders of
        magnitude — the worst case is 64 x 200 MB = 12.8 GB in one batch, which is an OOM
        rather than a slow query, and the sibling batch of thumbnails leaves its worker
        idle. Bounding bytes as well turns that into evenly-weighted work.

        One definition, used by `read`, `iter_batches` *and* `splits`, because a split that
        chunked differently from `iter_batches` would make the distributed result a
        different set of batches from the single-node one.
        """
        if self._chunk_cache is None:
            files = self._files()
            self._chunk_cache = pack_by_count_and_bytes(
                files, self._file_sizes(files), self._batch_files, self._batch_bytes
            )
        return self._chunk_cache

    def _file_sizes(self, files: list[str]) -> list[int]:
        """Every file's size, probed concurrently (see `_batching.probe_sizes`)."""
        return probe_sizes(files, self._fs.size)

    def schema(self) -> pa.Schema:
        """The output schema: common columns plus (if enabled) metadata columns."""
        fields = [pa.field(n, t) for n, t in _COMMON_FIELDS]
        if self._with_meta:
            fields += [pa.field(n, t) for n, t in self._meta_fields()]
        return pa.schema(fields)

    def read(self, projection: list[str] | None = None) -> list[pa.RecordBatch]:
        """Read every file-batch (the bounded-source / `collect()` path).

        `read()` returns all batches, so every payload is resident regardless — which
        lets it fetch *all* files in one wide concurrent wave (a single thread pool) and
        then slice them into `batch_files`-sized batches, rather than spinning a fresh
        pool and blocking on a serial read per file-batch. That removes the per-chunk pool
        churn + cross-chunk serialization that held image/clip ingest well under the raw
        parallel-read rate. `iter_batches` keeps its per-chunk streaming for the
        bounded-memory (larger-than-RAM) consumer, where reading everything up front would
        defeat the point.
        """
        files = self._files()
        chunks = self._chunks()
        if len(chunks) <= 1:
            return list(self.iter_batches(projection))
        reads = self._read_chunk(files)  # one concurrent wave over every file
        out: list[pa.RecordBatch] = []
        start = 0
        for chunk in chunks:
            sl = slice(start, start + len(chunk))
            batch = self._assemble(files[sl], reads[sl])
            out.append(batch.select(projection) if projection is not None else batch)
            start += len(chunk)
        return out

    def iter_batches(self, projection: list[str] | None = None) -> Iterator[pa.RecordBatch]:
        for chunk in self._chunks():
            batch = self._build_batch(chunk)
            yield batch.select(projection) if projection is not None else batch

    def row_count(self) -> int | None:
        """The number of media files — known from listing, without reading data."""
        return len(self._files())

    def identity(self) -> str:
        return f"{self.format_name}:{self._path}"

    def statistics(self) -> SourceStatistics:
        """Row count, total bytes and a `size` zone map — all from the listing.

        A media source used to report nothing, so Kyber planned it blind: it costed a
        directory of 200 MB videos exactly like one of 4 KB thumbnails, and the byte axes
        that gate broadcast eligibility and task sizing had only a generic per-row prior
        (36 B for a binary column) to work from. Every number here comes from the file
        listing and a stat — the same probe the batching already performs — so this costs
        nothing beyond what a read does anyway.

        The `size` bounds are **exact** (they are the values, not per-chunk bounds), which
        is what lets a `WHERE size > …` prune files outright rather than conservatively.

        Returns:
            The statistics, with an exact row count and an exact `size` column stat.
        """
        files = self._files()
        sizes = self._file_sizes(files)
        columns: dict[str, ColumnStat] = {}
        if sizes:
            columns["size"] = ColumnStat(
                min=min(sizes), max=max(sizes), null_count=0, provenance=Provenance.EXACT
            )
        return SourceStatistics(
            row_count=len(files),
            byte_size=sum(sizes) or None,
            columns=columns,
            exact_rows=True,
        )

    def splits(
        self, target_size: int | None = None, predicate: dict | None = None
    ) -> list[MediaSplit]:
        """One split per file-batch; each carries only its file-path locators.

        Args:
            target_size: Rough bytes to aim for per split. Overrides the source's
                `batch_bytes` for this call, which is how the distributed planner asks for
                coarser splits than a streaming read would use. Previously ignored, so a
                split was a fixed *file count* and a video split could be a thousand times
                the weight of a thumbnail split.
            predicate: Kyber's pushed filter, as its IR dictionary. A predicate over the
                columns a listing already knows — ``uri``, ``size``, ``mime`` — prunes
                whole files here, so they never become a split and are never opened. That
                is the difference between reading a directory and reading a terabyte for
                a query like ``WHERE mime = 'video/mp4' AND size < 50000000``. Any other
                predicate is ignored (the engine's `Filter` still applies it).
        """
        files = self._files()
        sizes = self._file_sizes(files)
        pruned = prune_files(files, sizes, predicate)
        if pruned is not None:
            files, sizes = pruned
        bound = target_size if target_size is not None else self._batch_bytes
        if pruned is None and bound == self._batch_bytes:
            chunks = self._chunks()  # the memoized, unpruned chunking
        else:
            chunks = pack_by_count_and_bytes(files, sizes, self._batch_files, bound)
        return [
            MediaSplit(
                self.format_name,
                tuple(chunk),
                self._with_meta,
                self._materialize_bytes,
            )
            for chunk in chunks
        ]

    # ---- batch assembly ---------------------------------------------------
    def _build_batch(self, chunk: list[str]) -> pa.RecordBatch:
        """Assemble one Arrow `RecordBatch` from a chunk of files (no decode).

        In reference mode (`materialize_bytes=False`) the ``bytes`` column is null
        and only the header + size are touched per file — so a chunk of GB videos
        costs kilobytes, not gigabytes.
        """
        return self._assemble(chunk, self._read_chunk(chunk))

    def _assemble(
        self, chunk: list[str], reads: list[tuple[bytes, bytes | None, int]]
    ) -> pa.RecordBatch:
        """Build one `RecordBatch` from files and their already-read payloads.

        Split from the read so `read()` can bulk-fetch every file in one wide concurrent
        wave and then assemble the batches, instead of a fresh thread pool + a serial
        read per file-batch (which left a many-file scan far under the raw parallel-read
        throughput — the ingest floor for a directory of many small images/clips).
        """
        uris: list[str] = []
        blobs: list[bytes | None] = []
        sizes: list[int] = []
        mimes: list[str] = []
        meta_rows: list[dict[str, Any]] = []
        meta_fields = self._meta_fields() if self._with_meta else []
        for path, read in zip(chunk, reads, strict=True):
            if read is None:  # unreadable and tolerated — contributes no row
                continue
            header, payload, size = read
            uris.append(path)
            blobs.append(payload)  # None in reference mode
            sizes.append(size)
            mimes.append(sniff_mime(path, header))
            if self._with_meta:
                meta_rows.append(self._safe_extract(header, meta_fields))
        arrays: list[pa.Array] = [
            pa.array(uris, pa.string()),
            pa.array(blobs, pa.large_binary()),
            pa.array(sizes, pa.int64()),
            pa.array(mimes, pa.string()),
        ]
        names = [n for n, _ in _COMMON_FIELDS]
        for name, dtype in meta_fields:
            arrays.append(pa.array([row.get(name) for row in meta_rows], dtype))
            names.append(name)
        return pa.RecordBatch.from_arrays(arrays, names=names)

    def _read_chunk(self, chunk: list[str]) -> list[tuple[bytes, bytes | None, int]]:
        """Read every file in ``chunk`` concurrently, preserving order.

        Each media file is one object-store round trip; the read releases the GIL, so a
        serial per-file loop leaves a many-file scan latency-bound on a single connection
        (the ingest bottleneck for a directory of many small images/clips). The pool is
        capped so a large chunk does not open an unbounded number of connections at once.
        """
        if len(chunk) <= 1:
            return [self._read_payload_safe(chunk[0])] if chunk else []
        from concurrent.futures import ThreadPoolExecutor

        # Latency-bound tiny-file reads scale with concurrency well past core count; cap
        # at the chunk size so a full chunk reads in one concurrent wave (raw byte reads
        # are thread-safe under fan-out — unlike a footer *parse*, which is not).
        workers = min(len(chunk), max(8, available_cpu_count() * 2), 64)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(self._read_payload_safe, chunk))  # order preserved

    def _read_payload_safe(self, path: str) -> tuple[bytes, bytes | None, int] | None:
        """`_read_payload`, honoring `on_error` — None marks a file to drop."""
        try:
            return self._read_payload(path)
        except Exception as exc:
            if not self._errors.tolerate(path, exc, format_name=self.format_name):
                raise
            return None

    def corrupt_files(self) -> list[str]:
        """The paths this source skipped, in failure order (empty unless `on_error="skip"`).

        Examples:
            .. doctest::

                >>> from batcher.io import ImageSource  # doctest: +SKIP
                >>> src = ImageSource("s3://bucket/imgs/", on_error="skip")  # doctest: +SKIP
                >>> _ = src.read()  # doctest: +SKIP
                >>> src.corrupt_files()  # doctest: +SKIP
                ['s3://bucket/imgs/truncated.jpg']

        Returns:
            The skipped paths. A skipped file leaves no row, so this is the only way to
            tell a clean read from a partial one.
        """
        return self._errors.skipped()

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

    @property
    def rows(self) -> int:
        """This split's exact row count — one row per file, known with no I/O.

        The distributed planner reads `rows` off a split to size its task fan-out and to
        weight the partition balance. Without it a media source looked *uncountable*: the
        fan-out fell back to a blunt worker count and every split weighed the same, so a
        split of 200 MB videos was balanced against one of thumbnails as if equal.
        """
        return len(self.files)

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
