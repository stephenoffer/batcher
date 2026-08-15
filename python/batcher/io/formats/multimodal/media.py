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
from typing import Any, ClassVar

import pyarrow as pa

from batcher._internal.errors import IOError as BatcherIOError
from batcher.io.base._tolerance import ErrorPolicy
from batcher.io.filesystem import resolve_filesystem
from batcher.io.formats.mime import sniff_mime
from batcher.io.formats.multimodal._batching import pack_by_count_and_bytes, probe_sizes
from batcher.io.formats.multimodal._pruning import prune_files
from batcher.io.formats.multimodal._split import MediaSplit
from batcher.plan.source_stats import SourceStatistics

__all__ = ["MediaSource", "MediaSplit"]

#: One file's read: header bytes, payload (None in reference mode), size, parsed header
#: metadata (None when the source declares no metadata columns). `None` for the whole
#: tuple means the file was unreadable and the error was tolerated.
_Read = tuple[bytes, "bytes | None", int, "dict[str, Any] | None"] | None

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
        "_size_cache",
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
        self._size_cache: dict[str, int] = {}

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
        """Every file's size — from a completed read where possible, else stat-probed.

        A stat is a full object-store round trip per file, and on a corpus of many small
        files that is not the negligible cost it looks like next to the payload read: it is
        a second round trip against the same latency. Measured on 100 S3 JPEGs it was 86 ms
        of a 272 ms read, a third of the whole thing, spent asking for a number the payload
        read returns anyway.

        So `read()` fetches first and fills `_size_cache` from what came back; this then
        answers from it. `iter_batches` still probes, and must: it bounds memory *before*
        fetching, which is the entire point of the byte bound on the streaming path.
        """
        if all(f in self._size_cache for f in files):
            return [self._size_cache[f] for f in files]
        return probe_sizes(files, self._fs.size)

    def schema(self) -> pa.Schema:
        """The output schema: common columns plus (if enabled) metadata columns."""
        fields = [pa.field(n, t) for n, t in _COMMON_FIELDS]
        if self._with_meta:
            fields += [pa.field(n, t) for n, t in self._meta_fields()]
        return pa.schema(fields)

    def read(self, projection: list[str] | None = None) -> list[pa.RecordBatch]:
        """Read every file-batch (the bounded-source / `collect()` path).

        `read()` returns all batches at once, so it fetches *all* files in one wide
        concurrent wave (a single thread pool) and slices them into `batch_files`-sized
        batches — removing the per-chunk pool churn + cross-chunk serialization of a fresh
        pool per file-batch that held image/clip ingest under the raw parallel-read rate.
        The fetch happens *before* chunking so the sizes come from it rather than from a
        stat per file (see `_file_sizes`). `iter_batches` keeps its per-chunk streaming,
        and its stat probe, for the larger-than-RAM consumer.
        """
        files = self._files()
        # one concurrent wave over every file, header-only when `bytes` is projected away
        reads = self._read_chunk(
            files, self._effective_materialize(projection), self._effective_meta(projection)
        )
        # Every read reports its file's size, so record them before chunking: `_chunks`
        # then needs no stat round trip. Chunk *boundaries* are unchanged — this only
        # changes where the sizes come from, so `read`, `iter_batches` and `splits` still
        # share the one definition.
        self._size_cache.update(
            {f: r[2] for f, r in zip(files, reads, strict=True) if r is not None}
        )
        chunks = self._chunks()
        out: list[pa.RecordBatch] = []
        start = 0
        for chunk in chunks:
            sl = slice(start, start + len(chunk))
            batch = self._assemble(files[sl], reads[sl])
            out.append(batch.select(projection) if projection is not None else batch)
            start += len(chunk)
        return out

    def iter_batches(self, projection: list[str] | None = None) -> Iterator[pa.RecordBatch]:
        materialize = self._effective_materialize(projection)
        want_meta = self._effective_meta(projection)
        for chunk in self._chunks():
            batch = self._build_batch(chunk, materialize, want_meta)
            yield batch.select(projection) if projection is not None else batch

    def _effective_materialize(self, projection: list[str] | None) -> bool:
        """Whether the ``bytes`` payload must actually be read for this projection.

        A projection that drops ``bytes`` (``select("uri", "width")``) needs no payload, so
        the read collapses to header-only — otherwise every payload is fetched then thrown
        away by the ``.select``, making a metadata query over GB videos a full download.
        """
        return self._materialize_bytes and (projection is None or "bytes" in projection)

    def _effective_meta(self, projection: list[str] | None) -> bool:
        """Whether the header-metadata columns are worth parsing for this projection.

        The mirror of `_effective_materialize`, and it was the missing half. Extraction is
        a **Python** parse per file — Pillow, soundfile or PyAV opening the header — and it
        ran unconditionally whenever `with_meta` was on, which is the default. So the most
        common pipeline in the namespace paid for it and never read it: a decode query
        (``read.images(decode=True, size=...)`` then ``select("image")``) spent a third of
        its wall clock parsing width/height/mode/format for columns the projection had
        already dropped. Measured on 2,000 JPEGs: 584 ms with the parse against 420 ms
        without, on a release build.

        A projection naming none of the metadata columns therefore skips the parse
        entirely. `None` means "every column", which does include them.
        """
        if not self._with_meta:
            return False
        if projection is None:
            return True
        return any(name in projection for name, _ in self._meta_fields())

    def row_count(self) -> int | None:
        """The number of media files — known from listing, without reading data."""
        return len(self._files())

    def identity(self) -> str:
        return f"{self.format_name}:{self._path}{'' if self._with_meta else ':nometa'}"

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
        from batcher.io.stats.file_listing import whole_file_statistics

        return whole_file_statistics(self._file_sizes(self._files()))

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
    def _build_batch(self, chunk: list[str], materialize: bool, want_meta: bool) -> pa.RecordBatch:
        """Assemble one Arrow `RecordBatch` from a chunk of files (no decode).

        `materialize` False (reference mode, or a projection that drops ``bytes``) leaves
        the ``bytes`` column null and touches only the header + size — so a chunk of GB
        videos costs kilobytes, not gigabytes. `want_meta` False skips the per-file header
        *parse* for the same reason: the projection is not going to read it.
        """
        return self._assemble(chunk, self._read_chunk(chunk, materialize, want_meta))

    def _assemble(self, chunk: list[str], reads: list[_Read]) -> pa.RecordBatch:
        """Build one `RecordBatch` from files and their already-read payloads.

        Split from the read so `read()` can bulk-fetch every file in one wide concurrent
        wave and then assemble the batches, instead of a fresh thread pool + a serial
        read per file-batch (which left a many-file scan far under the raw parallel-read
        throughput — the ingest floor for a directory of many small images/clips).

        Header metadata arrives already extracted (see `_read_payload_safe`), so this
        loop is list-appending only.
        """
        uris: list[str] = []
        blobs: list[bytes | None] = []
        sizes: list[int] = []
        mimes: list[str] = []
        meta_rows: list[dict[str, Any]] = []
        # The schema always declares the metadata columns when `with_meta` is on, so they
        # are always built — but a projection that named none of them skipped the parse,
        # leaving `meta_rows` empty. They come out all-null in that case, which is the
        # honest value for "not read" and is what the `.select` immediately discards.
        meta_fields = self._meta_fields() if self._with_meta else []
        for path, read in zip(chunk, reads, strict=True):
            if read is None:  # unreadable and tolerated — contributes no row
                continue
            header, payload, size, meta = read
            uris.append(path)
            blobs.append(payload)  # None in reference mode
            sizes.append(size)
            mimes.append(sniff_mime(path, header))
            if meta is not None:
                meta_rows.append(meta)
        arrays: list[pa.Array] = [
            pa.array(uris, pa.string()),
            pa.array(blobs, pa.large_binary()),
            pa.array(sizes, pa.int64()),
            pa.array(mimes, pa.string()),
        ]
        names = [n for n, _ in _COMMON_FIELDS]
        for name, dtype in meta_fields:
            values = [row.get(name) for row in meta_rows] if meta_rows else [None] * len(uris)
            arrays.append(pa.array(values, dtype))
            names.append(name)
        return pa.RecordBatch.from_arrays(arrays, names=names)

    def _read_chunk(self, chunk: list[str], materialize: bool, want_meta: bool) -> list[_Read]:
        """Read every file in ``chunk``, preserving order, concurrently where that helps.

        A **remote** media file is one object-store round trip and the read releases the
        GIL, so a serial loop leaves a many-file scan latency-bound on a single connection
        — the ingest bottleneck for a directory of many small images or clips.

        A **local** file is not that. It is a syscall on page cache, and fanning those
        across a pool costs more in dispatch than it saves in latency. This reader pooled
        unconditionally and paid for it: 2,000 local JPEGs (38.7 MB) read in **52 ms
        serially against 118 ms on an 8-thread pool and 130 ms on 64** — the pool was
        making the read two and a half times slower on the machine most people develop on.
        That is the same measurement `io._concurrent.read_each_file` had already made for
        footer reads, and the same conclusion; this reader simply was not using it.

        So the choice now comes from `read_each_file`, which owns it for every other
        connector in the tree. `materialize` chooses full-payload vs header-only reads
        (False in reference mode or when a projection drops ``bytes``); `want_meta`
        chooses whether the header is parsed at all.
        """
        from batcher.io._concurrent import read_each_file

        # `read_each_file` hands the filesystem to the callable; this reader closes over
        # its own, so the parameter is ignored rather than threaded twice.
        return read_each_file(
            self._fs,
            chunk,
            lambda _fs, path: self._read_payload_safe(path, materialize, want_meta),
        )

    def _read_payload_safe(self, path: str, materialize: bool, want_meta: bool) -> _Read:
        """Fetch one file and parse its header metadata; None marks a file to drop.

        Extraction runs **here**, inside the pool task, not in a serial loop after it: the
        fetch was already concurrent, so one Pillow / soundfile / av header parse per file
        on one thread was the part that was not. Each `_extract_meta` parses an independent
        `BytesIO` and shares no state, so it fans out safely — unlike the footer *parse*
        the pool-sizing comment below warns about.
        """
        try:
            header, payload, size = self._read_payload(path, materialize)
        except Exception as exc:
            self._errors.tolerate(path, exc, format_name=self.format_name)
            return None
        if not want_meta:
            return header, payload, size, None
        try:
            return header, payload, size, self._extract_meta(header)
        except Exception:  # a corrupt header nulls this file's metadata, not the batch
            return header, payload, size, dict.fromkeys(n for n, _ in self._meta_fields())

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

    def _read_payload(self, path: str, materialize: bool) -> tuple[bytes, bytes | None, int]:
        """Return ``(header_bytes, payload_or_None, size)`` for one file.

        With `materialize` the whole file is read (header == payload, size == len); without
        it only a header chunk is read and the size is a stat, leaving the payload `None` —
        so no GB payload is ever resident.
        """
        if materialize:
            with self._fs.open(path) as fh:
                data = fh.read()
            return data, data, len(data)
        with self._fs.open(path) as fh:
            header = fh.read(_HEADER_BYTES)
        return header, None, self._fs.size(path)

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
