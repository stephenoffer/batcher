"""`FileSource` — the Template-Method base every file-format reader subclasses.

Centralizes everything shared across file formats — path/glob/filesystem resolution,
schema caching, multi-file concatenation, projection plumbing, streaming, and split
generation — so a concrete format is a tiny subclass overriding only its per-file read
primitives. This is the shared-code spine that keeps each `io/formats/<fmt>.py` small
(the v2 antidote to v1's duplicated, mixin-heavy readers).

The `Source` protocol itself lives in `io.source`; this base structurally satisfies it.
"""

from __future__ import annotations

import hashlib
import os
from abc import ABC, abstractmethod
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import IO, Any, ClassVar

import pyarrow as pa

from batcher._internal.hardware import available_cpu_count
from batcher.io.filesystem import resolve_filesystem
from batcher.io.splits import FileSplit, Split

__all__ = ["FileSource"]

# How many files a streaming `iter_batches` decodes concurrently (bounded read-ahead).
# Caps the parallel-read memory to ~this many files while overlapping I/O + decode so a
# streaming consumer isn't throttled by a one-file-at-a-time read.
_ITER_READAHEAD_FILES = 16
# Concurrency for the driver's footer-read phase (`splits`/`row_count`). Footer reads are
# pure object-store *latency* (a small metadata GET each), not CPU or bandwidth, so a wide
# fan-out is safe and cuts the many-thousand-file driver stall the old cap of 16 left on the
# table. Env-overridable; capped at the file count so a small dataset spawns no idle threads.
_FOOTER_READ_CONCURRENCY = max(8, int(os.environ.get("BATCHER_FOOTER_CONCURRENCY", "64")))


class FileSource(ABC):
    """Base for a lazy, multi-file, projection-aware source over one format.

    Subclasses set `suffix` (for directory/glob expansion) and `format_name` (the
    registry key used to rebuild splits on a worker) and override `_read_schema`
    and `_read_file`. They may override `_iter_file` (streaming), `_file_row_count`
    (cheap counts), and `_file_splits` (sub-file split granularity).

    Examples:
        .. doctest::

            >>> from batcher.io import FileSource, ParquetSource
            >>> issubclass(ParquetSource, FileSource)
            True
    """

    suffix: ClassVar[str] = ""
    format_name: ClassVar[str] = ""

    __slots__ = ("_files_cache", "_fs", "_path", "_pinned", "_schema_cache", "_schema_mode")

    def __init__(
        self, path: str, *, schema_mode: str = "strict", files: list[str] | None = None
    ) -> None:
        self._path = path
        self._fs = resolve_filesystem(path)
        # `files` pins the source to an explicit subset of `path`'s data files instead of
        # expanding the directory. It is how a copy-on-write MERGE reads *only* the files
        # its key-pruning proved could match (`io.stats.key_pruning`): the whole point is
        # to never open the rest, so the file list cannot come from a directory listing.
        # One source still covers them all, which keeps row-group splits and the
        # single-source distributed path intact.
        #
        # Kept separately from `_files_cache` (which is merely a memo of the listing) because
        # `identity` must be able to tell "the whole directory" from "these files" — see there.
        self._pinned: list[str] | None = list(files) if files is not None else None
        self._files_cache: list[str] | None = list(files) if files is not None else None
        self._schema_cache: pa.Schema | None = None
        # "strict" (default) keeps the historical behavior — file 0's schema is
        # assumed for all. "union"/"latest" reconcile differing per-file schemas
        # (`io.schema_evolution`); each file's batches are normalized to the result.
        self._schema_mode = schema_mode

    # ---- shared, do-not-override ------------------------------------------
    def _files(self) -> list[str]:
        if self._files_cache is None:
            self._files_cache = self._fs.expand(self._path, suffix=self.suffix)
        return self._files_cache

    def _file_schema(self, path: str) -> pa.Schema:
        with self._fs.open(path) as fh:
            return self._read_schema(fh)

    def schema(self) -> pa.Schema:
        """The source's schema, read from metadata once and cached.

        In `strict` mode (the default) file 0's schema stands for all of them. The
        schema-evolution modes read every file's schema concurrently and unify them.

        Examples:
            .. doctest::

                >>> from batcher.io import ParquetSource  # doctest: +SKIP
                >>> ParquetSource("s3://bucket/events/").schema().names  # doctest: +SKIP
                ['id', 'ts']

        Returns:
            The Arrow schema every batch this source produces conforms to.
        """
        if self._schema_cache is None:
            files = self._files()
            if self._schema_mode == "strict":
                self._schema_cache = self._file_schema(files[0])
            else:
                from batcher.io.schema import unify_schemas

                # Schema evolution reads every file's schema; read them concurrently (each a
                # GIL-releasing metadata round trip) so a many-file unify isn't serialized.
                if len(files) <= 1:
                    schemas = [self._file_schema(f) for f in files]
                else:
                    cap = min(_FOOTER_READ_CONCURRENCY, len(files))
                    with ThreadPoolExecutor(max_workers=cap) as pool:
                        schemas = list(pool.map(self._file_schema, files))
                self._schema_cache = unify_schemas(schemas, self._schema_mode)
        return self._schema_cache

    def _read_by_path(
        self,
        path: str,  # noqa: ARG002 (the default reader has no path-based fast read)
        projection: list[str] | None,  # noqa: ARG002
    ) -> list[pa.RecordBatch] | None:
        """Read `path` without a Python file handle, or `None` to fall back to `open`.

        A format whose reader can do its own C++-side I/O (Parquet) overrides this: the
        handle otherwise serializes the reader's internal decode threads. Formats with no
        such path — and backends that cannot expose one — return `None` and are unchanged.
        """
        return None

    def read(self, projection: list[str] | None = None) -> list[pa.RecordBatch]:
        """Read every file, concurrently, into one list of batches in file order.

        Examples:
            .. doctest::

                >>> from batcher.io import ParquetSource  # doctest: +SKIP
                >>> batches = ParquetSource("s3://bucket/events/").read(["id"])  # doctest: +SKIP
                >>> batches[0].num_columns  # doctest: +SKIP
                1

        Args:
            projection: Columns the scan must produce. All columns when omitted;
                a columnar format reads only these.

        Returns:
            Every batch of every file, in file order.
        """
        files = self._files()

        def _read_one(f: str) -> list[pa.RecordBatch]:
            proj = self._file_proj(f, projection)
            batches = self._read_by_path(f, proj)
            if batches is not None:
                return list(self._normalize(batches, projection))
            with self._fs.open(f) as fh:
                return list(self._normalize(self._read_file(fh, proj), projection))

        # Read the files concurrently: the decode runs in the C++ layer with the GIL
        # released, so a many-small-files read (thousands of Parquet parts — the shape
        # every distributed producer and object-store dataset lands in) no longer opens
        # and parses them one at a time. `read()` already materializes the whole source,
        # so holding all batches adds no memory beyond what it already returns. Order is
        # preserved so a downstream that assumes file order is unaffected.
        if len(files) <= 1:
            return _read_one(files[0]) if files else []
        from concurrent.futures import ThreadPoolExecutor

        workers = min(len(files), available_cpu_count() * 2)
        out: list[pa.RecordBatch] = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for batches in pool.map(_read_one, files):  # order preserved
                out.extend(batches)
        return out

    def iter_batches(self, projection: list[str] | None = None) -> Iterator[pa.RecordBatch]:
        """Stream the files in order, decoding a bounded read-ahead window concurrently.

        Peak memory stays at ~16 in-flight files rather than the whole dataset, so a
        `read → transform → write` pipeline never materializes on the driver.

        Examples:
            .. doctest::

                >>> from batcher.io import ParquetSource  # doctest: +SKIP
                >>> src = ParquetSource("s3://bucket/events/")  # doctest: +SKIP
                >>> next(src.iter_batches()).num_rows  # doctest: +SKIP
                16384

        Args:
            projection: Columns the scan must produce. All columns when omitted.

        Returns:
            An iterator over every file's batches, in file order.
        """
        files = self._files()
        if len(files) <= 1:
            for f in files:
                yield from self._normalize(
                    self._iter_file(f, self._file_proj(f, projection)), projection
                )
            return

        # Bounded, order-preserving parallel read-ahead: keep ~`depth` files decoding
        # concurrently (Parquet/Arrow decode releases the GIL) so a streaming consumer —
        # a training loader's `iter_torch_batches`, a `read→map→write` — isn't throttled
        # by a one-file-at-a-time read (the serial read, not compute, is the ceiling).
        # Memory stays bounded to ~`depth` files (never the whole dataset), and files are
        # yielded in order so a downstream that assumes file order is unaffected.
        import itertools
        from collections import deque

        depth = min(len(files), max(2, min(available_cpu_count(), _ITER_READAHEAD_FILES)))

        def _read(f: str) -> list[pa.RecordBatch]:
            return list(self._iter_file(f, self._file_proj(f, projection)))

        remaining = iter(files)
        with ThreadPoolExecutor(max_workers=depth) as pool:
            pending = deque(pool.submit(_read, f) for f in itertools.islice(remaining, depth))
            while pending:
                fut = pending.popleft()
                nxt = next(remaining, None)
                if nxt is not None:
                    pending.append(pool.submit(_read, nxt))
                yield from self._normalize(iter(fut.result()), projection)

    def _file_proj(self, path: str, projection: list[str] | None) -> list[str] | None:
        """The columns to actually request from `path`. In non-strict mode a file may
        lack some unified/projected columns; request only those it has (the rest are
        filled with nulls by `normalize_batch`)."""
        if self._schema_mode == "strict" or projection is None:
            return projection
        present = set(self._file_schema(path).names)
        return [c for c in projection if c in present]

    def _normalize(
        self,
        batches: Iterator[pa.RecordBatch] | list[pa.RecordBatch],
        projection: list[str] | None,
    ) -> Iterator[pa.RecordBatch]:
        """In non-strict mode, reshape each batch to the unified (optionally
        projected) schema — adding missing columns as nulls and casting promoted
        types — so files with differing schemas concatenate cleanly."""
        if self._schema_mode == "strict":
            yield from batches
            return
        from batcher.io.schema import normalize_batch

        target = self.schema()
        if projection is not None:
            target = pa.schema([target.field(c) for c in projection])
        for b in batches:
            yield normalize_batch(b, target)

    def row_count(self) -> int | None:
        """The summed row count when every file knows its own cheaply, else None.

        Formats with a footer (Parquet, ORC) answer from metadata; the footers are read
        concurrently so a many-file source doesn't serialize the driver.

        Examples:
            .. doctest::

                >>> from batcher.io import ParquetSource  # doctest: +SKIP
                >>> ParquetSource("s3://bucket/events/").row_count()  # doctest: +SKIP
                600000000

        Returns:
            The exact total rows, or None if any file would need a data scan.
        """
        files = self._files()
        # Each `_file_row_count` reads a footer (a ~80ms object-store round trip for
        # Parquet, cached after the first read); over a many-file dataset the serial loop
        # dominates a distributed query's driver phase, so read them concurrently on a
        # small pool — exactly as `splits()` does. A single file skips the pool.
        if len(files) <= 1:
            counts = [self._file_row_count(f) for f in files]
        else:
            with ThreadPoolExecutor(max_workers=min(_FOOTER_READ_CONCURRENCY, len(files))) as pool:
                counts = list(pool.map(self._file_row_count, files))
        return None if any(c is None for c in counts) else sum(counts)  # type: ignore[misc]

    def identity(self) -> str:
        """The ``format:path`` key this source's learned statistics are stored under.

        A source pinned to an explicit **subset** of `path`'s files (see `files`) gets its
        own key, suffixed with a digest of that subset. It has to: statistics are cached and
        persisted under this key, and a subset is a *different relation* from the directory
        it lives in — same path, different rows. Sharing the directory's key would hand a
        one-file source the whole table's row count and zone maps, and the optimizer would
        size its joins against a relation fifty times bigger than the one it is about to
        read. (That is not hypothetical: it is what made a pruned MERGE estimate a 100,000-row
        join at 2.4 TB and spill it to disk.)

        Examples:
            .. doctest::

                >>> from batcher.io import ParquetSource  # doctest: +SKIP
                >>> ParquetSource("s3://bucket/events/").identity()  # doctest: +SKIP
                'parquet:s3://bucket/events/'

        Returns:
            A stable identifier for this source.
        """
        base = f"{self.format_name}:{self._path}"
        if self._pinned is None:
            return base
        digest = hashlib.sha256("\n".join(self._pinned).encode()).hexdigest()[:16]
        return f"{base}#{digest}"

    def splits(self, target_size: int | None = None, predicate: dict | None = None) -> list[Split]:
        """Independently-readable slices — one per file, or finer where the format allows.

        Parquet subdivides into row-group runs; line-delimited text into byte ranges.
        A schema-evolving read emits a single `WholeSourceSplit`, since a per-file split
        rebuilds a reader that knows nothing of the unified schema.

        Examples:
            .. doctest::

                >>> from batcher.io import ParquetSource  # doctest: +SKIP
                >>> len(ParquetSource("s3://bucket/events/").splits())  # doctest: +SKIP
                128

        Args:
            target_size: Rough size (bytes) to aim for per split. The format's own
                granularity is used when omitted.
            predicate: An optional pushed-down filter, as its IR dictionary. Splits whose
                recorded bounds prove they cannot match it are pruned, so a selective read
                never opens the files it does not need.

        Returns:
            The splits covering the source exactly once.
        """
        # Per-file splits each reconstruct a single-file reader with no knowledge of
        # the unified schema, so in a non-strict (schema-evolving) read they would
        # skip normalization and produce mismatched batches. Read such a source as a
        # single whole-source split — correct (the unification happens in `read`),
        # at the cost of per-file parallelism for evolving reads.
        if self._schema_mode != "strict":
            from batcher.io.splits import WholeSourceSplit

            return [WholeSourceSplit(self)]
        files = self._files()
        # `_file_splits` reads each file's footer (a ~100ms object-store round trip for
        # Parquet); over a many-file dataset that serial loop dominates a distributed
        # query's driver phase (TPC-H sf100: 100 files ≈ 12s of otherwise-idle driver
        # time while the workers wait). Read the footers concurrently on a small pool —
        # order is preserved so a downstream that assumes file order is unaffected. A
        # single file (the common small case) skips the pool entirely.
        if len(files) <= 1:
            return [s for f in files for s in self._file_splits(f, target_size, predicate)]
        with ThreadPoolExecutor(max_workers=min(_FOOTER_READ_CONCURRENCY, len(files))) as pool:
            per_file = pool.map(lambda f: self._file_splits(f, target_size, predicate), files)
        return [s for file_splits in per_file for s in file_splits]

    # ---- override points --------------------------------------------------
    @abstractmethod
    def _read_schema(self, fh: IO[Any]) -> pa.Schema:
        """Read the schema from an open file handle (no data scan where possible)."""

    @abstractmethod
    def _read_file(self, fh: IO[Any], projection: list[str] | None) -> list[pa.RecordBatch]:
        """Read one file's batches from an open handle, honoring `projection`."""

    def _iter_file(self, path: str, projection: list[str] | None) -> Iterator[pa.RecordBatch]:
        with self._fs.open(path) as fh:
            yield from self._read_file(fh, projection)

    def _file_row_count(self, path: str) -> int | None:  # noqa: ARG002 (default: unknown)
        return None

    def _reader_kwargs(self) -> dict[str, object]:
        """The non-path construction arguments a worker needs to rebuild a single-file reader.

        Empty for the common formats (Parquet/CSV/JSON/…) whose reader is fully determined by
        the path. A format whose ``__init__`` takes required or behavior-changing keyword
        arguments (a protobuf ``message_cls``, an Excel ``sheet``, a point-cloud
        ``columns``/``dtype``) MUST override this so `FileSplit` can reconstruct it on the
        worker; otherwise a distributed / multi-file read rebuilds a reader that raises or
        silently reads the wrong data. Values must be picklable (they ship to the worker).
        """
        return {}

    def _file_splits(
        self,
        path: str,
        target_size: int | None,  # noqa: ARG002 (a whole-file format has no sub-file granularity)
        predicate: dict | None = None,  # noqa: ARG002 (no footer stats to prune with)
    ) -> list[Split]:
        """This file's splits. A format with footer statistics overrides this to *prune*.

        `predicate` is Kyber's pushed filter, offered here so a format that records per-chunk
        bounds (Parquet row-groups, ORC stripes) can drop the chunks that provably hold no
        matching row — at **plan** time, so they never become a task, never get balanced, and
        never get opened. A format with no such statistics ignores it; the engine's `Filter`
        re-checks every row regardless, so ignoring it is always correct and merely slower.
        """
        return [FileSplit(self.format_name, path, self._reader_kwargs())]
