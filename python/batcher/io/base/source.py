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
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import IO, Any, ClassVar, TypeVar

import pyarrow as pa

from batcher._internal.hardware import available_cpu_count
from batcher.io._backend import _scheme
from batcher.io.base._readahead import ordered_readahead
from batcher.io.base._tolerance import ErrorPolicy
from batcher.io.base._transient import with_retry
from batcher.io.filesystem import resolve_filesystem
from batcher.io.splits import FileSplit, Split
from batcher.io.stats.file_identity import files_version

__all__ = ["FileSource"]

_T = TypeVar("_T")

# How many files a streaming `iter_batches` decodes concurrently (bounded read-ahead).
# Caps the parallel-read memory to ~this many files while overlapping I/O + decode so a
# streaming consumer isn't throttled by a one-file-at-a-time read.
_ITER_READAHEAD_FILES = 16
# Ceiling on the total *undelivered decoded bytes* the read-ahead window may hold. This,
# not the file count, is what bounds `iter_batches`: file count alone says nothing about
# memory when one row can be a 200 MB video and another 4 KB of text. 512 MiB keeps a
# streaming read comfortably inside a worker's envelope while still overlapping I/O.
_ITER_READAHEAD_BYTES = max(1 << 20, int(os.environ.get("BATCHER_READAHEAD_BYTES", str(512 << 20))))
# How many files a **remote** (object-store) source reads concurrently, in `read` and as the
# `iter_batches` read-ahead depth. Both used to be derived from `available_cpu_count()`, which
# is the wrong ruler for a remote read: an S3 GET is ~tens of ms of *latency*, so throughput
# tracks the number of requests in flight, not the number of cores available to decode them.
# A 4-core worker therefore read 4 files at a time and sat idle waiting on the network. The
# distributed scan already sizes its prefetch this way and measured it (`dist/executors/
# scan_read.py::_SCAN_PREFETCH`: 8 -> 32 cut a TPC-H sf100 distributed agg ~53s -> ~31s, and
# it plateaus past 32) — this is the same lever on the single-node path.
#
# Local files keep the core-count sizing: an NVMe read is bandwidth-bound, not latency-bound,
# so oversubscribing it buys nothing and costs resident batches.
#
# Memory: read-ahead depth multiplies the in-flight decoded data, so this raises the file
# count but NOT the ceiling on held bytes — `iter_batches` is bounded by
# `_ITER_READAHEAD_BYTES` (split `budget / depth` per in-flight file by `ordered_readahead`),
# which a deeper window divides more finely rather than exceeding. `read` materializes the
# whole source by definition, so extra concurrency there only widens the transient decode
# working set.
_REMOTE_READ_CONCURRENCY = max(2, int(os.environ.get("BATCHER_REMOTE_READ_CONCURRENCY", "32")))
# Concurrency for the driver's footer-read phase (`splits`/`row_count`). Footer reads are
# pure object-store *latency* (a small metadata GET each), not CPU or bandwidth, so a wide
# fan-out is safe and cuts the many-thousand-file driver stall the old cap of 16 left on the
# table. Env-overridable; capped at the file count so a small dataset spawns no idle threads.
_FOOTER_READ_CONCURRENCY = max(8, int(os.environ.get("BATCHER_FOOTER_CONCURRENCY", "64")))
# Attempts (including the first) for a read that fails *transiently* — an object-store
# throttle, 5xx, or dropped connection. 3 absorbs the blips a cloud SDK would absorb on its
# own without masking a real outage for long; 1 disables retrying. Non-transient failures
# (404/403/malformed) never consume an attempt, so a genuine error still fails on the first
# try. See `_transient.py` for why the classification, not the count, is the load-bearing part.
_READ_RETRY_ATTEMPTS = max(1, int(os.environ.get("BATCHER_READ_RETRY_ATTEMPTS", "3")))
# First retry's backoff ceiling in seconds, doubling per round with equal jitter. Jitter
# matters more than the base here: a wide scan retries hundreds of files at once, and
# without decorrelation a single throttle turns into a synchronized stampede.
_READ_RETRY_BACKOFF_S = max(0.0, float(os.environ.get("BATCHER_READ_RETRY_BACKOFF_S", "0.5")))


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

    __slots__ = (
        "_errors",
        "_files_cache",
        "_filesystem",
        "_fs",
        "_path",
        "_pinned",
        "_schema_cache",
        "_schema_mode",
        "_storage_options",
    )

    def __init__(
        self,
        path: str,
        *,
        schema_mode: str = "strict",
        files: list[str] | None = None,
        on_error: str = "raise",
        filesystem: object = None,
        storage_options: dict[str, str] | None = None,
    ) -> None:
        # `on_error` decides whether one unreadable file aborts the whole read; the
        # policy object also keeps the audit trail `corrupt_files()` exposes.
        self._errors = ErrorPolicy(on_error)
        self._path = path
        # Bring-your-own filesystem / credentials; threaded to workers via `_reader_kwargs`.
        # `storage_options` (a plain dict) is the portable choice — see `resolve_filesystem`.
        self._filesystem = filesystem
        self._storage_options = storage_options
        self._fs = resolve_filesystem(path, filesystem=filesystem, storage_options=storage_options)
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

    def _is_remote(self) -> bool:
        """Whether this source's files sit behind a network round trip.

        The scheme is the whole test: a bare path or ``file://`` is local disk, anything
        else (object store, HDFS, HTTP, an fsspec-backed scheme) pays per-request latency
        and is therefore sized by in-flight requests rather than by cores. See
        `_REMOTE_READ_CONCURRENCY`.
        """
        return _scheme(self._path) not in ("", "file")

    def _read_concurrency(self, n_files: int) -> int:
        """How many files a materializing `read` decodes at once — never more than there are.

        Shared with `ParquetSource.read`, which runs its own pool over the same files: the
        two must agree, and a second copy of the sizing rule is exactly the duplication that
        lets one path quietly keep the old core-count bound.
        """
        by_core = available_cpu_count() * 2
        return min(
            n_files, max(by_core, _REMOTE_READ_CONCURRENCY) if self._is_remote() else by_core
        )

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
            def _once() -> list[pa.RecordBatch]:
                proj = self._file_proj(f, projection)
                batches = self._read_by_path(f, proj)
                if batches is not None:
                    return list(self._normalize(batches, projection))
                with self._fs.open(f) as fh:
                    return list(self._normalize(self._read_file(fh, proj), projection))

            try:
                return self._read_with_retry(_once)
            except Exception as exc:
                if not self._errors.tolerate(f, exc, format_name=self.format_name):
                    raise
                return []

        # Read the files concurrently: the decode runs in the C++ layer with the GIL
        # released, so a many-small-files read (thousands of Parquet parts — the shape
        # every distributed producer and object-store dataset lands in) no longer opens
        # and parses them one at a time. `read()` already materializes the whole source,
        # so holding all batches adds no memory beyond what it already returns. Order is
        # preserved so a downstream that assumes file order is unaffected.
        if len(files) <= 1:
            return _read_one(files[0]) if files else []
        from concurrent.futures import ThreadPoolExecutor

        workers = self._read_concurrency(len(files))
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
        #
        # The bound is on *bytes*, not on files. Bounding by files (the obvious
        # `list(self._iter_file(f))` per worker) makes peak memory `depth x decoded-file
        # -size` — independent of the batch size, and unbounded in practice: 1 GB shards
        # at depth 16 is ~16 GB, and one row of a multimodal corpus can itself be a
        # 200 MB video. `ordered_readahead` streams batches instead, so a 1 GB shard and
        # a 4 KB one cost the same. Files are still yielded in order, so a downstream
        # that assumes file order is unaffected.
        yield from self._normalize(
            ordered_readahead(
                files,
                lambda f: self._tolerant_iter_file(f, projection),
                depth=self._iter_readahead_depth(len(files)),
                max_bytes=_ITER_READAHEAD_BYTES,
            ),
            projection,
        )

    def _iter_readahead_depth(self, n_files: int) -> int:
        """How many files `iter_batches` decodes concurrently — never more than there are.

        Depth is chosen by where the files live, not by how many cores can decode them: a
        remote read is latency-bound and wants many requests in flight, local disk is
        bandwidth-bound and does not (see `_REMOTE_READ_CONCURRENCY`). The *byte* bound is
        unchanged either way — a deeper window divides the same `_ITER_READAHEAD_BYTES` into
        more, smaller per-file budgets rather than holding more data — so depth trades
        nothing away for the extra overlap.

        This is a **method, not an inline expression**, because a per-file reader has to be
        able to ask it. A format whose own reader parallelizes internally (Parquet's native
        row-group windows) must not fan out underneath a read-ahead that is already fanning
        out — the two multiply into an oversubscribed decode that measured 3x *slower* than
        no fan-out at all. `parquet/_native_stream.py` carries that measurement; the point
        here is that the two decisions must be reading the same number.
        """
        local = max(2, min(available_cpu_count(), _ITER_READAHEAD_FILES))
        return min(n_files, _REMOTE_READ_CONCURRENCY if self._is_remote() else local)

    def _read_with_retry(self, op: Callable[[], _T]) -> _T:
        """Run a read, retrying an object-store blip before it counts as a failure.

        A throttle, a 5xx or a dropped connection is not a property of the data — the same
        GET a moment later succeeds — but the IO layer had no retry at all, so one blip
        either aborted the query or, under `on_error="skip"`, silently recorded a healthy
        file as corrupt and dropped its rows. Retrying *before* `ErrorPolicy` is consulted
        is what keeps `skip` meaning "genuinely unreadable". Non-transient failures
        (404/403/malformed) are re-raised on the first attempt, so a real error still fails
        fast. Local reads are unaffected in practice: nothing on local disk classifies as
        transient.
        """
        return with_retry(op, attempts=_READ_RETRY_ATTEMPTS, backoff_base_s=_READ_RETRY_BACKOFF_S)

    def _tolerant_iter_file(
        self, path: str, projection: list[str] | None
    ) -> Iterator[pa.RecordBatch]:
        """`_iter_file`, honoring `on_error`, retrying a blip that hits before any output.

        A failure mid-file yields the batches already decoded and then stops, rather than
        discarding them: they were read successfully, and for a truncated file (the common
        corruption) the valid prefix is exactly what the caller wants to keep.

        Retry is deliberately limited to a failure that lands *before the first batch is
        yielded*. Once rows have gone downstream, re-running the file would re-yield them
        and duplicate data — a silent correctness bug, and a far worse outcome than the
        failure being retried. So the streaming path gets the blip protection only where it
        is provably safe; `read()` (which materializes before returning anything) retries
        throughout.
        """

        def _open_and_prime() -> tuple[Iterator[pa.RecordBatch], pa.RecordBatch | None]:
            # Open + first decode as one retryable unit: this is where a connect/throttle
            # blip lands, and nothing has been yielded yet, so re-running is free of
            # duplicates. `None` means the file held no batches at all.
            it = iter(self._iter_file(path, self._file_proj(path, projection)))
            return it, next(it, None)

        try:
            it, first = self._read_with_retry(_open_and_prime)
            if first is not None:
                yield first
                yield from it  # past the first batch: a failure here is tolerated, not retried
        except Exception as exc:
            if not self._errors.tolerate(path, exc, format_name=self.format_name):
                raise

    def corrupt_files(self) -> list[str]:
        """The paths this source skipped, in failure order (empty unless `on_error="skip"`).

        Examples:
            .. doctest::

                >>> from batcher.io import ParquetSource  # doctest: +SKIP
                >>> src = ParquetSource("s3://bucket/events/", on_error="skip")  # doctest: +SKIP
                >>> _ = src.read()  # doctest: +SKIP
                >>> src.corrupt_files()  # doctest: +SKIP
                ['s3://bucket/events/part-0042.parquet']

        Returns:
            The skipped paths. A skipped file is invisible in the data, so this is the
            only way to tell a clean read from a partial one.
        """
        return self._errors.skipped()

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

    def stats_version(self) -> str | None:
        """A token that changes whenever this source's files could have changed.

        `identity()` names *which relation* this is; this names *which version of it*.
        Statistics are memoized under the identity, and a stale zone map yields a wrong
        answer rather than a slower plan — see `io.stats.file_identity.files_version`,
        which owns the computation and the reasoning.

        Examples:
            .. doctest::

                >>> import pyarrow as pa, pyarrow.parquet as pq
                >>> from batcher.io.formats.structured.parquet.source import ParquetSource
                >>> pq.write_table(pa.table({"x": [1]}), "v.parquet")
                >>> before = ParquetSource("v.parquet").stats_version()
                >>> pq.write_table(pa.table({"x": [1, 2, 3]}), "v.parquet")
                >>> before == ParquetSource("v.parquet").stats_version()
                False

        Returns:
            The version token, or None when any file cannot be identified — in which case
            the caller must not cache, having no way to notice a later change.
        """
        return files_version(self._files(), self._fs)

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
            return [s for f in files for s in self._tolerant_file_splits(f, target_size, predicate)]
        with ThreadPoolExecutor(max_workers=min(_FOOTER_READ_CONCURRENCY, len(files))) as pool:
            per_file = pool.map(
                lambda f: self._tolerant_file_splits(f, target_size, predicate), files
            )
        return [s for file_splits in per_file for s in file_splits]

    def _tolerant_file_splits(
        self, path: str, target_size: int | None, predicate: dict | None
    ) -> list[Split]:
        """`_file_splits`, honoring `on_error`.

        Planning a split reads the file's metadata (a Parquet footer, a text file's size),
        so it fails on exactly the corruption a tolerated read exists to survive — and it
        fails on the *driver*, before any worker reads a byte. Without this, `on_error="skip"`
        covered the read but not the plan, so one truncated shard still aborted the whole
        distributed query and the tolerance looked wired-up while being unreachable.

        A file that cannot be planned contributes no splits, which is what skipping it
        means. The drop is recorded here on the driver, so `corrupt_files()` names it.
        """
        try:
            return list(self._file_splits(path, target_size, predicate))
        except Exception as exc:
            if not self._errors.tolerate(path, exc, format_name=self.format_name):
                raise
            return []

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
        """Non-path construction args a worker needs to rebuild a single-file reader.

        The base returns the caller's bring-your-own filesystem / credentials
        (`storage_options`, `filesystem`) so a distributed read resolves the same backend on
        every worker instead of its own env vars, plus the `on_error` policy so a tolerated
        read stays tolerant off the driver. A format with behavior-changing keywords (a
        protobuf ``message_cls``, an Excel ``sheet``) overrides this as
        ``{**super()._reader_kwargs(), ...}`` — folding in the base, never replacing it, or
        the credentials are dropped. Values must be picklable (they ship to the worker); a
        live `filesystem` rides the split only if it pickles.

        `on_error` is carried for the same reason `splits()` degrades to a
        `WholeSourceSplit` in non-strict `schema_mode`: a worker rebuilds the reader as
        ``SOURCES.get(fmt)(path, **kwargs)``, so anything omitted here silently reverts to
        its constructor default. Omitting it made ``read(..., on_error="skip")`` a no-op on
        every split-based path — the distributed executor, the streaming reader, and the GPU
        backend — turning an explicitly tolerated read back into a fail-fast one. It is
        emitted only when it differs from the default so a clean read's split kwargs (and
        therefore its `identity()`) are unchanged.

        The skip *audit trail* stays worker-local: `corrupt_files()` on the driver-side
        source cannot see what a worker dropped. The distributed scan reports that
        separately via `skipped_splits()`.
        """
        extra: dict[str, object] = {}
        if self._storage_options is not None:
            extra["storage_options"] = self._storage_options
        if self._filesystem is not None:
            extra["filesystem"] = self._filesystem
        if self._errors.mode != "raise":
            extra["on_error"] = self._errors.mode
        return extra

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
