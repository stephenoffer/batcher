"""`ParquetSource` — lazy projection/predicate read of one or more Parquet files.

The workhorse scan: schema and row counts come from the footer (never a data scan), the
pushed predicate drives row-group + page-index pruning in the native Rust reader (pyarrow
`filters` when it has no native translation), and splits go down to row-group granularity
so a distributed read slices *within* a file.

Reads route to the native reader wherever one applies — whole files (`_read_by_path`,
`_native_read_many`), filtered files (`_native_read_filtered`), and streaming
(`_iter_file`, a bounded row-group window at a time). Every one of those falls back to
pyarrow silently and produces the same rows, so the choice is only ever about speed.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import IO, Any

import pyarrow as pa

from batcher._internal.errors import SchemaError
from batcher.io.base import FileSource
from batcher.io.formats.base import SOURCES
from batcher.io.formats.structured import _parquet_native
from batcher.io.formats.structured.parquet import _native_stream
from batcher.io.splits import FileSplit, Split, parquet_row_group_splits
from batcher.io.splits.parquet import _parquet_footer
from batcher.io.stats.file_identity import FileMetaCache, file_identity
from batcher.plan.source_stats import SourceStatistics

__all__ = ["ParquetSource"]

# Process-wide cache of per-file Parquet row counts, keyed by file identity. A footer read
# is a ~80 ms object-store round trip, and without this EVERY distributed `collect` re-reads
# every source file's footer just to size the worker fan-out (`learned_num_workers` →
# `total_source_rows`): measured ~0.9 s/collect on a 10-file sf10 groupby, dwarfing the
# 0.26 s shuffle it was sizing.
#
# Bounded, where it used to be a plain `dict`. An unbounded memo here grows with every file
# the *process* has ever counted rather than with any query's working set — a long-lived
# worker cycling through datasets leaks one entry per file, forever. The entries are single
# integers, so the budget is generous: comfortably more than the per-file-sweep ceiling, so a
# whole planning pass stays resident and the bound only ever trims history.
_ROW_COUNT_CACHE = FileMetaCache(65_536)


@SOURCES.register("parquet")
class ParquetSource(FileSource):
    """One or more Parquet files (single file, directory, or glob).

    Examples:
        .. doctest::

            >>> from batcher.io import ParquetSource  # doctest: +SKIP
            >>> src = ParquetSource("s3://bucket/lineitem/*.parquet")  # doctest: +SKIP
            >>> src.row_count()  # from the footers, no data scan  # doctest: +SKIP
            600037902
    """

    suffix = ".parquet"
    format_name = "parquet"
    # Predicate pushdown: Kyber's pushed predicate goes to the native Rust reader, which
    # prunes row-groups by footer statistics AND pages by the page index (ColumnIndex/
    # OffsetIndex) before decoding; one it cannot express (temporal literals) falls back to
    # pyarrow `filters`, row-groups only. Both superset-safe — the `Filter` stays above.
    supports_predicate = True

    __slots__ = ()

    def _read_schema(self, fh: IO[Any]) -> pa.Schema:
        import pyarrow.parquet as pq

        return pq.read_schema(fh)

    def _read_file(self, fh: IO[Any], projection: list[str] | None) -> list[pa.RecordBatch]:
        import pyarrow.parquet as pq

        return pq.read_table(fh, columns=projection).to_batches()

    def _native_read_many(self, projection: list[str] | None) -> list[pa.RecordBatch] | None:
        """All files in one batched native pass (see `_parquet_native.read_many`), or ``None``.

        The unfiltered multi-file read: one native call overlaps every file's I/O, beating a
        per-file thread pool on a many-small-files scan. ``None`` (use base's per-file path)
        for a single file, in non-strict mode (base owns normalization), with a byte cache,
        or for a bring-your-own backend the bare URI cannot address.

        The native reader returns each file with its own on-disk schema, so the result is
        conformed to the source's declared schema per file on the way out — the same check
        `_normalize` applies on the base path. Skipping it here is what let a directory
        whose files disagree return silently column-dropped rows on this fast path while
        the slower path raised.
        """
        files = self._files()
        if len(files) <= 1 or self._schema_mode != "strict":
            return None
        if not self._native_uri_is_addressable(files[0]):
            return None
        per_file = _parquet_native.read_many(files, projection)
        if per_file is None:
            return None
        return [
            b
            for path, file_batches in zip(files, per_file, strict=True)
            for b in self._normalize(file_batches, projection, path)
        ]

    def _native_uri_is_addressable(self, path: str) -> bool:
        """Whether the native reader can be handed `path` and reach the same bytes.

        The native FFI takes a bare URI (`bc_py::read_parquet*`) and resolves the backend
        itself from the environment and the URI's own query string. It therefore cannot see
        a caller-supplied ``filesystem=`` or ``storage_options=`` — and a dict-carried
        ``endpoint_override`` is exactly the case where that matters, because the bare URI
        then addresses *real* S3 rather than the on-prem MinIO/Ceph the caller configured.
        A wrong-store read is not a slower answer, it is a different object, so BYO
        credentials keep the pyarrow reader that honors them.

        Same trade `_file_splits` makes when it declines row-group splits for a BYO backend.
        A byte cache also withholds its target, since the native reader would bypass it.
        """
        if self._filesystem is not None or self._storage_options is not None:
            return False
        return self._fs.native_read_target(path) is not None

    def _native_read_filtered(
        self, projection: list[str] | None, predicate: dict
    ) -> list[pa.RecordBatch] | None:
        """Every file read natively, pruned AND filtered by `predicate`, or ``None``.

        The *selective* scan is what pushdown exists for, and it was the one case that never
        reached the native reader: `read` tried native only when there was NO predicate, so
        adding a filter — the thing that should make a scan cheaper — dropped it onto PyArrow
        and gave up the 3-4x `_parquet_native` measures on object storage.

        **Prune, then filter — the two are not alternatives.** `read_row_groups_filtered`
        skips row-groups and pages whose statistics prove no row matches, which is an I/O
        win; but it returns the survivors *unfiltered*, and how much that is depends entirely
        on how the data is clustered. On a clustered key pruning does nearly all the work; on
        a scattered one it can do none. Measured on a 20M-row/100-row-group file with a ~1%
        predicate whose matches land in every row-group, returning the pruned-only result
        handed the driver 20,000,000 rows / 320 MB where PyArrow's `filters=` handed it
        199,575 rows / 3.2 MB. That is contract-legal — the engine's `Filter` still produces
        the right answer — and it is an OOM on a large scan. So the pruned batches get the
        same predicate applied as a vectorized Arrow filter before they are returned: Arrow
        C++ filtering 20M rows is cheap next to materializing 320 MB downstream, and the
        result is then *exactly* the matching rows, as small as PyArrow's and decoded at
        native speed.

        ``None`` (caller falls back to PyArrow) when the predicate has no native translation
        — notably temporal literals, which `to_native_predicate` refuses because it cannot
        verify the parquet physical unit and which PyArrow *can* prune on — or when any
        file's native read or filter fails. Failing all-or-nothing keeps one read path per
        call, so a partial native result is never concatenated with a differently-derived one.
        """
        from batcher.io.predicate import to_native_predicate

        files = self._files()
        if not files or not self._native_uri_is_addressable(files[0]):
            return None
        if to_native_predicate(predicate) is None:
            return None
        # The Arrow-side filter. Every predicate `to_native_predicate` accepts is also
        # expressible here, so this is not expected to be None — but a superset is still a
        # correct answer, so an absent expression degrades to pruning alone rather than
        # failing. Built once and shared: it is immutable and thread-safe.
        pa_filter = self._pa_filter(predicate)

        def _read_one(f: str) -> list[pa.RecordBatch] | None:
            # `[]` row-groups = every row-group in the file; the reader prunes from there.
            batches = _parquet_native.read_row_groups_filtered(f, [], projection, predicate)
            if not batches:
                return batches  # None, or a file pruned away to nothing — both pass through
            # Conform before filtering: the filter is bound against the source's declared
            # schema, so a file whose column types differ must be cast to it first.
            batches = list(self._normalize(batches, projection, f))
            if pa_filter is None:
                return batches
            # Filter per file, as each read lands, so the accumulated result holds only
            # matching rows — filtering at the end would first materialize every file's
            # unfiltered batches at once, which is the memory this exists to avoid.
            try:
                return pa.Table.from_batches(batches).filter(pa_filter).to_batches()
            except Exception:
                return None  # a kernel the filter can't bind → whole read falls back

        if len(files) == 1:
            per_file = [_read_one(files[0])]
        else:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=self._read_concurrency(len(files))) as pool:
                per_file = list(pool.map(_read_one, files))  # order preserved
        if any(batches is None for batches in per_file):
            return None
        return [b for batches in per_file for b in batches]  # type: ignore[union-attr]

    def _read_by_path(self, path: str, projection: list[str] | None) -> list[pa.RecordBatch] | None:
        """The unfiltered read: native Rust reader when possible, else PyArrow.

        ``None`` (fall to the handle path) only when a read-through byte cache is active —
        the native reader and PyArrow's `filesystem=` read both bypass that cache, so its
        reads must go through `open`. Otherwise try the native Rust reader (3-4x faster on
        object storage) and fall back to PyArrow's native-filesystem read on anything it
        cannot handle.

        A bring-your-own `filesystem=`/`storage_options=` keeps PyArrow throughout: the
        native FFI resolves the backend from the bare URI and cannot see them, so a
        dict-carried ``endpoint_override`` would address real S3 instead of the configured
        MinIO/Ceph. That is a *different object*, not a slower read. Same trade `_file_splits`
        already makes when it declines row-group splits for a BYO backend.
        """
        if not self._native_uri_is_addressable(path):
            # Still let PyArrow do its own I/O when it can — only the native reader is
            # unsafe here, and the handle path is markedly slower on a wide projection.
            if self._fs.native_read_target(path) is None:
                return None
            return self._read_table(path, projection).to_batches()
        native = _parquet_native.read_one(path, projection)
        if native is not None:
            return native
        return self._read_table(path, projection).to_batches()

    def _read_table(
        self, path: str, projection: list[str] | None, pa_filter: Any = None
    ) -> pa.Table:
        """Read `path`, letting pyarrow do its own I/O when the backend allows it.

        Handed a Python file object, pyarrow's reader round-trips every read through the
        interpreter, and that serializes the decode threads it fans across column chunks
        — so the read gets *superlinearly* slower as the projection widens. Measured on
        TPC-H sf100 `lineitem` (one 16 GB file, 600 M rows): one column reads in 648 ms
        either way, four columns in 2,831 ms through a handle against 1,653 ms when
        pyarrow owns the I/O. Backends that cannot hand over a native target (fsspec
        behind a read-through cache) keep the handle; the result is identical.
        """
        import pyarrow.parquet as pq

        from batcher.io.filesystem import ensure_io_threads

        ensure_io_threads()  # lift the 8-thread IO cap so a wide S3 read isn't throttled
        target = self._fs.native_read_target(path)
        if target is not None:
            fs, in_path = target
            return pq.read_table(
                in_path, filesystem=fs, columns=projection, filters=pa_filter, pre_buffer=True
            )
        with self._fs.open(path) as fh:
            return pq.read_table(fh, columns=projection, filters=pa_filter)

    def _iter_file(self, path: str, projection: list[str] | None) -> Iterator[pa.RecordBatch]:
        import pyarrow.parquet as pq

        from batcher.io.filesystem import ensure_io_threads

        ensure_io_threads()
        # Prefer the native `(fs, path)` target with column-chunk pre-buffering: reading
        # through a Python handle serializes pyarrow's per-column decode threads (the same
        # anti-pattern `_read_table` avoids), and no `pre_buffer` leaves scattered column
        # chunks as many small GETs. Backends with no native target keep the handle.
        target = self._fs.native_read_target(path)
        if target is None:
            with self._fs.open(path) as fh:
                yield from pq.ParquetFile(fh).iter_batches(columns=projection)
            return
        fs, in_path = target
        pf = pq.ParquetFile(in_path, filesystem=fs, pre_buffer=True)
        # Streaming stayed on pyarrow long after the *materializing* read (`_read_by_path`)
        # moved to the native reader, so `read -> map -> write` — the shape that most wants
        # throughput, being the one that cannot afford to materialize — was the last one
        # paying pyarrow's row-group-at-a-time decode. Native windows fix that, but ONLY
        # where the read-ahead above is not already fanning across files: the two fan-outs
        # multiply into an oversubscribed decode. `_native_stream` owns that rule and the
        # measurement behind it. `pf` serves either way — it plans the windows from a footer
        # it has already parsed, and is their fallback reader — so nothing extra is read.
        if self._native_uri_is_addressable(path) and _native_stream.use_native_stream(
            self._iter_readahead_depth(len(self._files())), self._is_remote()
        ):
            yield from _native_stream.iter_windows(path, pf, projection)
            return
        yield from pf.iter_batches(columns=projection)

    @staticmethod
    def _pa_filter(predicate: dict | None) -> Any:
        if predicate is None:
            return None
        from batcher.io.predicate import to_pyarrow_expression

        return to_pyarrow_expression(predicate)

    def read(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> list[pa.RecordBatch]:
        """Read every file concurrently, pruning row-groups with `predicate`.

        The contract is only that the result is a **superset** of the matching rows — the
        engine keeps its `Filter` above the scan (`core.scan_only_result` declines its no-op
        shortcut whenever a predicate was pushed, for exactly this reason), so a filter the
        reader cannot bind never fails the query; it falls back to a coarser reader and reads
        more rows. In practice both pushdown paths return exactly the matching rows, and they
        take care to: a pruning-only result is correct but can be the *whole file* when the
        predicate's matches are scattered across every row-group, which is a 100x memory
        difference on a large scan. See `_native_read_filtered`.

        Examples:
            .. doctest::

                >>> from batcher.io import ParquetSource  # doctest: +SKIP
                >>> src = ParquetSource("s3://bucket/lineitem/")  # doctest: +SKIP
                >>> batches = src.read(["l_orderkey"])  # doctest: +SKIP
                >>> batches[0].num_columns  # doctest: +SKIP
                1

        Args:
            projection: Columns to read. All columns when omitted; only these
                column chunks are fetched and decoded.
            predicate: A predicate-IR filter, pushed to the native reader for
                row-group + page-index pruning, or to a pyarrow filter when it has
                no native translation.

        Returns:
            Every batch of every file, in file order.
        """
        # A schema-evolving read must reconcile every file's batches to the unified
        # schema (base `read` does this via `_normalize`). The pushdown fast path below
        # reads each file with its own on-disk schema and never normalizes, so int32 in
        # one file and int64 in another come back as differently-typed batches that fail
        # to concatenate. Pushdown is a pure I/O optimization, so defer to the normalizing
        # base read; the engine's `Filter` still applies the predicate.
        if self._schema_mode != "strict":
            return super().read(projection)
        if predicate is not None:
            # Selective scan: try the native filtered reader first (row-group + page-index
            # pruning in Rust), then PyArrow's `filters=`, then an unfiltered read. Each
            # step down reads more rows and none of them changes the answer.
            native = self._native_read_filtered(projection, predicate)
            if native is not None:
                return native
        pa_filter = self._pa_filter(predicate)
        if pa_filter is None:
            batched = self._native_read_many(projection)
            return batched if batched is not None else super().read(projection)
        return self._pyarrow_read_filtered(projection, pa_filter)

    def _pyarrow_read_filtered(
        self, projection: list[str] | None, pa_filter: Any
    ) -> list[pa.RecordBatch]:
        """Every file read through pyarrow's `filters=`, concurrently, in file order.

        The last stop before an unfiltered read, reached when the native reader declined the
        predicate (a temporal literal) or failed. This filters *exactly* rather than pruning,
        which is equally correct — the engine's `Filter` is idempotent over filtered input.
        """
        files = self._files()

        def _read_one(f: str) -> list[pa.RecordBatch]:
            table = self._read_table(f, projection, pa_filter)
            return list(self._normalize(table.to_batches(), projection, f))

        try:
            if len(files) <= 1:
                return _read_one(files[0]) if files else []
            # Read files concurrently: Parquet decode + filtering run in the C++ layer
            # with the GIL released, so a 100-file source no longer opens and filters one
            # file at a time (the serial loop left 95 of 96 cores idle). Order preserved.
            from concurrent.futures import ThreadPoolExecutor

            out: list[pa.RecordBatch] = []
            with ThreadPoolExecutor(max_workers=self._read_concurrency(len(files))) as pool:
                for batches in pool.map(_read_one, files):  # order preserved
                    out.extend(batches)
            return out
        except SchemaError:
            # A file that disagrees with the source's declared schema is a fact about the
            # data, not about this reader — falling back would re-read every file only to
            # raise the identical error from the base path. Surface it now.
            raise
        except Exception:
            # A filter the reader can't bind (e.g. a type it lacks a kernel for) must
            # never fail the query — the engine keeps the Filter operator, so an
            # unfiltered read is always correct, just reads more rows.
            return super().read(projection)

    def iter_batches(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> Iterator[pa.RecordBatch]:
        """Stream the files with row-group pruning, in bounded memory.

        Examples:
            .. doctest::

                >>> from batcher.io import ParquetSource  # doctest: +SKIP
                >>> src = ParquetSource("s3://bucket/lineitem/")  # doctest: +SKIP
                >>> next(src.iter_batches(["l_orderkey"])).num_rows  # doctest: +SKIP
                16384

        Args:
            projection: Columns to read. All columns when omitted.
            predicate: A predicate-IR filter. On any failure to bind it (a remote
                filesystem the dataset API can't resolve here) the stream falls
                back to unfiltered — the engine re-filters.

        Returns:
            An iterator over the surviving batches, in file order.
        """
        # A schema-evolving read reconciles each file to the unified schema in the base
        # `iter_batches` (via `_normalize`); the dataset pushdown path below reads files
        # with their own schemas and never normalizes, so differently-typed files yield
        # mismatched batches. Defer to the normalizing base stream — the engine's `Filter`
        # still applies the predicate, so this is only a lost I/O optimization.
        if self._schema_mode != "strict":
            yield from super().iter_batches(projection)
            return
        pa_filter = self._pa_filter(predicate)
        if pa_filter is None:
            yield from super().iter_batches(projection)
            return
        # Stream with row-group pruning + filtering via pyarrow.dataset over the
        # local files; on any failure (e.g. a remote filesystem the dataset can't
        # resolve here) fall back to unfiltered streaming — the engine re-filters.
        try:
            import pyarrow.dataset as pads

            dataset = pads.dataset(self._files(), format="parquet")
        except Exception:
            yield from super().iter_batches(projection)
            return
        yield from dataset.to_batches(columns=projection, filter=pa_filter)

    def _file_row_count(self, path: str) -> int | None:
        """The file's row count from its footer, cached per *version* of the file.

        Keyed on `(path, size, mtime)` rather than the path: a pipeline re-run overwrites
        its own output under the same deterministic name, and a path-keyed count then
        answers a `count()` with the previous file's total while `collect()` returns the
        new rows — a metadata shortcut contradicting the data it summarizes, with nothing
        in either result to reveal it. A file that cannot be stat-ed is counted uncached.

        On a miss the footer comes from the **shared** footer cache rather than a private
        read. Both caches are per file identity and both are filled from the same bytes, so
        reading privately meant a query that asked for a row count and then planned splits
        walked every footer in the dataset twice — the count pass filled a cache the split
        pass could not see, and threw away the metadata the split pass was about to re-fetch.
        On 4,096 local files that second walk was ~660 ms; on an object store it is a second
        round trip per file. The cheap int memo is kept in front of it because the footer
        cache is bounded by resident row-groups and may evict, and re-deriving a count that
        is already known should not cost a fetch.
        """
        identity = file_identity(path, self._fs)
        if identity is not None:
            hit = _ROW_COUNT_CACHE.get(identity)
            if hit is not None:
                return hit
        n = _parquet_footer(path, self._fs).num_rows
        if identity is not None:
            _ROW_COUNT_CACHE.put(identity, n)
        return n

    def _file_splits(
        self, path: str, target_size: int | None, predicate: dict | None = None
    ) -> list[Split]:
        # The row-group fast path re-resolves the filesystem from the bare path on the
        # worker (through a footer cache keyed on path only), so it cannot carry a
        # bring-your-own filesystem or `storage_options`. When the caller supplied either,
        # fall back to the whole-file `FileSplit`, which reconstructs the source via
        # `_reader_kwargs` and *does* carry them — trading finer sub-file granularity for
        # correct credentials on exactly the on-prem / custom-backend case that needs them.
        #
        # `on_error` rides the same fallback, for the same reason: a `RowGroupSplit` carries
        # no reader kwargs, so a tolerated read that stayed on the fast path would rebuild a
        # fail-fast reader on the worker. Planning already skips a file whose *footer* is
        # unreadable, but a footer can parse while a data page is truncated — that failure
        # surfaces mid-read on the worker, and only a `FileSplit` carries the policy that
        # survives it. Tolerance costs sub-file parallelism; silently dropping it would cost
        # the query.
        if (
            self._filesystem is not None
            or self._storage_options is not None
            or self._errors.mode != "raise"
        ):
            return [FileSplit(self.format_name, path, self._reader_kwargs())]
        return parquet_row_group_splits(path, target_size, predicate, self._fs)

    def statistics(self) -> SourceStatistics | None:
        """Footer-derived row count + per-column min/max/null, no data scan.

        This is what lets Kyber prune partitions and answer an unfiltered
        ``MIN``/``MAX`` before a single row is read.

        Examples:
            .. doctest::

                >>> from batcher.io import ParquetSource  # doctest: +SKIP
                >>> ParquetSource("s3://bucket/lineitem/").statistics().row_count  # doctest: +SKIP
                600037902

        Above the shared per-file-sweep ceiling this declines and falls back to the base's
        listing-derived byte size. The footer walk is one metadata round trip per file, and
        at a million files that is the whole query: twenty-odd minutes of driver time,
        before a task launches, to produce bounds for a scan that is already parallel a
        million ways. `splits()`, `row_count()`, and `_total_byte_size()` all refuse the
        same sweep at the same count; this was the one that still paid it.

        Returns:
            The source's row count and per-column bounds, or None if the footers
            cannot be read.
        """
        from batcher.io.stats import parquet_statistics

        if self._too_many_files_to_sweep():
            return super().statistics()
        try:
            return self._stats_apply(parquet_statistics(self._fs, self._files(), self.schema()))
        except Exception:
            return None
