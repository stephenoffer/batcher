"""ORC format — lazy, projection-pushdown read + write via `pyarrow.orc`.

ORC support ships with core pyarrow (no extra), but the import is still deferred
so importing this module never forces the ORC reader to load. Reads expose
*stripe-level* splits — one `ORCStripeSplit` per ``(file, stripe)`` — so a
distributed read parallelizes within a file, reading only its assigned stripe via
``pyarrow.orc.ORCFile.read_stripe``. Row counts come from the footer (no scan).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache
from typing import IO, Any

import pyarrow as pa

from batcher._internal.optional import require
from batcher.io.base import FileSink, FileSource
from batcher.io.filesystem import resolve_filesystem
from batcher.io.formats.base import SINKS, SOURCES
from batcher.io.splits import Split
from batcher.io.stats.file_identity import file_identity
from batcher.plan.source_stats import SourceStatistics

__all__ = ["ORCSink", "ORCSource", "ORCStripeSplit"]


def _require_orc() -> Any:
    """Import and return `pyarrow.orc` or raise `BackendError`."""
    return require(
        "pyarrow.orc", feature="ORC support", provides="pyarrow built with ORC", extra="all"
    )


@dataclass(frozen=True, slots=True)
class _ORCFooter:
    """The footer facts split planning needs, detached from the file handle that read them.

    Deliberately *not* the `ORCFile` itself. An `ORCFile` owns an open stream, so caching one
    would pin a file descriptor per cached entry for the life of the process — the cache would
    leak handles rather than save round trips. These three scalars are everything the planner
    asks a footer for, and they outlive the handle safely.
    """

    nrows: int
    nstripes: int
    schema: pa.Schema


def _orc_footer(path: str) -> _ORCFooter:
    """`path`'s ORC footer, read once per *version* of the file and cached.

    ORC had no footer cache at all while Parquet had one, and the asymmetry was not
    cosmetic: `ORCStripeSplit` re-opened the file and re-parsed the footer on *every*
    method call — `schema()`, `row_count()`, each `read()` — so planning a 1,000-stripe
    file cost 1,000 footer round trips (~100 ms each on object storage) to answer
    questions whose answer is one immutable document.

    Keyed on the file's identity — `(path, size, mtime_ns)` — never on the path alone, for
    the reason `io.stats.file_identity` spells out: `FileSink` writes deterministic names,
    so a re-run overwrites its own output and a path-keyed footer then reports the
    *previous* file's stripe count and row count for the new bytes.

    Bounded (`lru_cache(maxsize=1024)`) because an unbounded dict here grows with every
    file the process has ever touched — a leak proportional to a long-lived worker's whole
    scan history, not to its working set. A file that cannot be stat-ed is read uncached
    rather than cached under a token that could not detect it changing.
    """
    identity = file_identity(path)
    if identity is None:
        return _read_orc_footer(path)
    return _orc_footer_cached(identity)


@lru_cache(maxsize=1024)
def _orc_footer_cached(identity: tuple[str, int, int]) -> _ORCFooter:
    """`_orc_footer` keyed on the file identity (see there)."""
    return _read_orc_footer(identity[0])


def _read_orc_footer(path: str) -> _ORCFooter:
    orc = _require_orc()
    fs = resolve_filesystem(path)
    with fs.open(path) as fh:
        reader = orc.ORCFile(fh)
        return _ORCFooter(nrows=reader.nrows, nstripes=reader.nstripes, schema=reader.schema)


@dataclass(frozen=True, slots=True)
class ORCStripeSplit:
    """One stripe of a single ORC file, read in isolation on a worker.

    Carries only ``(path, stripe)`` so it pickles cheaply; `read` reopens the file
    and pulls just that stripe via ``ORCFile.read_stripe``.

    `rows` is the stripe's row count when the footer *proves* one (see `row_count`), captured
    at split time so balancing never re-opens the file.
    """

    path: str
    stripe: int
    rows: int | None = None

    def _file(self) -> Any:
        """A fresh reader over the file's *data*.

        Metadata questions must not come through here — they go to `_orc_footer`, which is
        cached. This opens a stream because a stripe read has to.
        """
        orc = _require_orc()
        fs = resolve_filesystem(self.path)
        return orc.ORCFile(fs.open(self.path))

    def read(self, projection: list[str] | None = None) -> list[pa.RecordBatch]:
        batch = self._file().read_stripe(self.stripe, columns=projection)
        # `read_stripe(columns=...)` returns columns in the file's schema order, not
        # the requested order — unlike Parquet/CSV/Arrow. Re-select so a projection is
        # honored as an ordered list (a `select("c", "a")` pushed to the scan must
        # yield `c, a`).
        if projection is not None:
            batch = batch.select(projection)
        return [batch]

    def schema(self) -> pa.Schema:
        return _orc_footer(self.path).schema  # cached footer, not a re-open

    def iter_batches(self, projection: list[str] | None = None) -> Iterator[pa.RecordBatch]:
        yield from self.read(projection)

    def row_count(self) -> int | None:
        """The stripe's rows when the footer proves them, else None — never a data read.

        This used to be ``read_stripe(self.stripe).num_rows``, which **decodes the entire
        stripe** to learn a number. `_balance` calls `row_count()` on every split to bin-pack
        them, so planning a distributed ORC read decoded the whole table on the driver before
        a single task was dispatched — the dataset was read twice, the first time serially, to
        decide how to read it in parallel.

        pyarrow's ORC reader exposes `nstripes` and `nrows` but no *per-stripe* row count
        (`ORCFile.nstripe_statistics` is only a count of statistics blobs; there is no
        accessor for their contents), so for a multi-stripe file the honest answer is None.
        That is exactly what the `Split.row_count` contract asks for — "the row count, or None
        when counting would cost a data scan" — and `_balance` degrades to equal-weight splits,
        which for ORC's uniformly sized stripes is a good approximation anyway.

        Returning None here is deliberately *not* an estimate. An even division of `nrows`
        across `nstripes` would balance marginally better and would be a wrong answer to a
        question whose contract says exact-or-unknown; `count()` is answered exactly and
        separately from the footers by `ORCSource.statistics`.

        The single-stripe file is the case where the footer does prove it: that stripe holds
        every row, so `nrows` is its exact count.
        """
        if self.rows is not None:
            return self.rows
        footer = _orc_footer(self.path)
        return footer.nrows if footer.nstripes == 1 else None

    def identity(self) -> str:
        return f"orc:{self.path}:stripe{self.stripe}"


@SOURCES.register("orc")
class ORCSource(FileSource):
    """One or more ORC files (single file, directory, or glob)."""

    suffix = ".orc"
    format_name = "orc"
    # Predicate pushdown: a pushed predicate → a pyarrow.dataset ORC filter, which
    # prunes stripes via their column statistics.
    supports_predicate = True

    __slots__ = ()

    def _read_schema(self, fh: IO[Any]) -> pa.Schema:
        orc = _require_orc()
        return orc.ORCFile(fh).schema

    def _read_file(self, fh: IO[Any], projection: list[str] | None) -> list[pa.RecordBatch]:
        orc = _require_orc()
        table = orc.ORCFile(fh).read(columns=projection)
        # `ORCFile.read(columns=...)` returns columns in file order, not the requested
        # order (Parquet/CSV/Arrow all preserve it). Re-select so projection order is
        # honored — otherwise `select("c", "a")` pushed to an ORC scan yields `a, c`.
        if projection is not None:
            table = table.select(projection)
        return table.to_batches()

    def _iter_file(self, path: str, projection: list[str] | None) -> Iterator[pa.RecordBatch]:
        """Stream one ORC file a stripe at a time rather than reading it whole.

        A stripe is ORC's own unit of columnar storage — the same role a Parquet row group
        plays — so reading stripe by stripe holds one stripe rather than the whole decoded
        file, and needs no re-chunking.
        """
        orc = _require_orc()
        with self._fs.open(path) as fh:
            reader = orc.ORCFile(fh)
            for i in range(reader.nstripes):
                # `read_stripe` yields a RecordBatch (unlike `read`, which yields a Table).
                batch = reader.read_stripe(i, columns=projection)
                # It shares `read`'s file-order column behaviour, so the same re-select is
                # needed for projection order to survive.
                if projection is not None:
                    batch = batch.select(projection)
                yield batch

    @staticmethod
    def _pa_filter(predicate: dict | None) -> Any:
        if predicate is None:
            return None
        from batcher.io.predicate import to_pyarrow_expression

        return to_pyarrow_expression(predicate)

    def read(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> list[pa.RecordBatch]:
        flt = self._pa_filter(predicate)
        if flt is None:
            return super().read(projection)
        import pyarrow.dataset as pads

        dataset = pads.dataset(self._files(), format="orc")
        return dataset.to_table(columns=projection, filter=flt).to_batches()

    def iter_batches(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> Iterator[pa.RecordBatch]:
        flt = self._pa_filter(predicate)
        if flt is None:
            yield from super().iter_batches(projection)
            return
        import pyarrow.dataset as pads

        dataset = pads.dataset(self._files(), format="orc")
        yield from dataset.to_batches(columns=projection, filter=flt)

    def _file_row_count(self, path: str) -> int | None:
        return _orc_footer(path).nrows

    def _file_splits(
        self,
        path: str,
        target_size: int | None,  # noqa: ARG002
        # ORC stripe pruning is not wired here because pyarrow exposes no way to read a
        # stripe's column statistics. `ORCFile.nstripe_statistics` reports only *how many*
        # statistics blobs the footer holds; there is no accessor for their min/max, and the
        # dataset API's ORC `FileFragment` has neither `split_by_row_group` nor `statistics`
        # (unlike its Parquet counterpart). So there is nothing to hand `file_prune_mask`.
        # This is a genuine gap in the reader, not a missing call: the moment pyarrow exposes
        # stripe statistics, the fix is to shape them into the add-action layout and reuse
        # `io.stats.file_skipping`, exactly as `io/splits/parquet.py::_surviving_row_groups`
        # does — never a second pruning implementation. Dropping the predicate is meanwhile
        # sound: every stripe survives, and the engine's `Filter` re-checks every row.
        predicate: dict | None = None,  # noqa: ARG002 (see above — no stripe statistics exist)
    ) -> list[Split]:
        footer = _orc_footer(path)
        # A single-stripe file's one stripe holds every row, so the footer's `nrows` is that
        # split's exact count — carry it so balancing never opens the file. With more than one
        # stripe the per-stripe split is unknowable from the footer (see `row_count`).
        rows = footer.nrows if footer.nstripes == 1 else None
        return [ORCStripeSplit(path, i, rows) for i in range(footer.nstripes)]

    def statistics(self) -> SourceStatistics | None:
        """Exact row count from the ORC footers (no data scan)."""
        from batcher.io.stats import orc_statistics

        try:
            return orc_statistics(self._fs, self._files())
        except Exception:
            return None


@SINKS.register("orc")
class ORCSink(FileSink):
    """Write an ORC file."""

    suffix = ".orc"
    format_name = "orc"

    __slots__ = ("compression",)

    def __init__(self, compression: str = "zstd", **kwargs: Any) -> None:
        super().__init__(**kwargs)  # carries filesystem= / storage_options=
        self.compression = compression

    def _write_file(self, table: pa.Table, fh: IO[Any]) -> None:
        orc = _require_orc()
        orc.write_table(table, fh, compression=self.compression)

    def _open_stream_writer(self, fh: IO[Any], schema: pa.Schema) -> Any:  # noqa: ARG002 (ORCWriter infers the schema from the first write)
        # Incremental ORCWriter: `write_stream` appends one batch at a time so a
        # breaker-free read→transform→write never materializes the whole result
        # (bounded memory), instead of the base default that buffers one table.
        orc = _require_orc()
        return orc.ORCWriter(fh, compression=self.compression)

    def _write_batch(self, writer: Any, batch: pa.RecordBatch) -> None:
        writer.write(pa.Table.from_batches([batch]))

    def _close_stream_writer(self, writer: Any) -> None:
        writer.close()
