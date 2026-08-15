"""`ParquetDatasetSource` — a Hive-partitioned Parquet directory tree, read at scale.

The PB-scale read path. Its splits are *distributed listing*: the driver enumerates
only the top-level ``col=val`` dirs (one cheap, non-recursive list) and each worker
lists only its own subtree, so the listing cost is O(subtree) per worker rather than
O(whole dataset) on the driver.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pyarrow as pa

from batcher._internal.logging import note_suppressed
from batcher.io.base._paths import hive_segment
from batcher.io.formats.base import SOURCES
from batcher.io.formats.structured.parquet.partitions import (
    date_typed_partitioning,
    partition_bounds,
    partitioning_arg,
    typed_partition_value,
)
from batcher.io.splits import Split, WholeSourceSplit
from batcher.io.stats.file_skipping import PARTITION_PREFIX, surviving_files

if TYPE_CHECKING:
    from batcher.plan.source_stats import SourceStatistics

__all__ = ["ParquetDatasetSource", "ParquetFragmentSplit", "PartitionDirSplit"]

# File count past which the driver stops sweeping every footer to build full statistics.
# Mirrors `FileSource`'s `_MAX_FOOTER_PLAN_FILES` (same env var): a footer sweep is O(files)
# object-store round trips on the driver, a good trade at hundreds of files and a
# catastrophic one at millions. Above it, the cheap exact `row_count()` still answers.
_MAX_FOOTER_PLAN_FILES = max(1, int(os.environ.get("BATCHER_MAX_FOOTER_PLAN_FILES", "10000")))


def _footer_row_counter(fs: Any) -> Any:
    """A ``path -> rows`` reader over the shared, version-keyed footer cache.

    `_parquet_footer` is the same helper the flat reader and the codec probe use, so a file
    whose footer was already fetched to plan a split costs nothing to count — and passing
    `fs` keeps it on the filesystem that listed the directory, which is what stops the probe
    from spending a `HEAD` per file to re-learn a size the listing already reported.
    """
    from batcher.io.splits.parquet import _parquet_footer

    def rows(path: str) -> int | None:
        try:
            return _parquet_footer(path, fs).num_rows
        except Exception as exc:
            note_suppressed("io", "read a footer for the row-count sample", exc)
            return None

    return rows


@dataclass(frozen=True, slots=True)
class ParquetFragmentSplit:
    """One file of a partitioned Parquet dataset, read independently on a worker.

    Carries only locators (dataset root + partitioning + the fragment's path), so
    a worker reads just this file and recovers partition columns from the dataset
    schema — the whole dataset never materializes on the driver. Projection +
    predicate are pushed into the per-fragment read.
    """

    root: str
    partitioning: str | bytes
    file_path: str

    def _table(self, projection: list[str] | None, predicate: dict | None) -> pa.Table:
        import pyarrow.dataset as pads

        from batcher.io.splits import fragment_index

        # List the dataset once per worker (cached), then O(1) lookup — never
        # re-list per read (which would be O(files^2) over a per-file split).
        dataset, index = fragment_index(
            ("parquet", self.root, self.partitioning),
            lambda: pads.dataset(
                self.root, format="parquet", partitioning=partitioning_arg(self.partitioning)
            ),
        )
        flt = None
        if predicate is not None:
            from batcher.io.predicate import to_pyarrow_expression

            # With the schema, a literal the scanner has no kernel for (a string against a
            # `date32` partition key -- now the common case, since date keys are typed) is
            # *declined* rather than pushed and raised on. The engine's own Filter re-checks
            # every row, so declining costs pruning and never a row.
            flt = to_pyarrow_expression(predicate, dataset.schema)
        frag = index.get(self.file_path)
        if frag is not None:
            return frag.to_table(schema=dataset.schema, columns=projection, filter=flt)
        empty = dataset.schema.empty_table()
        return empty.select(projection) if projection is not None else empty

    def schema(self) -> pa.Schema:
        import pyarrow.dataset as pads

        return pads.dataset(
            self.root, format="parquet", partitioning=partitioning_arg(self.partitioning)
        ).schema

    def row_count(self) -> int | None:
        """Exact rows from this fragment's footer (no data scan)."""
        import pyarrow.parquet as pq

        from batcher.io.filesystem import resolve_filesystem

        try:
            fs = resolve_filesystem(self.file_path)
            with fs.open(self.file_path) as fh:
                return pq.ParquetFile(fh).metadata.num_rows
        except Exception:
            return None

    def read(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> list[pa.RecordBatch]:
        return self._table(projection, predicate).to_batches()

    def iter_batches(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> Iterator[pa.RecordBatch]:
        yield from self._table(projection, predicate).to_batches()

    def identity(self) -> str:
        return f"parquet_dataset:{self.root}:{self.file_path}"


@dataclass(frozen=True, slots=True)
class PartitionDirSplit:
    """One top-level partition directory of a Hive dataset, listed on the worker.

    Distributed listing: the driver enumerates only the top-level ``col=val`` dirs
    (one cheap, non-recursive list); each worker then lists *only its own subtree*
    and reads it, so the per-worker file-listing cost is O(subtree), not O(whole
    dataset). The top-level partition value lives in the dir name (not the data
    files), so it is appended, typed via the dataset schema carried on the split.
    """

    #: Asking this split for its row count re-lists its subtree and opens every footer in
    #: it, so the driver's split-assignment weighing must **not** ask (it would sweep the
    #: whole table one partition at a time, which is what this split exists to avoid). Read
    #: by `dist.executors.partition_io.assignment._weight`; the split still answers exactly
    #: when a caller genuinely wants the count.
    row_count_needs_a_sweep = True

    subdir: str
    partitioning: str | bytes
    part_name: str
    part_value: str
    dataset_schema: pa.Schema

    @property
    def clustering_columns(self) -> tuple[str, ...]:
        """The top-level partition column, whose value is constant across this whole subtree.

        Read by `io.splits.clustering`, which turns "constant within a split" into "resident
        on one worker" for a whole split set — the guarantee that lets a consumer grouping on
        this column skip its shuffle.

        The *top-level* column is the whole guarantee, not part of one. A nested
        ``year=/month=`` tree emits one split per year, so grouping by ``(year, month)`` is
        still covered — those groups sit inside ``year`` groups — while grouping by ``month``
        alone is not co-located and never can be, because ``month=1`` lives under every year.
        Splitting per leaf directory would not change that; it would only make a split's value
        ``(year, month)``, which ``month`` alone still straddles.
        """
        return (self.part_name,)

    @property
    def clustering_value(self) -> tuple[Any, ...]:
        """This subtree's partition value, typed, so distinctness is judged on values.

        Typed rather than raw because the raw form is a directory name: ``x=01`` and ``x=1``
        are two names for one integer, and comparing the names would call them distinct
        partitions of the same value — exactly the over-claim this check exists to refuse.
        Falls back to the raw string only when the value cannot be typed, which
        keeps the comparison conservative (two spellings then read as two partitions and
        the clustering is simply not claimed).
        """
        try:
            return (self._typed_value(),)
        except Exception:
            return (self.part_value,)

    def _typed_value(self) -> Any:
        return typed_partition_value(
            self.part_value, self.dataset_schema.field(self.part_name).type
        )

    def _table(self, projection: list[str] | None, predicate: dict | None) -> pa.Table:
        import pyarrow.dataset as pads

        from batcher.io.splits import fragment_index

        # List only this partition subtree (cached per worker), not the whole dataset.
        dataset, _index = fragment_index(
            ("pq_subdir", self.subdir, self.partitioning),
            lambda: pads.dataset(
                self.subdir, format="parquet", partitioning=partitioning_arg(self.partitioning)
            ),
        )
        want = list(projection) if projection is not None else list(self.dataset_schema.names)
        data_cols = [c for c in want if c != self.part_name]
        flt = None
        if predicate is not None:
            from batcher.io.predicate import to_pyarrow_expression

            flt = to_pyarrow_expression(predicate, self.dataset_schema)
        # Push the filter into the subtree read (row-group pruning) when it doesn't
        # reference the top-level partition column the sub-dataset lacks.
        prefiltered = False
        try:
            table = dataset.to_table(columns=data_cols, filter=flt)
            prefiltered = flt is not None
        except Exception:
            table = dataset.to_table(columns=data_cols)
        if self.part_name in want:
            value = self._typed_value()
            target = self.dataset_schema.field(self.part_name).type
            table = table.append_column(
                self.part_name, pa.array([value] * table.num_rows, type=target)
            )
        table = table.select(want)
        if flt is not None and not prefiltered:
            table = table.filter(flt)
        return table

    def schema(self) -> pa.Schema:
        return self.dataset_schema

    def read(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> list[pa.RecordBatch]:
        return self._table(projection, predicate).to_batches()

    def iter_batches(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> Iterator[pa.RecordBatch]:
        yield from self._table(projection, predicate).to_batches()

    def row_count(self) -> int | None:
        """Exact rows in this partition subtree from footers (no data scan)."""
        import pyarrow.dataset as pads

        try:
            return pads.dataset(
                self.subdir, format="parquet", partitioning=partitioning_arg(self.partitioning)
            ).count_rows()
        except Exception:
            return None

    def identity(self) -> str:
        return f"parquet_dataset:{self.subdir}"


#: What counts as a partition directory, shared with `FileSource` so the reader that
#: *recovers* partition columns and the reader that must warn it is *dropping* them
#: cannot disagree about which layouts are partitioned.
_hive_segment = hive_segment


@SOURCES.register("parquet_dataset")
class ParquetDatasetSource:
    """A Hive-partitioned Parquet dataset read via `pyarrow.dataset`.

    Recursively discovers files under `path`, recovers partition columns from the
    directory layout (``col=val/``), and applies partition + row-group pruning.
    This reads the directories `write.parquet(..., partition_by=...)` produces.
    `splits()` emits one `ParquetFragmentSplit` per data file, so a distributed
    read fans the files across workers and never materializes the whole dataset on
    the driver.
    """

    # Predicate pushdown: Kyber's pushed predicate is translated to a pyarrow
    # dataset filter (partition + row-group + page pruning).
    supports_predicate = True

    __slots__ = (
        "_built",
        "_dirs",
        "_partitioning",
        "_path",
        "_resolved",
        "_schema_cache",
        "_schema_mode",
    )

    def __init__(
        self, path: str, *, partitioning: str = "hive", schema_mode: str = "strict"
    ) -> None:
        self._path = path
        self._partitioning = partitioning
        self._schema_mode = schema_mode
        self._resolved = False
        self._schema_cache: pa.Schema | None = None
        #: The discovered dataset, built at most once per source. See `_dataset`.
        self._built: Any = None
        #: The top-level ``col=value`` directories, listed at most once. See `_partition_dirs`.
        self._dirs: list[tuple[str, tuple[str, str]]] | None = None

    def __getstate__(self) -> dict[str, Any]:
        """Everything but the discovered dataset, which must not ride to a worker.

        The dataset *is* the whole file listing — the thing this class exists to avoid
        producing on the driver and shipping. A worker rebuilds (and caches) its own for the
        subtree it was given, so sending this one would move the cost rather than remove it.
        """
        return {slot: getattr(self, slot, None) for slot in self.__slots__ if slot != "_built"}

    def __setstate__(self, state: dict[str, Any]) -> None:
        for slot in self.__slots__:
            setattr(self, slot, state.get(slot))

    def _unified_schema(self, built: Any) -> pa.Schema | None:
        """Every column any fragment has, or None when discovery's schema already is that.

        `pyarrow.dataset` types the whole dataset from its *first* fragment, so a table that
        gained a column midway reads back without it — silently, since the rows are all
        there. That is tolerable on a read and not on a **compaction**, which writes the
        result back and deletes the files the column still lived in. Union mode is the
        caller saying the files legitimately disagree, so it costs one footer read per file
        to find out what they hold; the default mode does not pay it.

        Args:
            built: The dataset as discovery produced it.

        Returns:
            The unified schema, or None to keep discovery's.
        """
        if self._schema_mode != "union":
            return None
        try:
            schemas = [frag.physical_schema for frag in built.get_fragments()]
        except Exception:
            return None
        if not schemas:
            return None
        # Discovery's schema goes first so the partition columns it recovered survive the
        # union -- a fragment's *physical* schema has no partition field in it at all, and
        # unifying only those is how `schema_mode="union"` used to drop the partition
        # column on the way to keeping the evolved one.
        return pa.unify_schemas([built.schema, *schemas])

    def _dataset(self) -> Any:
        """The pyarrow dataset, with date-valued partition keys typed as dates.

        Discovery runs once and is then *replaced* rather than repeated: the promotion is
        decided from the dictionaries that first walk already collected, and only a tree
        that actually has a date-valued key pays for the second construction.

        The result is held for the source's lifetime, because building it is a **recursive
        listing of the whole tree** — the single most expensive thing this class does, and
        `O(files)` object-store round trips on the driver. Six methods reach for it
        (`schema`, `read`, `iter_batches`, `row_count`, `statistics`, `splits`), and a query
        that plans a scan touches several, so the same million-file walk was paid once per
        call. Measured on 1,200 local files: 72 ms per discovery against 0.7 ms for the
        directory listing `splits` plans from, and every one of those milliseconds is a
        network round trip on an object store.

        Memoizing it is safe for the reason `FileSource._files` gives for memoizing its
        listing: every input is fixed for the source's lifetime, and a source could not
        observe files appearing underneath it through this method today either.
        """
        if self._built is not None:
            return self._built
        self._built = self._discover()
        return self._built

    def _discover(self) -> Any:
        """Build the dataset, resolving date-valued partition keys and any unified schema."""
        import pyarrow.dataset as ds

        built = ds.dataset(
            self._path, format="parquet", partitioning=partitioning_arg(self._partitioning)
        )
        if not self._resolved:
            self._resolved = True
            promoted = date_typed_partitioning(built)
            if promoted is not None:
                self._partitioning = promoted
                built = ds.dataset(
                    self._path, format="parquet", partitioning=partitioning_arg(promoted)
                )
        unified = self._unified_schema(built)
        if unified is None:
            return built
        return ds.dataset(
            self._path,
            format="parquet",
            partitioning=partitioning_arg(self._partitioning),
            schema=unified,
        )

    def _pa_filter(self, predicate: dict | None) -> Any:
        """The pushed filter for `predicate`, typed against this dataset's own schema.

        The schema is what lets `to_pyarrow_expression` decline a comparison arrow has no
        kernel for instead of pushing it and having the scanner raise from inside the task.
        Passing it became load-bearing once date-valued partition keys started reading as
        `date32`: ``filter(col("day") == "2024-01-02")`` — the spelling that worked while
        the key was text, and the one SQL uses — is a `date32`-against-string comparison.

        Unlike the four wrappers this outlived, it does something: it binds the schema. The
        `predicate is None` guard it also carried is gone, because `to_pyarrow_expression`
        answers that itself now.
        """
        from batcher.io.predicate import to_pyarrow_expression

        return to_pyarrow_expression(predicate, self.schema())

    def schema(self) -> pa.Schema:
        if self._schema_cache is None:
            self._schema_cache = self._dataset().schema
        return self._schema_cache

    def read(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> list[pa.RecordBatch]:
        table = self._dataset().to_table(columns=projection, filter=self._pa_filter(predicate))
        return table.to_batches()

    def iter_batches(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> Iterator[pa.RecordBatch]:
        yield from self._dataset().to_batches(columns=projection, filter=self._pa_filter(predicate))

    def row_count(self) -> int | None:
        """The exact total rows, or None when counting them costs more than it is worth.

        `count_rows` opens a footer per data file, so past `_MAX_FOOTER_PLAN_FILES` it is the
        same O(files) driver sweep that `splits`, `statistics`, and every `FileSource` path
        already refuse — a million round trips before a task launches, to produce a number
        the planner uses to size a join. This is the one place on the partitioned reader that
        still paid it. Above the ceiling `statistics()` answers with a sampled estimate
        instead, and a `count()` that genuinely needs the exact figure reads the data.

        Returns:
            The exact row count, or None above the ceiling.
        """
        dataset = self._dataset()
        if len(self._file_paths(dataset)) > _MAX_FOOTER_PLAN_FILES:
            return None
        return dataset.count_rows()

    @staticmethod
    def _file_paths(dataset: Any) -> list[str]:
        """Every data file in the dataset, or an empty list if they cannot be enumerated.

        Discovery has already walked the tree by the time this is called (`_dataset` holds
        the result), so this is a pass over what it collected rather than a second listing.
        """
        try:
            return [frag.path for frag in dataset.get_fragments()]
        except Exception as exc:
            note_suppressed("io", "enumerate the dataset's files", exc)
            return []

    def statistics(self) -> SourceStatistics | None:
        """Full footer statistics for the partitioned tree — the same the flat reader gives.

        The partitioned reader is the PB-scale path, yet it surfaced only an exact row count
        while the flat `ParquetSource` mined the footers for per-column min/max/null counts,
        on-disk byte size, row-group count, and proven global sortedness. This closes that
        gap: it runs the *same* `parquet_statistics` extractor over the dataset's data files,
        so Kyber prunes partitions, sizes joins, and answers ``MIN``/``MAX``/``null_count``
        on a Hive tree exactly as it does on a single directory.

        The Hive partition columns live in the directory names rather than the file footers,
        so they carry no column bounds here but are recorded in `partition_keys` — which is
        what partition pruning keys on. Above `_MAX_FOOTER_PLAN_FILES` the footer sweep is
        skipped (the driver would stall reading millions of footers) and a **sampled** row
        count answers in its place, marked `exact_rows=False`. Reporting nothing there was
        the worse of the two: a table large enough to decline the sweep is exactly the one
        whose join order and memory envelope are worth getting right, and above the ceiling
        every one of those was sized on a default. Best-effort — any footer failure yields
        None.
        """
        from batcher.io.filesystem import resolve_filesystem
        from batcher.io.stats import parquet_statistics

        try:
            dataset = self._dataset()
        except Exception:
            return None
        files = self._file_paths(dataset)
        if not files:
            return None
        if len(files) > _MAX_FOOTER_PLAN_FILES:
            return self._sampled_statistics(files)
        try:
            stats = parquet_statistics(resolve_filesystem(self._path), files, dataset.schema)
        except Exception:
            return None
        if stats is None:
            return None
        part_keys = self._partition_key_names(dataset)
        if not part_keys:
            return stats
        import dataclasses

        return dataclasses.replace(
            stats,
            partition_keys=part_keys,
            columns={**stats.columns, **partition_bounds(self._partition_dirs(), dataset.schema)},
        )

    def _sampled_statistics(self, files: list[str]) -> SourceStatistics | None:
        """An advisory row count and byte size for a tree too large to sweep, or None.

        The same fixed-cost estimate `FileSource` uses above its ceiling
        (`estimate_rows_from_footer_sample`): sixty-four footers spread across the file list,
        scaled by the tree's total on-disk size, whether the tree holds twenty thousand files
        or ten million. Column bounds are *not* estimated — a sampled min/max would be a
        bound the data can fall outside, and every consumer of `columns` treats a bound as
        provable.

        Args:
            files: Every data file in the tree, as discovery reported them.

        Returns:
            The advisory statistics, or None when no estimate could be produced.
        """
        from batcher.io._concurrent import total_file_bytes
        from batcher.io.filesystem import resolve_filesystem
        from batcher.io.stats.row_estimate import estimate_rows_from_footer_sample
        from batcher.plan.source_stats import SourceStatistics

        fs = resolve_filesystem(self._path)
        try:
            byte_size = total_file_bytes(fs, files)
            rows = estimate_rows_from_footer_sample(
                fs, files, _footer_row_counter(fs), total_bytes=byte_size
            )
        except Exception as exc:
            note_suppressed("io", "estimate a large tree's row count", exc)
            return None
        if rows is None and byte_size is None:
            return None
        return SourceStatistics(
            row_count=rows,
            byte_size=byte_size,
            exact_rows=False,
            partition_keys=self._partition_key_names(self._dataset()),
        )

    @staticmethod
    def _partition_key_names(dataset: Any) -> tuple[str, ...]:
        """The Hive partition column names, from the dataset's partitioning schema."""
        try:
            schema = getattr(dataset.partitioning, "schema", None)
            return tuple(schema.names) if schema is not None else ()
        except Exception:
            return ()

    def identity(self) -> str:
        return f"parquet_dataset:{self._path}"

    def clustering_columns(self) -> tuple[str, ...]:
        """The columns this dataset's splits will hold constant, without enumerating them.

        The cheap precondition for `io.splits.clustering`: a consumer asking whether it can
        skip an exchange has to know before it plans a read, because planning one over a
        50,000-file dataset is seconds of driver time it would then throw away. This costs one
        already-memoized, non-recursive listing, and a flat dataset answers `()` from it.

        Named for what the splits declare rather than for the table's partitioning, because
        the two differ and confusing them is the trap. `SourceStatistics.partition_keys`
        reports every partition column a nested tree has (``year``, ``month``); this reports
        the one a *split* holds constant (``year``). A caller asking "can I skip the exchange"
        needs the second.

        A *necessary* condition only — that the layout exists, not that the split set delivers
        it. `io.splits.declared_clustering` still checks the splits themselves.

        Returns:
            The top-level Hive partition column, or an empty tuple for a flat dataset.
        """
        if any(ch in self._path for ch in "*?["):
            return ()  # a glob reads per-file splits, which record no partition value
        dirs = self._partition_dirs()
        return (dirs[0][1][0],) if dirs else ()

    def splits(
        self,
        target_size: int | None = None,  # noqa: ARG002
        predicate: dict | None = None,
    ) -> list[Split]:
        """Distributed-listing splits — never list the whole tree on the driver.

        For a Hive-partitioned directory, the driver enumerates only the top-level
        ``col=val`` dirs (one cheap list) and emits one `PartitionDirSplit` per dir;
        each worker lists only its own subtree. For a non-partitioned dataset it
        falls back to per-file `ParquetFragmentSplit`s.

        A pushed `predicate` prunes at *plan* time, which is the whole reason a table is
        partitioned: a dropped directory is never listed, never read, and never becomes a
        task. Without it a one-day query over a table with a directory per day still
        scheduled every day's directory, and the engine's `Filter` discarded the rest —
        correct, and 3,650x the work it needed to be at PB scale.

        Args:
            target_size: Unused — a partition directory is the unit this source splits on.
            predicate: The filter Kyber pushed to this scan, as its IR dictionary.

        Returns:
            The splits covering the surviving partitions exactly once.
        """
        if not any(ch in self._path for ch in "*?["):
            partition_dirs = self._partition_dirs()
            if partition_dirs:
                schema = self.schema()
                return [
                    PartitionDirSplit(d, self._partitioning, name, value, schema)
                    for d, (name, value) in self._surviving_dirs(partition_dirs, predicate, schema)
                ]
        # Flat dataset (or non-listable): per-file splits read each file directly.
        # `get_fragments(filter=…)` applies the dataset's own partition-expression pruning,
        # so a nested (`a=1/b=2`) tree reached through this branch drops its non-matching
        # fragments here rather than shipping them to a worker to return no rows.
        try:
            surviving = self._dataset().get_fragments(self._pa_filter(predicate))
            paths = [frag.path for frag in surviving]
        except Exception:
            try:
                paths = [frag.path for frag in self._dataset().get_fragments()]
            except Exception:
                return [WholeSourceSplit(self)]
        if not paths:
            return [WholeSourceSplit(self)]
        return [ParquetFragmentSplit(self._path, self._partitioning, p) for p in paths]

    def _partition_dirs(self) -> list[tuple[str, tuple[str, str]]]:
        """The top-level ``col=value`` directories, as ``(dir, (key, raw_value))``.

        One cheap, non-recursive listing, memoized for the source's lifetime because three
        callers want it — split planning, partition-column bounds, and the pruning that sits
        between them — and it is a network round trip on an object store.
        """
        from batcher.io.filesystem import resolve_filesystem

        if self._dirs is None:
            try:
                listed = resolve_filesystem(self._path).list_dirs(self._path)
            except Exception as exc:
                note_suppressed("io", "list the partition directories", exc)
                listed = []
            self._dirs = [(d, seg) for d in listed if (seg := _hive_segment(d)) is not None]
        return self._dirs

    def _surviving_dirs(
        self, partition_dirs: list[tuple[str, tuple[str, str]]], predicate: dict | None, schema: Any
    ) -> list[tuple[str, tuple[str, str]]]:
        """`partition_dirs` minus the ones `predicate` proves hold no matching row.

        The directory name records the partition column's value exactly, so the value is
        both its minimum and its maximum — which is the add-action manifest layout
        `io.stats.file_skipping` already prunes, vectorized, for the lakehouse connectors.
        Building that one-column manifest here reuses their evaluator rather than restating
        the three-valued logic, so a Hive tree and a Delta table prune a compound predicate
        identically, and an undecidable one keeps every directory in both.

        Args:
            partition_dirs: ``(dir, (key, raw_value))`` for each top-level partition dir.
            predicate: The pushed predicate IR, or None.
            schema: The dataset schema, which types the partition column.

        Returns:
            The surviving entries, in the input order. Every entry when nothing is provable.
        """
        if predicate is None or not partition_dirs:
            return partition_dirs
        key = partition_dirs[0][1][0]
        try:
            target = schema.field(key).type
            manifest = pa.table(
                {
                    "path": pa.array([d for d, _ in partition_dirs], pa.string()),
                    f"{PARTITION_PREFIX}{key}": pa.array(
                        [typed_partition_value(v, target) for _, (_, v) in partition_dirs], target
                    ),
                }
            )
        except Exception as exc:
            note_suppressed("io", "build the partition-pruning manifest", exc)
            return partition_dirs
        kept = surviving_files(predicate, manifest)
        if kept is None:
            return partition_dirs
        keep = set(kept)
        return [entry for entry in partition_dirs if entry[0] in keep]
