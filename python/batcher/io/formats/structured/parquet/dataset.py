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

from batcher.io.base._paths import hive_segment
from batcher.io.formats.base import SOURCES
from batcher.io.splits import Split, WholeSourceSplit

if TYPE_CHECKING:
    from batcher.plan.source_stats import SourceStatistics

__all__ = ["ParquetDatasetSource", "ParquetFragmentSplit", "PartitionDirSplit"]

# File count past which the driver stops sweeping every footer to build full statistics.
# Mirrors `FileSource`'s `_MAX_FOOTER_PLAN_FILES` (same env var): a footer sweep is O(files)
# object-store round trips on the driver, a good trade at hundreds of files and a
# catastrophic one at millions. Above it, the cheap exact `row_count()` still answers.
_MAX_FOOTER_PLAN_FILES = max(1, int(os.environ.get("BATCHER_MAX_FOOTER_PLAN_FILES", "10000")))


@dataclass(frozen=True, slots=True)
class ParquetFragmentSplit:
    """One file of a partitioned Parquet dataset, read independently on a worker.

    Carries only locators (dataset root + partitioning + the fragment's path), so
    a worker reads just this file and recovers partition columns from the dataset
    schema — the whole dataset never materializes on the driver. Projection +
    predicate are pushed into the per-fragment read.
    """

    root: str
    partitioning: str
    file_path: str

    def _table(self, projection: list[str] | None, predicate: dict | None) -> pa.Table:
        import pyarrow.dataset as pads

        from batcher.io.splits import fragment_index

        # List the dataset once per worker (cached), then O(1) lookup — never
        # re-list per read (which would be O(files^2) over a per-file split).
        dataset, index = fragment_index(
            ("parquet", self.root, self.partitioning),
            lambda: pads.dataset(self.root, format="parquet", partitioning=self.partitioning),
        )
        flt = None
        if predicate is not None:
            from batcher.io.predicate import to_pyarrow_expression

            flt = to_pyarrow_expression(predicate)
        frag = index.get(self.file_path)
        if frag is not None:
            return frag.to_table(schema=dataset.schema, columns=projection, filter=flt)
        empty = dataset.schema.empty_table()
        return empty.select(projection) if projection is not None else empty

    def schema(self) -> pa.Schema:
        import pyarrow.dataset as pads

        return pads.dataset(self.root, format="parquet", partitioning=self.partitioning).schema

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

    subdir: str
    partitioning: str
    part_name: str
    part_value: str
    dataset_schema: pa.Schema

    def _typed_value(self) -> Any:
        if self.part_value == "__HIVE_DEFAULT_PARTITION__":
            return None
        # `part_value` is the RAW directory-name segment, which the writer URL-encoded
        # (`_hive_str`: `quote(value, safe="")`, so `x/y` → `x%2Fy`). The `pyarrow.dataset`
        # read path URI-decodes segment values, so this worker-side path must too — else a
        # distributed read returns the encoded value (`x%2Fy`) where a single-node read
        # returns `x/y`, breaking the single-node == distributed invariant.
        from urllib.parse import unquote

        target = self.dataset_schema.field(self.part_name).type
        return pa.scalar(unquote(self.part_value), pa.string()).cast(target).as_py()

    def _table(self, projection: list[str] | None, predicate: dict | None) -> pa.Table:
        import pyarrow.dataset as pads

        from batcher.io.splits import fragment_index

        # List only this partition subtree (cached per worker), not the whole dataset.
        dataset, _index = fragment_index(
            ("pq_subdir", self.subdir, self.partitioning),
            lambda: pads.dataset(self.subdir, format="parquet", partitioning=self.partitioning),
        )
        want = list(projection) if projection is not None else list(self.dataset_schema.names)
        data_cols = [c for c in want if c != self.part_name]
        flt = None
        if predicate is not None:
            from batcher.io.predicate import to_pyarrow_expression

            flt = to_pyarrow_expression(predicate)
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
                self.subdir, format="parquet", partitioning=self.partitioning
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

    __slots__ = ("_partitioning", "_path", "_schema_cache")

    def __init__(self, path: str, *, partitioning: str = "hive") -> None:
        self._path = path
        self._partitioning = partitioning
        self._schema_cache: pa.Schema | None = None

    def _dataset(self) -> Any:
        import pyarrow.dataset as ds

        return ds.dataset(self._path, format="parquet", partitioning=self._partitioning)

    @staticmethod
    def _pa_filter(predicate: dict | None) -> Any:
        if predicate is None:
            return None
        from batcher.io.predicate import to_pyarrow_expression

        return to_pyarrow_expression(predicate)

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
        return self._dataset().count_rows()

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
        skipped (the driver would stall reading millions of footers); the exact
        `row_count()` still answers. Best-effort — any footer failure yields None.
        """
        from batcher.io.filesystem import resolve_filesystem
        from batcher.io.stats import parquet_statistics

        try:
            dataset = self._dataset()
            files = [frag.path for frag in dataset.get_fragments()]
        except Exception:
            return None
        if not files or len(files) > _MAX_FOOTER_PLAN_FILES:
            return None
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

        return dataclasses.replace(stats, partition_keys=part_keys)

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

    def splits(self, target_size: int | None = None) -> list[Split]:  # noqa: ARG002
        """Distributed-listing splits — never list the whole tree on the driver.

        For a Hive-partitioned directory, the driver enumerates only the top-level
        ``col=val`` dirs (one cheap list) and emits one `PartitionDirSplit` per dir;
        each worker lists only its own subtree. For a non-partitioned dataset it
        falls back to per-file `ParquetFragmentSplit`s.
        """
        from batcher.io.filesystem import resolve_filesystem

        if not any(ch in self._path for ch in "*?["):
            try:
                dirs = resolve_filesystem(self._path).list_dirs(self._path)
            except Exception:
                dirs = []
            partition_dirs = [(d, seg) for d in dirs if (seg := _hive_segment(d)) is not None]
            if partition_dirs:
                schema = self.schema()
                return [
                    PartitionDirSplit(d, self._partitioning, name, value, schema)
                    for d, (name, value) in partition_dirs
                ]
        # Flat dataset (or non-listable): per-file splits read each file directly.
        try:
            paths = [frag.path for frag in self._dataset().get_fragments()]
        except Exception:
            return [WholeSourceSplit(self)]
        if not paths:
            return [WholeSourceSplit(self)]
        return [ParquetFragmentSplit(self._path, self._partitioning, p) for p in paths]
