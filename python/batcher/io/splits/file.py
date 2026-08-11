"""File-locator splits — a whole file, an IPC stream file, or a byte range of one.

`FileSplit` is the default a file source advertises: a worker rebuilds a single-file
reader from the format registry and reads that file alone. `IpcFileSplit` is the same
idea for a distributed stage's on-disk Arrow IPC result. `LineRangeSplit` goes finer
for line-delimited text, letting one huge NDJSON file fan across workers as
newline-aligned byte ranges.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pyarrow as pa

if TYPE_CHECKING:
    from batcher.io.source import Source

__all__ = [
    "FileSplit",
    "IpcFileSplit",
    "LineRangeSplit",
    "MultiFileSplit",
    "NormalizedFileSplit",
    "pack_files",
    "read_aligned_range",
]


def pack_files(sizes: list[int], target_bytes: int, min_runs: int) -> list[tuple[int, int]]:
    """Group adjacent files into ``[start, stop)`` runs of roughly `target_bytes` each.

    One split per file is the right unit when files are large and few. It is the wrong one
    when they are small and many: every split is a scheduled task, a serialized locator, and
    a worker round trip, so a directory of a million 4 KB files becomes a million tasks whose
    overhead dwarfs the bytes they move. Packing them by size is what makes that corpus read
    like the few gigabytes it actually is. This is the same lever as Spark's
    ``maxPartitionBytes`` over its ``FilePartition``.

    A file is never *divided* here — only grouped — so a run always holds at least one file
    however large it is, and a dataset of big files packs to one file per run, unchanged.

    `min_runs` is the floor that keeps packing from destroying parallelism, and it binds at
    ``min(min_runs, len(sizes))`` — never more runs than there are files, but never fewer
    than the files could have supplied. Grouping eight 10 MB files under a 128 MiB target
    would otherwise yield a single task and idle every core but one, and grouping *two* tiny
    files would halve a two-file read for no reason at all. The consequence worth stating:
    a dataset with no more files than the floor packs to exactly one file per run, so
    grouping only ever engages once files outnumber the parallelism available to read them
    — which is the only situation it was introduced for.

    Args:
        sizes: Each file's size in bytes, in the order the files will be read.
        target_bytes: Rough bytes to aim for per run.
        min_runs: The parallelism floor, capped at the file count.

    Returns:
        The ``[start, stop)`` index ranges covering `sizes` exactly once, in order.
    """
    if not sizes:
        return []
    floor = min(min_runs, len(sizes))
    runs = _pack(sizes, max(1, target_bytes))
    if len(runs) < floor:
        runs = _pack(sizes, max(1, sum(sizes) // floor))
    return runs


def _pack(sizes: list[int], target: int) -> list[tuple[int, int]]:
    """`sizes` as ``[start, stop)`` runs, each at least one file and about `target` bytes."""
    runs: list[tuple[int, int]] = []
    start = 0
    acc = 0
    for i, size in enumerate(sizes):
        # `i > start` keeps a run non-empty: a single file larger than the target is its
        # own run rather than being split, which this packing cannot do.
        if i > start and acc + size > target:
            runs.append((start, i))
            start, acc = i, 0
        acc += size
    runs.append((start, len(sizes)))
    return runs


def read_aligned_range(path: str, start: int, end: int) -> bytes:
    """Read the newline-aligned byte range ``[start, end)`` of a line-delimited file.

    Returns the bytes for every line whose *first* byte falls in ``[start, end)``:
    a leading partial line (owned by the previous range) is skipped, and a trailing
    line crossing ``end`` is completed. Concatenating all ranges of a file thus
    reconstructs it exactly once. Used by NDJSON and CSV byte-range splits.
    """
    from batcher.io.filesystem import resolve_filesystem

    fs = resolve_filesystem(path)
    with fs.open(path) as fh:
        if start == 0:
            real_start = 0
        else:
            fh.seek(start - 1)
            if fh.read(1) == b"\n":  # `start` is exactly a line boundary
                real_start = start
            else:
                fh.seek(start)
                real_start = start + len(fh.readline())  # skip the continuing line
        n = end - real_start
        if n <= 0:
            return b""
        fh.seek(real_start)
        data = fh.read(n)
        if data and not data.endswith(b"\n"):
            data += fh.readline()  # complete the line crossing `end`
        return data


@dataclass(frozen=True, slots=True)
class IpcFileSplit:
    """One whole Arrow IPC stream file, read locator-only by path.

    The unit a `MaterializedSource` advertises: a distributed-stage result that
    stayed on disk (one IPC file per reducer) is re-scanned shared-nothing, each
    worker reading only its own file directly — the intermediate is never collected
    back to the driver. `rows` is the exact count captured when the file was written,
    so balancing never re-opens it. Reads IPC via `pyarrow` directly so `io` stays
    free of any `dist` dependency.
    """

    path: str
    rows: int | None = None

    def schema(self) -> pa.Schema:
        with pa.OSFile(self.path, "rb") as src, pa.ipc.open_stream(src) as reader:
            return reader.schema

    def read(self, projection: list[str] | None = None) -> list[pa.RecordBatch]:
        with pa.OSFile(self.path, "rb") as src, pa.ipc.open_stream(src) as reader:
            batches = list(reader)
        if projection is not None:
            batches = [b.select(projection) for b in batches]
        return batches

    def iter_batches(self, projection: list[str] | None = None) -> Iterator[pa.RecordBatch]:
        with pa.OSFile(self.path, "rb") as src, pa.ipc.open_stream(src) as reader:
            for b in reader:
                yield b.select(projection) if projection is not None else b

    def row_count(self) -> int | None:
        return self.rows

    def identity(self) -> str:
        return f"ipc:{self.path}"


@dataclass(frozen=True, slots=True)
class FileSplit:
    """One whole file, reconstructed on the worker via the format registry.

    Carries ``(format_name, path, kwargs)``; `read` looks the format up in `SOURCES`
    and constructs a single-file reader as ``SOURCES.get(format_name)(path, **kwargs)``.
    `kwargs` are the source's non-path construction arguments (a protobuf `message_cls`, an
    Excel `sheet`, a point-cloud `columns`/`dtype`) — WITHOUT them a worker rebuilds a
    reader that either raises (a required arg is missing) or silently reads the wrong data
    (a defaulted `sheet`/stride). Empty for the common formats whose reader needs only the
    path. This is the default file-source split. `kwargs` must be picklable (it ships to the
    worker); the sources that set it carry only plain values / a class object.

    Examples:
        .. doctest::

            >>> from batcher.io import FileSplit  # doctest: +SKIP
            >>> split = FileSplit("parquet", "s3://bucket/part-00000.parquet")  # doctest: +SKIP
            >>> split.identity()  # doctest: +SKIP
            'parquet:s3://bucket/part-00000.parquet'
    """

    format_name: str
    path: str
    kwargs: dict[str, object] = field(default_factory=dict)

    def _reader(self) -> Source:
        from batcher.io.formats.base import SOURCES

        return SOURCES.get(self.format_name)(self.path, **self.kwargs)

    def schema(self) -> pa.Schema:
        """The file's schema, read by a freshly-rebuilt single-file reader.

        Examples:
            .. doctest::

                >>> from batcher.io import FileSplit  # doctest: +SKIP
                >>> FileSplit("parquet", "part-00000.parquet").schema().names  # doctest: +SKIP
                ['id', 'ts']

        Returns:
            The Arrow schema of the file.
        """
        return self._reader().schema()

    def read(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> list[pa.RecordBatch]:
        """Read the whole file, pushing `predicate` down where the format supports it.

        Examples:
            .. doctest::

                >>> from batcher.io import FileSplit  # doctest: +SKIP
                >>> len(FileSplit("parquet", "part-00000.parquet").read(["id"]))  # doctest: +SKIP
                1

        Args:
            projection: Columns to read. All columns when omitted.
            predicate: A filter the format may apply during the read. The engine
                re-checks it regardless, so ignoring it is still correct.

        Returns:
            The file's batches, materialized.
        """
        from batcher.io.source import read_source

        return read_source(self._reader(), projection, predicate)

    def iter_batches(self, projection: list[str] | None = None) -> Iterator[pa.RecordBatch]:
        """Stream the file batch by batch (the bounded-memory read path).

        Examples:
            .. doctest::

                >>> from batcher.io import FileSplit  # doctest: +SKIP
                >>> split = FileSplit("parquet", "part-00000.parquet")  # doctest: +SKIP
                >>> next(split.iter_batches()).num_rows  # doctest: +SKIP
                16384

        Args:
            projection: Columns to read. All columns when omitted.

        Returns:
            An iterator over the file's batches.
        """
        yield from self._reader().iter_batches(projection)

    def row_count(self) -> int | None:
        """The file's row count when the format knows it cheaply (a footer), else None.

        Examples:
            .. doctest::

                >>> from batcher.io import FileSplit  # doctest: +SKIP
                >>> FileSplit("parquet", "part-00000.parquet").row_count()  # doctest: +SKIP
                1000

        Returns:
            The row count, or None when counting would cost a data scan.
        """
        return self._reader().row_count()

    def identity(self) -> str:
        """The ``format:path`` key naming this file.

        Examples:
            .. doctest::

                >>> from batcher.io import FileSplit  # doctest: +SKIP
                >>> FileSplit("csv", "events.csv").identity()  # doctest: +SKIP
                'csv:events.csv'

        Returns:
            A key that distinguishes this file from its siblings.
        """
        return f"{self.format_name}:{self.path}"


@dataclass(frozen=True, slots=True)
class MultiFileSplit:
    """A **run of whole files** read by one task, rebuilt on the worker per file.

    The unit that lets a small-file corpus scale. `FileSplit` is one file per task, which
    is correct and is the wrong granularity once files outnumber workers by orders of
    magnitude: at a million 4 KB files it is a million scheduled tasks, a million pickled
    locators, and a million worker round trips to move four gigabytes. `pack_files` groups
    adjacent files by size and this reads a group, so the task count tracks the *bytes* in
    the dataset rather than the number of objects it happens to be stored in.

    Carries the same ``(format_name, kwargs)`` a `FileSplit` does and rebuilds a single-file
    reader per path through the registry, so it needs nothing from a format that a
    one-file split does not already need — every format that can be a `FileSplit` can be
    part of a `MultiFileSplit`.

    `rows` is the group's exact row count when the planner already knew it for free; it is
    `None` otherwise and is never computed by opening the files, which would reintroduce
    the per-file round trip the grouping exists to remove.
    """

    format_name: str
    paths: tuple[str, ...]
    kwargs: dict[str, object] = field(default_factory=dict)
    rows: int | None = None

    def _reader(self, path: str) -> Source:
        from batcher.io.formats.base import SOURCES

        return SOURCES.get(self.format_name)(path, **self.kwargs)

    def schema(self) -> pa.Schema:
        """The group's schema, from its first file.

        Returns:
            The Arrow schema every file in this group conforms to.
        """
        return self._reader(self.paths[0]).schema()

    def read(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> list[pa.RecordBatch]:
        """Read every file in the group, concurrently on a remote store.

        A group exists because its files are small, and a small remote file is almost all
        latency — so reading them one after another inside the task would trade the
        scheduler round trips away only to pay them again as serialized GETs.
        `read_each_file` owns that policy: concurrent for an object store, serial for local
        disk, where a read is a syscall and a pool costs more than it saves.

        Args:
            projection: Columns to read. All columns when omitted.
            predicate: A filter the format may apply during the read. The engine re-checks
                it regardless, so ignoring it is still correct.

        Returns:
            Every file's batches, in file order.
        """
        from batcher.io._concurrent import read_each_file
        from batcher.io.source import read_source

        per_file = read_each_file(
            None,
            list(self.paths),
            lambda _fs, path: read_source(self._reader(path), projection, predicate),
        )
        return [batch for batches in per_file for batch in batches]

    def iter_batches(self, projection: list[str] | None = None) -> Iterator[pa.RecordBatch]:
        """Stream the group's files one after another, in order.

        Deliberately serial, unlike `read`: this is the bounded-memory path, and reading
        several files at once here would hold several files' decoded batches at once, which
        is the thing the caller chose this method to avoid.

        Args:
            projection: Columns to read. All columns when omitted.

        Returns:
            An iterator over the group's batches, file by file.
        """
        for path in self.paths:
            yield from self._reader(path).iter_batches(projection)

    def row_count(self) -> int | None:
        """The group's exact rows when the planner captured them, else None.

        Returns:
            The row count, or None when it was not known without opening the files.
        """
        return self.rows

    def identity(self) -> str:
        """The ``format:first-path+count`` key naming this group.

        The first path and the file count identify it exactly: groups are disjoint runs of
        one ordered file list, so no two can begin at the same file. Naming every member
        would put a million paths in a statistics key.

        Returns:
            A key that distinguishes this group from the source's others.
        """
        return f"{self.format_name}:{self.paths[0]}+{len(self.paths)}"


@dataclass(frozen=True, slots=True)
class NormalizedFileSplit:
    """One whole file, reshaped on the worker to a schema unified across all of them.

    The split a **schema-evolving** read (`schema_mode="union"`/`"latest"`) advertises, one
    per file. Without it such a read collapses to a single `WholeSourceSplit` — because a
    plain `FileSplit` rebuilds a single-file reader that knows only its own file's schema,
    so it would skip normalization and emit batches that will not concatenate with its
    siblings'. That correctness argument is right, but the consequence was that a
    schema-evolving dataset of any size became exactly **one task**, on one worker, however
    large the cluster: the read lost all parallelism precisely where it is most needed.

    Carrying the driver-computed unified schema on the split resolves it. The driver already
    unified every file's schema to answer `schema()`, so shipping the result costs nothing
    extra, and each worker then normalizes its own file to that target — adding missing
    columns as nulls and casting promoted types — exactly as the whole-source read does. The
    result is identical; only the number of tasks it takes changes.
    """

    format_name: str
    path: str
    target: pa.Schema
    kwargs: dict[str, object] = field(default_factory=dict)

    def _reader(self) -> Source:
        from batcher.io.formats.base import SOURCES

        return SOURCES.get(self.format_name)(self.path, **self.kwargs)

    def _target(self, projection: list[str] | None) -> pa.Schema:
        if projection is None:
            return self.target
        return pa.schema([self.target.field(c) for c in projection])

    def _file_projection(self, projection: list[str] | None) -> list[str] | None:
        """`projection` narrowed to the columns this file actually has.

        A file predating a column addition simply does not hold it; asking its reader for
        that column is an error, so the request is trimmed here and `normalize_batch` fills
        the column back in with nulls.
        """
        if projection is None:
            return None
        present = set(self._reader().schema().names)
        return [c for c in projection if c in present]

    def schema(self) -> pa.Schema:
        """The unified schema this split's batches conform to, narrowed by `projection`.

        Returns:
            The Arrow schema every batch this split produces conforms to.
        """
        return self.target

    def _normalized(
        self, batches: Iterator[pa.RecordBatch], projection: list[str] | None
    ) -> Iterator[pa.RecordBatch]:
        from batcher.io.schema import normalize_batch

        target = self._target(projection)
        for batch in batches:
            yield normalize_batch(batch, target)

    def read(
        self,
        projection: list[str] | None = None,
        predicate: dict | None = None,  # noqa: ARG002 (the engine re-checks it regardless)
    ) -> list[pa.RecordBatch]:
        """Read the whole file and reshape it to the unified schema.

        Args:
            projection: Columns to read. All columns when omitted.
            predicate: Ignored; the engine's `Filter` re-checks every row regardless.

        Returns:
            The file's batches, conforming to the unified schema.
        """
        reader = self._reader()
        batches = reader.read(self._file_projection(projection))
        return list(self._normalized(iter(batches), projection))

    def iter_batches(self, projection: list[str] | None = None) -> Iterator[pa.RecordBatch]:
        """Stream the file, reshaping each batch to the unified schema.

        Args:
            projection: Columns to read. All columns when omitted.

        Returns:
            An iterator over the file's normalized batches.
        """
        reader = self._reader()
        yield from self._normalized(
            iter(reader.iter_batches(self._file_projection(projection))), projection
        )

    def row_count(self) -> int | None:
        """The file's row count when the format knows it cheaply, else None.

        Normalization only reshapes columns, so it never changes the row count.

        Returns:
            The row count, or None when counting would cost a data scan.
        """
        return self._reader().row_count()

    def identity(self) -> str:
        """The ``format:path`` key naming this file.

        Returns:
            A key that distinguishes this file from its siblings.
        """
        return f"{self.format_name}:{self.path}"


@dataclass(frozen=True, slots=True)
class LineRangeSplit:
    """A newline-aligned byte range of a line-delimited file (NDJSON).

    The split owns every line whose first byte falls in ``[start, end)``: it skips
    a leading partial line (owned by the previous split) and completes a trailing
    line that crosses ``end``, so concatenating all splits reconstructs the file
    exactly once. This lets a single huge NDJSON file fan across workers — each
    reads only its byte range, not the whole file.
    """

    format_name: str
    path: str
    start: int
    end: int
    #: The source's constructor keywords, carried for the same reason `FileSplit` carries
    #: them: a worker rebuilds the reader from the split alone, so anything omitted reverts
    #: to its default *on the distributed path only*. Only the whole-file `FileSplit`
    #: branches used to pass them, so a JSON file merely large enough to subdivide silently
    #: lost `on_error=`, `on_bad_lines=`, `filesystem=` and `storage_options=` — a tolerated
    #: read became fail-fast, and an explicitly configured store became whatever the
    #: worker's environment resolved.
    options: dict[str, Any] | None = None

    def _source(self):
        """The reader this range belongs to, rebuilt with the options it was given."""
        from batcher.io.formats.base import SOURCES

        return SOURCES.get(self.format_name)(self.path, **(self.options or {}))

    def _aligned_bytes(self) -> bytes:
        return read_aligned_range(self.path, self.start, self.end)

    def _table(self, projection: list[str] | None) -> pa.Table:
        import pyarrow.json as pajson

        schema = self.schema()
        buf = self._aligned_bytes()
        if not buf.strip():
            empty = schema.empty_table()
            return empty.select(projection) if projection is not None else empty
        # Force each range to the file's declared schema. `pyarrow.json.read_json`
        # infers types independently per call, so without this an all-integer range
        # of a column parses as int64 while a range that holds a float parses it as
        # double, and a field absent from one range is missing from that range's
        # schema entirely — the ranges of one file disagree with each other and with
        # the source schema, so their batches cannot concatenate. Pinning the
        # explicit schema (unioned over the whole file) makes every range parse to
        # the same schema the source advertises; `ignore` keeps a truly-unexpected
        # field from re-introducing per-range drift. Mirrors `CSVRangeSplit`.
        parse = pajson.ParseOptions(explicit_schema=schema, unexpected_field_behavior="ignore")
        from batcher.io.formats.semistructured.json_tolerance import read_json_records

        table = read_json_records(buf, parse, self._policy())
        return table.select(projection) if projection is not None else table

    def _policy(self):
        """This range's bad-record policy, so a stray line costs a line and not the range."""
        from batcher.io.base._bad_rows import bad_row_handler

        mode = str((self.options or {}).get("on_bad_lines", "error"))
        return bad_row_handler(mode, self.path, format_name=self.format_name)

    def schema(self) -> pa.Schema:
        return self._source().schema()

    def read(self, projection: list[str] | None = None) -> list[pa.RecordBatch]:
        return self._table(projection).to_batches()

    def iter_batches(self, projection: list[str] | None = None) -> Iterator[pa.RecordBatch]:
        yield from self._table(projection).to_batches()

    def row_count(self) -> int | None:
        return None

    def identity(self) -> str:
        return f"{self.format_name}:{self.path}:{self.start}-{self.end}"
