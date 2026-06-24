"""Splits — independently-readable, picklable slices of a source.

A `Split` is the unit of distributed read parallelism. A source advertises its
splits via `Source.splits()`; each split carries only *locators* (a format name +
path, a set of row-group ids, …) — never data — so it serializes cheaply to a
remote worker that then reads just its slice directly from storage. The default
for a source that cannot subdivide is a single `WholeSourceSplit`, which
reproduces today's whole-source read.

Splits intentionally mirror the `Source` read surface (`schema`/`read`/
`iter_batches`/`row_count`/`identity`) so a worker treats a split exactly like a
source.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import pyarrow as pa

if TYPE_CHECKING:
    from batcher.io.source import Source

__all__ = [
    "FileSplit",
    "IpcFileSplit",
    "LineRangeSplit",
    "RowGroupSplit",
    "Split",
    "WholeSourceSplit",
    "fragment_index",
    "read_aligned_range",
]


# Per-process cache of ``key -> (dataset, {fragment_path: fragment})``. A worker
# lists/opens a dataset ONCE and reuses the path→fragment index across all the
# splits it reads, instead of re-listing the whole dataset on every read (which
# would be O(files^2) over a per-file-split read — catastrophic at scale).
_FRAGMENT_INDEX_CACHE: dict[Any, tuple[Any, dict[str, Any]]] = {}
_FRAGMENT_CACHE_MAX = 8


def fragment_index(key: Any, build_dataset: Any) -> tuple[Any, dict[str, Any]]:
    """Return ``(dataset, {fragment_path: fragment})`` for `key`, building once.

    `build_dataset` is a zero-arg callable returning a `pyarrow.dataset.Dataset`.
    The index is cached per process so each worker lists the dataset a single time
    regardless of how many of its fragments it reads. A small bound caps memory if
    a worker touches many distinct tables.
    """
    cached = _FRAGMENT_INDEX_CACHE.get(key)
    if cached is None:
        dataset = build_dataset()
        index = {frag.path: frag for frag in dataset.get_fragments()}
        if len(_FRAGMENT_INDEX_CACHE) >= _FRAGMENT_CACHE_MAX:
            _FRAGMENT_INDEX_CACHE.clear()
        cached = (dataset, index)
        _FRAGMENT_INDEX_CACHE[key] = cached
    return cached


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


@runtime_checkable
class Split(Protocol):
    """An independently-readable, picklable slice of a source."""

    def schema(self) -> pa.Schema: ...
    def read(self, projection: list[str] | None = None) -> list[pa.RecordBatch]: ...
    def iter_batches(self, projection: list[str] | None = None) -> Iterator[pa.RecordBatch]: ...
    def row_count(self) -> int | None: ...
    def identity(self) -> str: ...


@dataclass(frozen=True, slots=True)
class WholeSourceSplit:
    """A non-subdividable source read as a single split.

    Holds the source object itself, so it is only as picklable as that source
    (fine for in-memory / iterator sources, which carry their own data/closure).
    File and table sources never use this — they emit locator-only splits.
    """

    source: Source

    def schema(self) -> pa.Schema:
        return self.source.schema()

    def read(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> list[pa.RecordBatch]:
        from batcher.io.source import read_source

        return read_source(self.source, projection, predicate)

    def iter_batches(self, projection: list[str] | None = None) -> Iterator[pa.RecordBatch]:
        return self.source.iter_batches(projection)

    def row_count(self) -> int | None:
        return self.source.row_count()

    def identity(self) -> str:
        return self.source.identity()


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

    Carries only ``(format_name, path)``; `read` looks the format up in `SOURCES`
    and constructs a single-file reader. This is the default file-source split.
    """

    format_name: str
    path: str

    def _reader(self) -> Source:
        from batcher.io.formats.base import SOURCES

        return SOURCES.get(self.format_name)(self.path)

    def schema(self) -> pa.Schema:
        return self._reader().schema()

    def read(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> list[pa.RecordBatch]:
        from batcher.io.source import read_source

        return read_source(self._reader(), projection, predicate)

    def iter_batches(self, projection: list[str] | None = None) -> Iterator[pa.RecordBatch]:
        yield from self._reader().iter_batches(projection)

    def row_count(self) -> int | None:
        return self._reader().row_count()

    def identity(self) -> str:
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

    def _aligned_bytes(self) -> bytes:
        return read_aligned_range(self.path, self.start, self.end)

    def _table(self, projection: list[str] | None) -> pa.Table:
        import io

        import pyarrow.json as pajson

        buf = self._aligned_bytes()
        if not buf.strip():
            from batcher.io.formats.base import SOURCES

            empty = SOURCES.get(self.format_name)(self.path).schema().empty_table()
            return empty.select(projection) if projection is not None else empty
        table = pajson.read_json(io.BytesIO(buf))
        return table.select(projection) if projection is not None else table

    def schema(self) -> pa.Schema:
        from batcher.io.formats.base import SOURCES

        return SOURCES.get(self.format_name)(self.path).schema()

    def read(self, projection: list[str] | None = None) -> list[pa.RecordBatch]:
        return self._table(projection).to_batches()

    def iter_batches(self, projection: list[str] | None = None) -> Iterator[pa.RecordBatch]:
        yield from self._table(projection).to_batches()

    def row_count(self) -> int | None:
        return None

    def identity(self) -> str:
        return f"{self.format_name}:{self.path}:{self.start}-{self.end}"


@dataclass(frozen=True, slots=True)
class RowGroupSplit:
    """A contiguous run of Parquet row-groups within one file.

    The finest distributed-read granularity for Parquet: a worker reads only its
    assigned row-groups (a single S3 range read), not the whole file. `rows` is the
    footer-derived row count captured when the split was built, so balancing the
    splits never re-opens the file just to count.
    """

    path: str
    row_groups: tuple[int, ...]
    rows: int | None = None

    def _file(self) -> Any:
        import pyarrow.parquet as pq

        from batcher.io.filesystem import resolve_filesystem

        fs = resolve_filesystem(self.path)
        return pq.ParquetFile(fs.open(self.path))

    def schema(self) -> pa.Schema:
        return self._file().schema_arrow

    def read(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> list[pa.RecordBatch]:
        table = self._file().read_row_groups(list(self.row_groups), columns=projection)
        if predicate is not None:
            from batcher.io.predicate import to_pyarrow_expression

            expr = to_pyarrow_expression(predicate)
            if expr is not None:
                table = table.filter(expr)  # reduce rows before the shuffle
        return table.to_batches()

    def iter_batches(self, projection: list[str] | None = None) -> Iterator[pa.RecordBatch]:
        yield from self._file().iter_batches(row_groups=list(self.row_groups), columns=projection)

    def row_count(self) -> int | None:
        if self.rows is not None:
            return self.rows
        meta = self._file().metadata
        return sum(meta.row_group(i).num_rows for i in self.row_groups)

    def identity(self) -> str:
        return f"parquet:{self.path}:rg{','.join(map(str, self.row_groups))}"
