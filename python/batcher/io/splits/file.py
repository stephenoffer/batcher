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
from typing import TYPE_CHECKING

import pyarrow as pa

if TYPE_CHECKING:
    from batcher.io.source import Source

__all__ = ["FileSplit", "IpcFileSplit", "LineRangeSplit", "read_aligned_range"]


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
        parse = pajson.ParseOptions(
            explicit_schema=schema, unexpected_field_behavior="ignore"
        )
        table = pajson.read_json(io.BytesIO(buf), parse_options=parse)
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
