"""`include_path=` — a file read that also says which file each row came from.

Spark's ``input_file_name()``, DuckDB's ``filename=true`` and Polars' ``include_file_paths``
all answer "which file did this row come from", and a file reader here could not: the
answer is lost the moment a reader's batches leave the file that produced them. This module
keeps it, for every file format at once, by reading **one file per split** and appending a
constant column naming that file.

Why a wrapper rather than an option threaded through `FileSource`: every read path a file
format has — the concurrent `read`, the read-ahead `iter_batches`, the row-group and
byte-range splits, the coalesced multi-file splits, the native Parquet reader — would each
have to carry the name to the place a batch is produced, and the coalesced paths do not know
it at all. Reading one file per split makes the attribution exact by construction, at the
cost of the sub-file granularity those paths add. The column is opt-in, so nothing that does
not ask for it pays that.

The column is dictionary-encoded: one dictionary entry per batch and a zero-filled index
array, so tagging costs nothing per row in Python. The engine's boundary decodes a dictionary
to its value type, so the relation reads it as a plain string (`plan.types.widen`).

Layer: `io`, neutral. Nothing here touches a row.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow as pa

from batcher._internal.errors import FormatError

if TYPE_CHECKING:
    from batcher.io.base.source import FileSource
    from batcher.io.splits import Split

__all__ = ["PathColumnSource", "PathTaggedFileSplit", "with_path_column"]

#: The column `include_path=True` adds, the name DuckDB's ``filename`` flag and Polars'
#: ``include_file_paths`` leave to the caller. A string names it explicitly.
DEFAULT_PATH_COLUMN = "path"

#: How a file's name is typed: one dictionary entry per batch, decoded by the engine.
_PATH_TYPE = pa.dictionary(pa.int32(), pa.string())


def with_path_column(source: Any, include_path: bool | str) -> Any:
    """`source`, reading one file per split and naming each row's file in a column.

    Args:
        source: The file source a reader built.
        include_path: ``True`` for a column named ``"path"``, or the column's name.

    Returns:
        `source` unchanged when `include_path` is falsy, else the tagging wrapper.

    Raises:
        FormatError: When `source` is not a file source, or the name is already a column.
    """
    if not include_path:
        return source
    from batcher.io.base.source import FileSource

    if not isinstance(source, FileSource):
        raise FormatError(
            f"include_path= names the file each row came from, so it applies to file "
            f"readers only; {type(source).__name__} does not read files. A Hive-partitioned "
            "Parquet directory is read through its partition-aware reader, which this "
            "option bypasses: pass format='parquet' with a glob to read it file by file."
        )
    column = DEFAULT_PATH_COLUMN if include_path is True else str(include_path)
    if column in source.schema().names:
        raise FormatError(
            f"include_path={include_path!r}: the data already has a column named "
            f"{column!r}. Pass include_path='<another name>' to name the file column."
        )
    return PathColumnSource(source, column)


def _tag(batch: pa.RecordBatch, column: str, path: str) -> pa.RecordBatch:
    """`batch` with a constant, dictionary-encoded `column` holding `path`."""
    indices = pa.array(np.zeros(batch.num_rows, dtype=np.int32))
    values = pa.DictionaryArray.from_arrays(indices, pa.array([path], pa.string()))
    return batch.append_column(pa.field(column, _PATH_TYPE), values)


def _split_projection(projection: list[str] | None, column: str) -> tuple[list[str] | None, bool]:
    """The projection to ask the file for, and whether the path column was requested."""
    if projection is None:
        return None, True
    return [c for c in projection if c != column], column in projection


@dataclass(frozen=True, slots=True)
class PathTaggedFileSplit:
    """One whole file, conformed to the source's schema, with its path appended.

    Rebuilt on a worker from the format name and reader keywords, the way `FileSplit` is,
    and conformed the way the whole-source read conforms each file: held to the declared
    schema in strict mode, reshaped to the unified one under schema evolution.
    """

    format_name: str
    path: str
    target: pa.Schema
    column: str
    strict: bool
    kwargs: dict[str, object] = field(default_factory=dict)

    def _reader(self) -> Any:
        from batcher.io.formats.base import SOURCES

        return SOURCES.get(self.format_name)(self.path, **self.kwargs)

    def schema(self) -> pa.Schema:
        """The source's data schema, plus the path column last.

        Returns:
            The Arrow schema every batch this split produces conforms to.
        """
        return self.target.append(pa.field(self.column, _PATH_TYPE))

    def _file_projection(self, reader: Any, wanted: list[str] | None) -> list[str] | None:
        """`wanted` narrowed to what this file holds, under schema evolution only."""
        if self.strict or wanted is None:
            return wanted
        try:
            present = set(reader.schema().names)
        except Exception:  # an unreadable file; its own reader decides whether to skip it
            return wanted
        return [c for c in wanted if c in present]

    def iter_batches(self, projection: list[str] | None = None) -> Iterator[pa.RecordBatch]:
        """Stream the file, conformed and tagged.

        Args:
            projection: Columns to produce, possibly including the path column.

        Returns:
            An iterator over the file's batches.
        """
        from batcher.io.schema import conform_batch, normalize_batch

        wanted, tagged = _split_projection(projection, self.column)
        if wanted == [] and self.target.names:
            # Only the path column (or nothing) was asked for, but the rows still have to be
            # counted, and a reader asked for no columns may report none. Read the narrowest
            # thing there is, one column, and drop it below.
            wanted = self.target.names[:1]
        target = (
            self.target if wanted is None else pa.schema([self.target.field(c) for c in wanted])
        )
        reshape = conform_batch if self.strict else normalize_batch
        reader = self._reader()
        for batch in reader.iter_batches(self._file_projection(reader, wanted)):
            out = reshape(batch, target, path=self.path)
            if tagged:
                out = _tag(out, self.column, self.path)
            yield out if projection is None else out.select(projection)

    def read(
        self,
        projection: list[str] | None = None,
        predicate: dict | None = None,  # noqa: ARG002 (the engine re-checks it regardless)
    ) -> list[pa.RecordBatch]:
        """Read the whole file, conformed and tagged.

        Args:
            projection: Columns to produce, possibly including the path column.
            predicate: Ignored; the engine's `Filter` re-checks every row regardless.

        Returns:
            The file's batches.
        """
        return list(self.iter_batches(projection))

    def row_count(self) -> int | None:
        """The file's row count when its format knows it cheaply, else None.

        Returns:
            The row count, or None when counting would cost a data scan.
        """
        return self._reader().row_count()

    def identity(self) -> str:
        """The ``format:path`` key naming this file, marked as carrying the path column.

        Returns:
            A key distinct from the same file read without the column.
        """
        return f"{self.format_name}:{self.path}+path_column={self.column}"


class PathColumnSource:
    """A file source read one file per split, each row naming the file it came from.

    Built by `with_path_column` for ``include_path=``. Everything the column does not
    change — the file list, the schema mode, the statistics, the identity of the data — is
    the wrapped source's. Declares no predicate pushdown: the engine filters, which is
    always correct, including on the path column itself.

    Args:
        inner: The file source to read.
        column: The name of the path column.
    """

    __slots__ = ("_column", "_inner")

    def __init__(self, inner: FileSource, column: str) -> None:
        self._inner = inner
        self._column = column

    def schema(self) -> pa.Schema:
        """The wrapped source's schema, plus the path column last.

        Returns:
            The Arrow schema every batch this source produces conforms to.
        """
        return self._inner.schema().append(pa.field(self._column, _PATH_TYPE))

    def _file_splits(self) -> list[PathTaggedFileSplit]:
        inner = self._inner
        target, kwargs = inner.schema(), inner._reader_kwargs()
        strict = inner._schema_mode == "strict"
        return [
            PathTaggedFileSplit(inner.format_name, f, target, self._column, strict, kwargs)
            for f in inner._files()
        ]

    def splits(self, target_size: int | None = None) -> list[Split]:  # noqa: ARG002
        """One split per file, or a single whole-source split for a capped read.

        Args:
            target_size: Unused: a file is the unit, since a split must name one file.

        Returns:
            The splits covering the source exactly once.
        """
        from batcher.io.splits import WholeSourceSplit

        if self._inner._n_rows is not None or not self._inner._files():
            return [WholeSourceSplit(self)]
        return list(self._file_splits())

    def iter_batches(self, projection: list[str] | None = None) -> Iterator[pa.RecordBatch]:
        """Stream every file in order, a bounded window of them decoding concurrently.

        Args:
            projection: Columns to produce, possibly including the path column.

        Returns:
            An iterator over every file's tagged batches, in file order.
        """
        from batcher.io.base._readahead import ordered_readahead
        from batcher.io.base.source import _ITER_READAHEAD_BYTES

        splits = {s.path: s for s in self._file_splits()}
        if not splits:
            return
        stream = ordered_readahead(
            list(splits),
            lambda f: splits[f].iter_batches(projection),
            depth=self._inner._iter_readahead_depth(len(splits)),
            max_bytes=_ITER_READAHEAD_BYTES,
        )
        yield from self._inner._cap(stream)

    def read(self, projection: list[str] | None = None) -> list[pa.RecordBatch]:
        """Read every file, tagged, in file order.

        Args:
            projection: Columns to produce, possibly including the path column.

        Returns:
            Every batch of every file.
        """
        return list(self.iter_batches(projection))

    def row_count(self) -> int | None:
        """The wrapped source's row count; the column adds no rows.

        Returns:
            The row count, or None when counting would cost a data scan.
        """
        return self._inner.row_count()

    def statistics(self) -> Any:
        """The wrapped source's statistics; the path column has none, so prunes nothing.

        Returns:
            The wrapped source's `SourceStatistics`, or None.
        """
        return self._inner.statistics()

    def corrupt_files(self) -> list[str]:
        """The files the wrapped source skipped under ``on_error="skip"``.

        Returns:
            The skipped paths.
        """
        return self._inner.corrupt_files()

    @property
    def node_local(self) -> bool:
        """Whether the files may be readable only by this node (see `FileSource`).

        Returns:
            The wrapped source's answer.
        """
        return self._inner.node_local

    def identity(self) -> str:
        """The wrapped source's identity, marked as carrying the path column.

        Returns:
            A key distinct from the same files read without the column.
        """
        return f"{self._inner.identity()}+path_column={self._column}"
