"""Byte-range splits for line-delimited text — what lets one huge log fan across workers.

`TextSource` and `LogSource` advertised one split per *file*, so a single 50 GB log was a
single task however many nodes the cluster had. Adding nodes did nothing, which is the
sharpest form of "does not scale": the file is the unit and there is only one of it.

The obstacle was never the bytes — NDJSON and CSV already range-split the same way — it is
the **`line_number` column**. It counts from the start of the file, so a range beginning at
byte 4 GB cannot know what its first line's number is without reading the 4 GB before it,
which is the whole cost the split exists to avoid.

So the range split is offered only when the scan does not ask for `line_number`, which the
planner can see: `io.source.plan_splits` passes Kyber's pushed projection to any `splits()`
that accepts one. A query selecting the line text (the common one — grep a log, count
matches, extract a field) range-splits and scales; a query that genuinely wants the line
numbers keeps the whole-file split and its exact answer. Neither silently gets the other.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import pyarrow as pa

from batcher.io.splits.base import Split
from batcher.io.splits.file import read_aligned_range

__all__ = ["TextRangeSplit", "line_range_splits"]


@dataclass(frozen=True, slots=True)
class TextRangeSplit:
    """A newline-aligned byte range of a text or log file, read without `line_number`.

    Owns every line whose *first* byte falls in ``[start, end)`` — a leading partial line
    belongs to the previous range and a trailing line crossing `end` is completed here — so
    concatenating a file's ranges reconstructs it exactly once. This is `read_aligned_range`'s
    contract, shared with the NDJSON and CSV range splits.

    `columns` is the projection the split was planned for, and it never contains
    `line_number`; see the module docstring for why that is a property of the split rather
    than of the read.
    """

    format_name: str
    path: str
    start: int
    end: int
    text_column: str
    columns: tuple[str, ...] = ()
    splitlines: bool = True
    kwargs: dict[str, object] = field(default_factory=dict)

    def _table(self, projection: list[str] | None) -> pa.Table:
        from batcher.io.base._lines import lines_of

        wanted = list(projection) if projection is not None else list(self.columns)
        if "line_number" in wanted:
            # Unreachable through the planner, which only builds this split for a projection
            # without it. Raising beats returning a plausible column: a line number counted
            # from the start of a *range* looks exactly like one counted from the start of
            # the file, and every row of every split but the first would be wrong.
            from batcher._internal.errors import PlanError

            raise PlanError(
                f"{self.format_name!r} byte-range split cannot produce 'line_number': it "
                f"counts from the start of the file, and this split starts at byte "
                f"{self.start}. This split should not have been planned for a scan that "
                f"reads 'line_number'."
            )
        lines = lines_of(
            read_aligned_range(self.path, self.start, self.end), splitlines=self.splitlines
        )
        arrays: list[pa.Array] = []
        names: list[str] = []
        for name in wanted:
            names.append(name)
            arrays.append(
                pa.repeat(pa.scalar(self.path, pa.string()), len(lines))
                if name == "path"
                else lines
            )
        return pa.Table.from_arrays(arrays, names=names)

    def schema(self) -> pa.Schema:
        """The schema this split produces: its planned columns, in order.

        Returns:
            The Arrow schema of the projected columns.
        """
        return pa.schema([(name, pa.string()) for name in self.columns])

    def read(self, projection: list[str] | None = None) -> list[pa.RecordBatch]:
        """Every batch of this byte range.

        Args:
            projection: Columns to produce; the split's planned columns when omitted.

        Returns:
            The range's batches, in file order.
        """
        return self._table(projection).to_batches()

    def iter_batches(self, projection: list[str] | None = None) -> Iterator[pa.RecordBatch]:
        """Stream this byte range's batches.

        Args:
            projection: Columns to produce; the split's planned columns when omitted.

        Yields:
            One `RecordBatch` per morsel of the range.
        """
        yield from self._table(projection).to_batches()

    def row_count(self) -> int | None:
        """The range's row count, which is not known without reading it.

        Returns:
            None — a byte range's line count needs the bytes.
        """
        return None

    def identity(self) -> str:
        """A stable identity for this range, for the worker's scan cache.

        Returns:
            A string naming the format, file and byte range.
        """
        return f"{self.format_name}:{self.path}:{self.start}-{self.end}"


def line_range_splits(
    format_name: str,
    path: str,
    size: int,
    chunk: int,
    *,
    text_column: str,
    columns: tuple[str, ...],
    splitlines: bool,
) -> list[Split]:
    """`path` as newline-aligned byte ranges of about `chunk` bytes each.

    Args:
        format_name: The registry name of the source being split.
        path: The file to range over.
        size: The file's size in bytes.
        chunk: Rough bytes per range.
        text_column: The name of the column holding the line text.
        columns: The projected columns each range will produce.
        splitlines: Line semantics — see `io.base._lines`.

    Returns:
        The ranges covering `path` exactly once, in order.
    """
    return [
        TextRangeSplit(
            format_name,
            path,
            start,
            min(start + chunk, size),
            text_column,
            columns,
            splitlines,
        )
        for start in range(0, size, chunk)
    ]
