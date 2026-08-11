"""`FileLayout` — how a write divides its rows into files, resolved wherever the rows are.

A write's file layout is expressed three ways, and only one of them is directly
actionable by a sink:

* `max_rows_per_file` — a row cap, which `FileSink.write_partitioned` takes as-is;
* `num_files` — split into exactly that many files;
* `target_bytes_per_file` — the small-files fix, sizing each file to about that many bytes.

The latter two can only become a row cap once the rows are in hand, because both need the
data's size. On the single-node path that happens on the driver after `_collect`. On the
distributed paths it has to happen **on the worker**, over the shard the worker is about to
write — the driver never sees those rows, which is the entire point of the streaming write.

Holding the three in one object is what lets both places resolve them with the same
arithmetic instead of the driver resolving them and the workers ignoring them, which is
what used to happen: every distributed write silently dropped the layout and wrote one
unbounded file per shard, so ``repartition(target_size_mb=128).write(out, distributed=True)``
produced whatever size the shard happened to be.
"""

from __future__ import annotations

from dataclasses import dataclass

from batcher._internal.mathx import ceil_div

__all__ = ["FileLayout"]


@dataclass(frozen=True, slots=True)
class FileLayout:
    """A write's file-sizing request, in whichever of the three forms the caller used.

    All-`None` is the default layout: one file per shard, no cap. Instances are frozen
    and picklable, so a driver ships one to every worker unchanged.

    Args:
        max_rows_per_file: Hard cap on the rows in one output file.
        num_files: Split the output into exactly this many files.
        target_bytes_per_file: Size each output file to about this many bytes.
    """

    max_rows_per_file: int | None = None
    num_files: int | None = None
    target_bytes_per_file: int | None = None

    @property
    def is_default(self) -> bool:
        """Whether this layout asks for nothing (one uncapped file per shard).

        Returns:
            True when no sizing option was set.
        """
        return (
            self.max_rows_per_file is None
            and self.num_files is None
            and self.target_bytes_per_file is None
        )

    def for_shard(self, index: int, shards: int) -> FileLayout:
        """This layout as seen by shard `index` of `shards`.

        Only `num_files` is global — it names a total across the whole write, so each
        shard gets its share of the file budget (the remainder spread over the first
        shards, so ``num_files=7`` over 3 shards is 3+2+2 rather than 3+3+3). A row cap
        and a byte target are already per-file, so they pass through untouched.

        Args:
            index: This shard's zero-based index.
            shards: How many shards the write is divided into.

        Returns:
            The layout this shard should resolve against its own rows.
        """
        if self.num_files is None or shards <= 1:
            return self
        share = self.num_files // shards + (1 if index < self.num_files % shards else 0)
        return FileLayout(
            max_rows_per_file=self.max_rows_per_file,
            num_files=max(1, share),
            target_bytes_per_file=self.target_bytes_per_file,
        )

    def rows_per_file(self, rows: int, nbytes: int) -> int | None:
        """The row cap this layout resolves to for a table of `rows` rows and `nbytes` bytes.

        An explicit `max_rows_per_file` always wins — it is the caller naming the answer
        directly. Otherwise `num_files` divides the rows evenly, and
        `target_bytes_per_file` scales the row count by the table's measured bytes-per-row.

        `nbytes` is the in-memory Arrow footprint, so a compressed format lands under the
        target rather than over it. Undershooting is the safe direction for a *target*: the
        alternative is guessing a compression ratio that varies per column and per codec.

        Args:
            rows: Rows in the table about to be written.
            nbytes: The table's in-memory size.

        Returns:
            A positive row cap, or None when this layout imposes none.
        """
        if self.max_rows_per_file is not None:
            return self.max_rows_per_file
        if rows <= 0:
            return None
        if self.num_files is not None:
            return max(1, ceil_div(rows, self.num_files))
        if self.target_bytes_per_file is not None and nbytes > 0:
            return max(1, rows * self.target_bytes_per_file // nbytes)
        return None
