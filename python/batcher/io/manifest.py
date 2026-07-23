"""Write results — the manifest a sink returns and a commit consumes.

A `WrittenFile` records one physically-written data file; a `WriteManifest`
collects them for a whole write. The manifest is what makes distributed writes
mergeable: each worker returns its `WrittenFile`s and the driver concatenates
them (a commutative merge) into one manifest, then performs a single commit
(publishing a directory marker for file sinks, or an atomic transaction-log
commit for lakehouse sinks).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["WriteManifest", "WrittenFile"]


@dataclass(frozen=True, slots=True)
class WrittenFile:
    """One data file written by a sink.

    `partition_values` carries the Hive key values encoded in the file's directory
    path, and is empty for an unpartitioned write.

    `stats` carries the file's per-column bounds — ``num_records``, ``min_values``,
    ``max_values``, ``null_counts`` — collected by the worker that wrote the file,
    where the data is already in memory and they cost nothing. A transactional sink
    records them in its commit, which is what lets a *later* read skip this file from
    the log alone (`io.stats.file_skipping`). Empty when the sink does not collect
    them; the names are format-neutral and each sink maps them to its own log format.

    Examples:
        .. doctest::

            >>> from batcher.io import WrittenFile
            >>> WrittenFile(path="part-00000.parquet", rows=3, bytes=1024).rows
            3
    """

    path: str
    rows: int
    bytes: int
    partition_values: dict[str, Any] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class WriteManifest:
    """The set of files produced by a write, plus rolled-up totals.

    `schema` is the write's output schema, attached by the driver (which knows the
    plan's output type) so a transactional `commit` can register the files without
    re-deriving it. A data file does not carry it: a partitioned write stores its
    partition columns in the *path*, not the file, so a footer alone cannot reconstruct
    the table's type. `None` for sinks that need no schema at commit time.

    Examples:
        .. doctest::

            >>> from batcher.io import WriteManifest, WrittenFile
            >>> m = WriteManifest((WrittenFile("part-00000.parquet", 3, 1024),))
            >>> m.num_files
            1
    """

    files: tuple[WrittenFile, ...] = ()
    schema: Any | None = None

    @property
    def total_rows(self) -> int:
        """Rows across every file in the manifest.

        Examples:
            .. doctest::

                >>> from batcher.io import WriteManifest, WrittenFile
                >>> files = (WrittenFile("a", 3, 10), WrittenFile("b", 2, 8))
                >>> WriteManifest(files).total_rows
                5

        Returns:
            The summed row counts.
        """
        return sum(f.rows for f in self.files)

    @property
    def total_bytes(self) -> int:
        """Bytes on storage across every file in the manifest.

        Examples:
            .. doctest::

                >>> from batcher.io import WriteManifest, WrittenFile
                >>> files = (WrittenFile("a", 3, 10), WrittenFile("b", 2, 8))
                >>> WriteManifest(files).total_bytes
                18

        Returns:
            The summed file sizes.
        """
        return sum(f.bytes for f in self.files)

    @property
    def num_files(self) -> int:
        """How many files the write produced.

        Examples:
            .. doctest::

                >>> from batcher.io import WriteManifest, WrittenFile
                >>> WriteManifest((WrittenFile("a", 3, 10),)).num_files
                1

        Returns:
            The file count.
        """
        return len(self.files)

    def merge(self, other: WriteManifest) -> WriteManifest:
        """Combine two manifests (used to roll up distributed writer results).

        The merge is a concatenation, so it is associative and commutative: each
        worker returns its own manifest and the driver folds them in any order.

        Examples:
            .. doctest::

                >>> from batcher.io import WriteManifest, WrittenFile
                >>> a = WriteManifest((WrittenFile("a", 3, 10),))
                >>> b = WriteManifest((WrittenFile("b", 2, 8),))
                >>> a.merge(b).total_rows
                5

        Args:
            other: The manifest to fold in.

        Returns:
            A new manifest holding both manifests' files.
        """
        return WriteManifest(files=self.files + other.files, schema=self.schema or other.schema)
