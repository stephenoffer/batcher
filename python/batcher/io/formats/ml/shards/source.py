"""The relational view of a shard corpus — the same directory, read as rows.

`ShardReader` reads a corpus *by sample index*, which is what a trainer needs and no help at
all for the questions asked around a training run. This is the other half: a registered
source, so the corpus is an ordinary `Dataset`.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pyarrow as pa

from batcher.io.base._paths import normalize_path
from batcher.io.formats.base import SOURCES
from batcher.io.formats.ml.shards.index import ShardIndex, read_shard_index

__all__ = ["TrainingShardsSource"]


@SOURCES.register("training_shards")
class TrainingShardsSource:
    """Read a `write_shards` directory back as an ordinary relation.

    The return leg of the training corpus. `shard_stream_loader` reads a shard directory
    *by sample index*, which is what a trainer needs and no help at all for the questions
    asked around a training run — what is the class balance, which rows have a null label,
    does this corpus match the one the features were fitted on. Those are relational
    questions, and answering them meant either reaching past the public API or re-deriving
    the corpus from its source.

    So the same directory is also a registered source. It is deliberately thin, because the
    shards are plain Arrow IPC: reading one is `ArrowIPCSource`'s job, and each split simply
    names a shard for it. What this class adds is the *index* — the row count without a
    data read, the shard order that makes a scan reproduce the corpus order, and the schema
    the writer recorded.

    Examples:
        .. doctest::

            >>> import batcher as bt, os, tempfile
            >>> out = os.path.join(tempfile.mkdtemp(), "corpus")
            >>> _ = bt.from_pydict({"x": [1, 2, 3]}).ml.write_shards(out, rows_per_shard=2)
            >>> bt.read.training_shards(out).count()
            3
    """

    bounded = True

    __slots__ = ("_directory", "_filesystem", "_index", "_storage_options")

    def __init__(
        self,
        directory: Any,
        *,
        filesystem: object = None,
        storage_options: dict[str, str] | None = None,
    ) -> None:
        """Open the shard directory's index; the shards themselves are read on demand."""
        self._directory = normalize_path(directory, what="the shard directory")
        self._filesystem = filesystem
        self._storage_options = storage_options
        self._index = read_shard_index(self._directory)

    @property
    def index(self) -> ShardIndex:
        """The directory's `ShardIndex`."""
        return self._index

    def schema(self) -> pa.Schema:
        """The corpus schema, from the index where the writer recorded one."""
        if self._index.schema is not None:
            return self._index.schema
        return self._shard_source(0).schema()

    def row_count(self) -> int:
        """The exact row count, straight from the index — no data is read.

        Exact rather than estimated, which is the distinction that lets a terminal such as
        `Dataset.count` be answered from metadata instead of by executing a scan.
        """
        return self._index.total_rows

    def identity(self) -> str:
        """A stable key for this corpus, for learned statistics."""
        return f"training_shards:{self._directory}"

    def statistics(self) -> Any:
        """Row count and stored size from the index, with no scan.

        Returns:
            A `batcher.plan.SourceStatistics` with an exact row count, or ``None`` if the
            statistics contract cannot be imported.
        """
        from batcher.plan.source_stats import SourceStatistics

        return SourceStatistics(row_count=self._index.total_rows, exact_rows=True)

    def read(self, projection: list[str] | None = None) -> list[pa.RecordBatch]:
        """Read the whole corpus, in shard order.

        Args:
            projection: Columns the scan must produce; all of them when omitted.

        Returns:
            Every batch of the corpus, in the order it was written.
        """
        return list(self.iter_batches(projection))

    def iter_batches(self, projection: list[str] | None = None) -> Iterator[pa.RecordBatch]:
        """Yield the corpus a shard at a time, so only one shard is ever resident.

        Args:
            projection: Columns the scan must produce; all of them when omitted.

        Yields:
            The corpus's record batches, in the order it was written.
        """
        for shard in range(self._index.shard_count):
            yield from self._shard_source(shard).iter_batches(projection)

    def splits(self, target_size: int | None = None) -> list[Any]:  # noqa: ARG002
        """One split per shard — the granularity the corpus was written at.

        A shard is already the writer's unit of work and is sized by `rows_per_shard`, so it
        is the natural read task too. Each split names the shard for the registered ``arrow``
        source rather than for this class: a shard is a plain Arrow IPC file, and pointing a
        worker at the whole directory to read one of its shards would have it re-read the
        index and re-derive an offset it was already told.

        `target_size` is ignored, and unusually it is right to ignore it. A shard's size is
        fixed when the corpus is written, and repacking would need every shard's byte size —
        one stat call per shard, which on an object store is the whole point of having an
        index instead. `rows_per_shard` at write time is the knob that sets read
        parallelism here.

        Args:
            target_size: Ignored, per above.

        Returns:
            The splits covering the corpus exactly once, in shard order.
        """
        from batcher.io.splits import FileSplit

        kwargs: dict[str, object] = {}
        if self._storage_options:
            # Forwarded so a worker resolves the same credentials the caller did. The
            # `filesystem` object itself is deliberately not: it need not be picklable, and
            # `storage_options` is the spelling that survives the trip to a worker.
            kwargs["storage_options"] = self._storage_options
        return [
            FileSplit("arrow", self._index.shard_path(i), dict(kwargs))
            for i in range(self._index.shard_count)
        ]

    def _shard_source(self, shard: int) -> Any:
        """The registered ``arrow`` source over one shard."""
        return SOURCES.get("arrow")(
            self._index.shard_path(shard),
            filesystem=self._filesystem,
            storage_options=self._storage_options,
        )
