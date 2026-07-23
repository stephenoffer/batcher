"""Reading a Delta table as an unbounded stream via its Change Data Feed.

`DeltaStreamSource` is the `readStream` half of a medallion pipeline: each pass reads
only the commits made since the last one and advances a cursor, so chaining bronze →
silver → gold reprocesses nothing. The cursor is a Delta version, which makes the source
checkpointable — a restarted query resumes exactly where it stopped.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pyarrow as pa

from batcher._internal.errors import BackendError
from batcher.io.formats.base import SOURCES
from batcher.io.formats.lakehouse._arrow import normalize_engine_types
from batcher.io.formats.lakehouse.delta._snapshot import require_deltalake
from batcher.io.splits import Split, WholeSourceSplit

__all__ = ["DeltaStreamSource"]

# CDF metadata columns delta-rs adds to a change feed.
_CDF_META = ("_change_type", "_commit_version", "_commit_timestamp")


@SOURCES.register("delta_stream")
class DeltaStreamSource:
    """A Delta table read incrementally as an unbounded stream (Spark ``readStream``).

    Each `iter_batches` pass reads the Change Data Feed for every commit after the
    last-processed version and advances the cursor — so chaining medallion layers
    (bronze → silver → gold) reads only new commits. ``change_feed=False`` (default)
    yields appended rows in the table's own schema (insert changes, metadata columns
    dropped); ``change_feed=True`` yields the full CDC stream including
    ``_change_type``/``_commit_version``/``_commit_timestamp`` (updates/deletes too).
    Requires ``delta.enableChangeDataFeed = true`` on the table.

    Checkpointable: the read position is the Delta version, so a streaming query
    resumes exactly-once after a restart.
    """

    bounded = False

    __slots__ = ("_cdf", "_cursor", "_storage_options", "_table_uri")

    def __init__(
        self,
        table_uri: str,
        *,
        starting_version: int = 0,
        change_feed: bool = False,
        storage_options: dict[str, str] | None = None,
    ) -> None:
        self._table_uri = table_uri
        self._cdf = change_feed
        self._storage_options = storage_options
        self._cursor = starting_version - 1  # next read starts at starting_version

    def _delta_table(self) -> Any:
        return require_deltalake().DeltaTable(
            self._table_uri, storage_options=self._storage_options
        )

    def schema(self) -> pa.Schema:
        # delta-rs returns an Arrow C-interface (arro3) schema; adapt to pyarrow.
        base = pa.schema(self._delta_table().schema().to_arrow())
        if not self._cdf:
            return base
        extra = [
            pa.field("_change_type", pa.string()),
            pa.field("_commit_version", pa.int64()),
            pa.field("_commit_timestamp", pa.timestamp("us")),
        ]
        return pa.schema(list(base) + extra)

    def _latest_version(self) -> int:
        return int(self._delta_table().version())

    def snapshot_position(self) -> dict:
        return {"version": self._cursor}

    def seek(self, position: dict) -> None:
        self._cursor = int(position["version"])

    def row_count(self) -> int | None:
        return None

    def identity(self) -> str:
        return f"delta_stream:{self._table_uri}"

    def splits(self, target_size: int | None = None) -> list[Split]:  # noqa: ARG002
        return [WholeSourceSplit(self)]

    def read(self, projection: list[str] | None = None) -> list[pa.RecordBatch]:
        return list(self.iter_batches(projection))

    def _shape(self, batch: pa.RecordBatch, projection: list[str] | None) -> pa.RecordBatch | None:
        """One change-feed batch normalized, filtered to appends, and projected.

        The per-batch form of what used to be a whole-window pass. Append mode keeps only
        `insert` changes and presents the table's own schema; `change_feed=True` passes the
        CDC columns through. A batch left empty by the filter is dropped rather than
        yielded — a zero-row batch is not a change.
        """
        import pyarrow.compute as pc

        table = normalize_engine_types(pa.Table.from_batches([batch]))
        if not self._cdf:
            change_type = pc.cast(table.column("_change_type"), pa.string())
            table = table.filter(pc.equal(change_type, "insert")).drop(list(_CDF_META))
        if projection is not None:
            table = table.select(projection)
        if table.num_rows == 0:
            return None
        return table.combine_chunks().to_batches()[0]

    def iter_batches(self, projection: list[str] | None = None) -> Iterator[pa.RecordBatch]:
        latest = self._latest_version()
        start = self._cursor + 1
        if start > latest:
            return  # no new commits since the last pass
        try:
            reader = self._delta_table().load_cdf(starting_version=start, ending_version=latest)
            batches = pa.RecordBatchReader.from_stream(reader)
        except Exception as exc:
            raise BackendError(
                f"failed to read Delta change feed for {self._table_uri!r} "
                f"(is delta.enableChangeDataFeed set?): {exc}"
            ) from exc
        # Streamed batch by batch. This used to be `.read_all()` — the entire change-feed
        # window built into one table, filtered and projected whole, and only then
        # re-chunked into the batches it yielded. A window is *every commit since the last
        # pass*, so on a first run (`starting_version=0`) it is the table's whole history,
        # and on a resumed one it is however much accumulated while the query was down —
        # exactly the cases where memory is already tight. An unbounded source whose
        # micro-batch peak is the size of its backlog is not a stream.
        for batch in batches:
            shaped = self._shape(batch, projection)
            if shaped is not None:
                yield shaped
        # Advance ONLY after every batch has been handed to the consumer.
        #
        # This used to run before the first `yield`, which turned the cursor into a promise
        # the stream had not kept: `snapshot_position()` reported the window as consumed
        # while the batches were still inside an unstarted generator. A consumer that
        # checkpoints its position — which is the entire reason `snapshot_position`/`seek`
        # exist — and then fails partway through the drain resumed at `latest + 1` and
        # never saw the rest. Silent, unrecoverable, and worse the larger the window.
        #
        # Placing it here makes the stream at-least-once: a consumer that dies mid-drain
        # replays the whole window, which is the correct failure mode for a change feed and
        # the one every checkpointing consumer is already built to handle. Abandoning the
        # generator early likewise leaves the cursor where it was, by construction.
        self._cursor = latest
