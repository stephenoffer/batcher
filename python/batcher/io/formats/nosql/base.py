"""Shared shape for NoSQL / operational-store scan sources.

These stores (MongoDB, Cassandra, DynamoDB, Redis, …) are row-based key/value or
document engines with no Arrow-native file layout. Reading them is the same
recipe every time: open a per-worker connection from serialized connection
kwargs, enumerate the store's natural parallel unit (a token range, a scan
segment, a query offset window — never a live connection), fetch each unit's rows,
and assemble Arrow at *batch* granularity (one `pa.RecordBatch` per chunk of
rows). `ScanSource` captures that recipe so each concrete connector overrides
only two things: how to enumerate partitions and how to fetch one partition.

The connection kwargs are stored verbatim and **never logged** — they carry
credentials. A `_ScanSplit` is a frozen, picklable value object that holds only
the connector class, the (never-logged) connection kwargs, and the opaque
partition locator; the worker reconstructs the connector from those and fetches
just its partition. Missing optional drivers raise `BackendError` with an
actionable ``pip install 'batcher-engine[<extra>]'`` hint.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import pyarrow as pa

from batcher._internal.errors import BackendError
from batcher.config import active_config

__all__ = ["ScanSource", "offset_windows", "rows_to_batches"]

# An opaque, picklable partition locator (token range, segment id, offset, …).
# It is connector-defined; the base treats it as a black box it round-trips to
# ``_read_partition`` on the worker.
PartitionLocator = Any


def require_driver(module: str, extra: str) -> Any:
    """Import `module` or raise `BackendError` pointing at the right extra.

    Args:
        module: The dotted import path of the optional driver (e.g. ``"pymongo"``).
        extra: The Batcher extras key that installs it (e.g. ``"mongo"``).

    Returns:
        The imported module object.

    Raises:
        BackendError: If the driver is not installed.
    """
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise BackendError(
            f"{module} is required for this source: pip install 'batcher-engine[{extra}]'"
        ) from exc


def rows_to_batches(
    rows: Iterator[dict[str, Any]],
    schema: pa.Schema | None = None,
    batch_rows: int | None = None,
) -> Iterator[pa.RecordBatch]:
    """Assemble an iterator of row dicts into Arrow batches of `batch_rows`.

    This is the row→Arrow bridge for drivers with no Arrow-native reader: rows
    accumulate into a buffer and are converted in bulk (`pa.RecordBatch.from_pylist`)
    once per batch — never per row in the hot path of an operator, only at the
    source boundary where the data is intrinsically row-shaped.

    Args:
        rows: An iterator of row dictionaries (column name → scalar value).
        schema: Optional Arrow schema to coerce each batch to; inferred if None.
        batch_rows: Target row count per emitted batch; defaults to the engine's
            configured morsel size (`ExecutionConfig.morsel_rows`) so source batches
            match downstream operator granularity.

    Yields:
        `pa.RecordBatch`, each holding up to `batch_rows` rows.
    """
    if batch_rows is None:
        batch_rows = active_config().execution.morsel_rows
    buffer: list[dict[str, Any]] = []
    for row in rows:
        buffer.append(row)
        if len(buffer) >= batch_rows:
            yield pa.RecordBatch.from_pylist(buffer, schema=schema)
            buffer = []
    if buffer:
        yield pa.RecordBatch.from_pylist(buffer, schema=schema)


@dataclass(frozen=True, slots=True)
class _ScanSplit:
    """A picklable, independently-readable slice of a `ScanSource`.

    Carries only locators: the connector class, the (never-logged) connection
    kwargs needed to rebuild a connection on the worker, and the opaque partition
    locator. It holds **no live connection** — the worker reconstructs the
    connector and calls `_read_partition` for just this partition.
    """

    source_cls: type[ScanSource]
    conn_kwargs: dict[str, Any]
    partition: PartitionLocator
    identity_prefix: str
    predicate: dict | None = None

    def _source(self) -> ScanSource:
        return self.source_cls(**self.conn_kwargs)

    def schema(self) -> pa.Schema:
        return self._source().schema()

    def read(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> list[pa.RecordBatch]:
        return list(self.iter_batches(projection, predicate))

    def iter_batches(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> Iterator[pa.RecordBatch]:
        # The predicate Kyber pushed at *plan* time is baked into the split (`self.predicate`);
        # `predicate` is the one the distributed reader hands us at *read* time. They are the
        # same filter arriving by two routes, so either will do — but one of them must arrive,
        # or the worker fetches the whole store and the engine's `Filter` throws it away.
        pushed = predicate if predicate is not None else self.predicate
        yield from self._source()._read_partition(self.partition, projection, pushed)

    def row_count(self) -> int | None:
        return None

    def identity(self) -> str:
        return f"{self.identity_prefix}:part={self.partition!r}"


def offset_windows(total: int | None, segments: int) -> list[tuple[int, int]]:
    """``(offset, limit)`` windows that are a disjoint **and exhaustive** cover of a result.

    An offset window is the only way to split a store that has no native shard/token/segment
    primitive — Couchbase's SQL++ and Neo4j's Cypher both fall here. It is easy to get subtly,
    silently wrong, and it was:

        return [(i * _WINDOW_ROWS, _WINDOW_ROWS) for i in range(segments)]

    With a fixed window size, `segments` windows cover only ``segments * _WINDOW_ROWS`` rows.
    Every row past that was **dropped, with no error**: eight segments over a billion-row
    collection returned 800,000 rows and reported success. Turning on parallelism — the thing
    you do *because* the data is large — was what silently truncated it.

    Two properties make a set of windows a cover, and both are enforced here:

    * **Exhaustive** — the final window is *unbounded* (``limit == 0``), so the cover runs to
      the end of the result no matter how far off `total` is, including rows written after it
      was measured. This is what makes the function safe even when the count is a lie.
    * **Disjoint** — offsets are strictly increasing and each window's limit exactly reaches
      the next offset, so no row is read twice.

    Without a `total` there is no way to size the windows, and a guess would either truncate
    (fatal) or overlap. So an unknown `total` yields a **single unbounded window**: one serial
    reader, correct, slow — which is the right trade, because a slow right answer is a result
    and a fast wrong one is a bug.

    Args:
        total: The result's row count, if the store can be asked cheaply. None if it cannot.
        segments: The requested parallelism.

    Returns:
        The windows, in offset order. Always non-empty; the last is always unbounded.
    """
    if segments <= 1 or total is None or total <= 0:
        return [(0, 0)]  # 0 limit = unbounded: read to the end
    # More segments than rows only makes empty windows; one row per window is the useful floor.
    segments = min(segments, total)
    step = -(-total // segments)  # ceil, so the windows reach `total` before the last one
    windows = [(i * step, step) for i in range(segments - 1)]
    windows.append(((segments - 1) * step, 0))  # the tail, unbounded — this is the cover
    return windows


@dataclass(frozen=True, slots=True)
class PartitionSpec:
    """How a `ScanSource` is divided into parallel read units.

    Concrete connectors interpret this against their store: `segments` is the
    requested parallelism (DynamoDB ``TotalSegments``, slice count, offset
    windows), and `extra` carries connector-specific knobs (page size, ring
    token count, …). It is a picklable value object so it travels inside a split.
    """

    segments: int = 1
    extra: dict[str, Any] = field(default_factory=dict)


class ScanSource(ABC):
    """Base for a row-based NoSQL/operational store read as Arrow.

    Construction stores the connection kwargs (verbatim, never logged) plus an
    optional `PartitionSpec`. The base implements the `Source` surface
    (`schema`/`read`/`iter_batches`/`row_count`/`identity`/`splits`) in terms of
    two overrides:

    * `_enumerate_partitions()` — the store's natural parallel units as opaque,
      picklable locators (token ranges, scan segments, offset windows).
    * `_read_partition(partition, projection)` — fetch one partition's rows and
      yield Arrow batches (typically via `rows_to_batches`).

    Subclasses also set `format_name` (the registry key) and implement
    `_infer_schema()`.
    """

    # The registry name and the picklable connection kwargs. Subclasses set
    # `format_name`; the base keeps `_conn_kwargs` opaque and never logs it.
    format_name: str = ""

    __slots__ = ("_conn_kwargs", "_partition_spec", "_schema_cache")

    def __init__(
        self,
        *,
        partition_spec: PartitionSpec | None = None,
        **conn_kwargs: Any,
    ) -> None:
        self._conn_kwargs = conn_kwargs
        self._partition_spec = partition_spec or PartitionSpec()
        self._schema_cache: pa.Schema | None = None

    # ---- shared, do-not-override ------------------------------------------
    def schema(self) -> pa.Schema:
        if self._schema_cache is None:
            self._schema_cache = self._infer_schema()
        return self._schema_cache

    def read(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> list[pa.RecordBatch]:
        return list(self.iter_batches(projection, predicate))

    def iter_batches(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> Iterator[pa.RecordBatch]:
        for partition in self._enumerate_partitions():
            yield from self._read_partition(partition, projection, predicate)

    def row_count(self) -> int | None:
        return None

    def identity(self) -> str:
        return f"{self.format_name}:{self._identity_suffix()}"

    def splits(
        self,
        target_size: int | None = None,  # noqa: ARG002 (protocol signature; a store splits by partition)
        predicate: dict | None = None,
    ) -> list[_ScanSplit]:
        """The store's partitions, each carrying the predicate Kyber pushed to this scan.

        Declaring `predicate` here is what connects a NoSQL store to the optimizer on the
        **distributed** path. `io.source.plan_splits` only passes a predicate to a `splits()`
        that asks for one, and `dist...scan_read._split_read` only passes it to a `Split.read`
        that asks for one — so a source that omits it silently reads its *entire* table on
        every worker and lets the engine's `Filter` discard the rows. Correct, and ruinous: on
        DynamoDB you pay read-capacity for every item; on Mongo you drag the collection across
        the network.

        Baking it into the split (rather than into per-read instance state, which is what the
        Cassandra/DynamoDB sources used to do) is also what makes pushdown *survive the trip to
        the worker*: a split is picklable and self-contained, and `_source()` rebuilds a clean
        connector from `conn_kwargs` that knows nothing about any earlier call.
        """
        prefix = self.identity()
        return [
            _ScanSplit(
                source_cls=type(self),
                conn_kwargs=dict(self._conn_kwargs),
                partition=partition,
                identity_prefix=prefix,
                predicate=predicate,
            )
            for partition in self._enumerate_partitions()
        ]

    # ---- override points --------------------------------------------------
    @abstractmethod
    def _infer_schema(self) -> pa.Schema:
        """Determine the Arrow schema (sampling a row or reading store metadata)."""

    @abstractmethod
    def _enumerate_partitions(self) -> list[PartitionLocator]:
        """Return the store's natural parallel units as opaque, picklable locators."""

    @abstractmethod
    def _read_partition(
        self,
        partition: PartitionLocator,
        projection: list[str] | None,
        predicate: dict | None = None,
    ) -> Iterator[pa.RecordBatch]:
        """Fetch one partition's rows and yield Arrow batches.

        `predicate` is the filter to push **into the store's own query** — a DynamoDB
        `FilterExpression`, a CQL `WHERE`, a Mongo query document, an ES DSL clause. It is
        passed as an argument rather than held on the instance so that pushdown is a pure
        function of `(partition, projection, predicate)`: reentrant under concurrent reads,
        and identical whether it runs on the driver or on a worker that rebuilt this source
        from a pickled split. A connector that cannot push it simply ignores it — the engine's
        `Filter` re-checks every row regardless, so ignoring a predicate is always correct and
        merely slower.
        """

    def _identity_suffix(self) -> str:
        """A non-secret identity suffix; defaults to the connection target.

        Subclasses override to surface a stable, credential-free locator (host +
        keyspace, cluster + collection, …). The base falls back to a generic tag
        so credentials in `_conn_kwargs` never leak into an identity string.
        """
        return "store"
