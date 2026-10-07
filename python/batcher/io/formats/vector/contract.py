"""The vector-store contract: what every vector-store connector agrees on before it dials out.

Qdrant, Pinecone, Milvus and Turbopuffer disagree about almost everything a client sees: what
a point id may be, what the distance metric is called, whether metadata may hold a null. They
agree on what a *frame* bound for any of them must look like, and on what goes wrong with
one, so this module states that once and the four connectors in this package share it.

**The frame.** One id column (``id`` by default), one vector column per vector the store
holds (``embedding`` by default, the column `ml.embed` produces and Lance indexes), and every
other column is payload: Qdrant's payload, Pinecone's metadata, Milvus' scalar fields,
Turbopuffer's attributes. A vector column is ``fixed_size_list<float32, dim>``. A
``fixed_size_list`` of another numeric type, a one-dimensional fixed-shape-tensor column, and a
variable-length ``list`` whose rows all have one length are accepted and normalized to it,
because those are the shapes an embedding pipeline actually produces.

**Validation happens before the first remote call.** Every store refuses a malformed point,
but it refuses it in the middle of a write, after the batches before it have landed, and with
a message about a request rather than a row. So `prepare` checks the whole shard first, with
Arrow compute rather than a Python loop: an id that is null or repeated, a vector that is null,
ragged, of the wrong dimension, or holds a NaN or an infinity (including a float64 that
overflows float32). A shard with any such row writes nothing and raises `VectorWriteError`
naming each point. The sink then asks the store for the target's dimension and metric where its
API says them, and refuses a mismatch -- also before writing.

**Retries are idempotent because ids are stable.** A write is sent in batches of
``batch_size`` points, and a failed batch is retried with backoff. A retried *upsert* lands on
the same ids, so it cannot duplicate a point; that is why every write here needs an id column
rather than letting the store assign one, and why `stable_point_id` derives a store-legal id
from the user's id deterministically rather than randomly. An insert-only write (Milvus'
``append``) is never retried, because a retry of a request that landed but whose response was
lost would duplicate it.

**Failures are reported per point.** A batch that still fails after its retries does not stop
the write: the remaining batches are sent, and the sink raises one `VectorWriteError` at the
end whose `failures` name every point that did not land and why, and whose `written` counts the
ones that did. A partially applied upsert is then repairable by re-running it.
"""

from __future__ import annotations

import time
import uuid
from abc import abstractmethod
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from batcher._internal.errors import BackendError, PlanError
from batcher.io.formats.nosql.base import BulkSink
from batcher.io.manifest import WrittenFile

__all__ = [
    "DEFAULT_ID_COLUMN",
    "DEFAULT_VECTOR_COLUMN",
    "METRICS",
    "PointFailure",
    "PreparedPoints",
    "RemoteTarget",
    "VectorSink",
    "VectorWriteError",
    "check_remote",
    "chunks",
    "prepare",
    "resolve_metric",
    "send_with_retries",
    "stable_point_id",
    "vector_field",
]

#: The id column a frame is read from and written with unless told otherwise.
DEFAULT_ID_COLUMN = "id"

#: The vector column, named after what `ml.embed` writes and `ml.vector_search` reads.
DEFAULT_VECTOR_COLUMN = "embedding"

#: The portable metric names. Each connector maps them onto its own spelling, and also
#: accepts that spelling directly, so a user may write ``"cosine"`` everywhere or the
#: store's own ``"dotproduct"`` / ``"IP"`` / ``"Dot"`` where they already know it.
METRICS = ("cosine", "euclidean", "dot")

#: The first retry waits this long, and each later one twice as long, up to the cap.
_BACKOFF_SECONDS = 0.5
_BACKOFF_CAP_SECONDS = 8.0

#: The namespace a non-UUID string id is hashed under to give a store a stable UUID.
_ID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "https://batcher.dev/vector-point-id")

#: Points named individually in an error message before the rest are only counted.
_SHOWN_FAILURES = 5


@dataclass(frozen=True, slots=True)
class PointFailure:
    """One point that was refused before the write or not applied by the store.

    Attributes:
        id: The point's id, as it appeared in the id column.
        reason: Why it was not written.
    """

    id: Any
    reason: str


class VectorWriteError(BackendError):
    """A vector-store write left some points unwritten; `failures` names each of them.

    Raised once per shard, after every batch has been attempted, so `written` is exact: a
    validation failure raises before anything is sent (``written == 0``), and a batch that
    failed after its retries is listed point by point while the other batches still land.
    The failures and count survive pickling, so a distributed write reports them too.
    """

    def __init__(
        self,
        message: str = "",
        *,
        failures: Sequence[PointFailure] = (),
        written: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.failures: tuple[PointFailure, ...] = tuple(failures)
        self.written = written


@dataclass(frozen=True, slots=True)
class RemoteTarget:
    """What a store says about an existing write target: dimensions per vector, and metric.

    Attributes:
        dimensions: Vector name (or column) to its dimension. Empty when the store did not say.
        metric: The store's own spelling of the target's metric, or None when it did not say.
    """

    dimensions: dict[str, int] = field(default_factory=dict)
    metric: str | None = None


@dataclass(frozen=True, slots=True)
class PreparedPoints:
    """A validated shard: vector columns normalized to ``fixed_size_list<float32>``.

    Attributes:
        table: The shard, with each vector column replaced by its normalized form.
        dimensions: Vector column to its dimension.
    """

    table: pa.Table
    dimensions: dict[str, int]


def resolve_metric(metric: str | None, native: Mapping[str, str], *, store: str) -> str | None:
    """The store's own spelling of `metric`, accepting a portable name or the store's own.

    Args:
        metric: A name from `METRICS`, the store's own spelling (any case), or None.
        native: The store's mapping from each portable name it supports to its spelling.
        store: The store's name, for the error.

    Returns:
        The store's spelling, or None when `metric` is None.

    Raises:
        PlanError: If the store has no such metric.

    Examples:
        .. doctest::

            >>> from batcher.io.formats.vector.contract import resolve_metric
            >>> resolve_metric("dot", {"cosine": "Cosine", "dot": "Dot"}, store="qdrant")
            'Dot'
            >>> resolve_metric("COSINE", {"cosine": "Cosine"}, store="qdrant")
            'Cosine'
    """
    if metric is None:
        return None
    key = metric.strip().lower()
    if key in native:
        return native[key]
    for spelling in native.values():
        if key == spelling.lower():
            return spelling
    raise PlanError(
        f"{store} has no distance metric {metric!r}",
        available=[*native, *(v for v in native.values() if v.lower() not in native)],
        available_label="Accepted metrics",
        hint="Name one of them, or omit metric= to skip the check.",
    )


def check_remote(
    remote: RemoteTarget | None,
    *,
    dimensions: Mapping[str, int],
    metric: str | None,
    store: str,
    target: str,
) -> None:
    """Refuse a write whose vectors or metric disagree with the existing target.

    A dimension the store did not report, or a target it has not created yet, is not
    checked: there is nothing to disagree with. The comparison is by the store's own
    spelling of the metric, which `resolve_metric` produced.

    Args:
        remote: What the store said about the target, or None when it does not exist yet.
        dimensions: The frame's dimension per vector (as the store names the vector).
        metric: The metric the caller asked for, in the store's spelling, or None.
        store: The store's name, for the error.
        target: The collection, index or namespace, for the error.

    Raises:
        PlanError: If a dimension or the metric disagrees.
    """
    if remote is None:
        return
    for name, dim in dimensions.items():
        expected = remote.dimensions.get(name)
        if expected is not None and expected != dim:
            raise PlanError(
                f"{store} {target!r} holds {expected}-dimensional vectors"
                + (f" under {name!r}" if name else "")
                + f", but this frame's are {dim}-dimensional; nothing was written",
                hint="Write to a target created for this embedding model.",
            )
    if metric is not None and remote.metric is not None and metric.lower() != remote.metric.lower():
        raise PlanError(
            f"{store} {target!r} uses the {remote.metric!r} metric, not {metric!r}; "
            "nothing was written",
            hint="Omit metric= to accept the target's, or write to another target.",
        )


def vector_field(column: str, dimension: int) -> pa.Field:
    """The Arrow field a vector column is read back as: ``fixed_size_list<float32, dim>``.

    Args:
        column: The column name.
        dimension: The vector dimension.

    Returns:
        The field.
    """
    return pa.field(column, pa.list_(pa.float32(), dimension))


def prepare(
    table: pa.Table,
    *,
    id_column: str,
    vector_columns: Sequence[str],
    dimension: int | None = None,
) -> PreparedPoints:
    """Validate a shard and normalize its vector columns, before anything is sent.

    Args:
        table: The shard to write.
        id_column: The id column.
        vector_columns: The vector columns; empty for a delete, which needs only ids.
        dimension: The dimension every vector must have; None to require only that each
            column's vectors agree with one another.

    Returns:
        The shard with its vector columns as ``fixed_size_list<float32, dim>``.

    Raises:
        PlanError: If a named column is missing or of a type no store can take.
        VectorWriteError: If any row is invalid, naming each one; nothing was written.
    """
    _require_columns(table, [id_column, *vector_columns])
    ids = table.column(id_column)
    if not (
        pa.types.is_integer(ids.type)
        or pa.types.is_string(ids.type)
        or pa.types.is_large_string(ids.type)
    ):
        raise PlanError(
            f"id column {id_column!r} is {ids.type}; a vector store needs integer or string ids",
            hint="Cast it with .cast('int64') or .cast('string').",
        )
    bad: dict[int, str] = {}
    _flag(bad, np.flatnonzero(np.asarray(pc.is_null(ids))), "null id")
    _flag_duplicates(bad, ids)
    out = table
    dims: dict[str, int] = {}
    for column in vector_columns:
        normalized, dim = _normalize_vectors(table.column(column), column, dimension, bad)
        if normalized is not None:
            out = out.set_column(out.schema.get_field_index(column), column, normalized)
        dims[column] = dim
    if bad:
        id_values = ids.take(pa.array(sorted(bad), pa.int64())).to_pylist()
        failures = [
            PointFailure(i, bad[row]) for i, row in zip(id_values, sorted(bad), strict=True)
        ]
        raise VectorWriteError(
            _failure_message(failures, written=0, total=table.num_rows)
            + " Nothing was written: the shard is checked before its first request.",
            failures=failures,
        )
    return PreparedPoints(out, dims)


def chunks(table: pa.Table, size: int) -> Iterator[pa.Table]:
    """Zero-copy slices of `table`, `size` rows each (the last may be shorter).

    Args:
        table: The table to slice.
        size: Rows per slice.

    Yields:
        Each slice in order.
    """
    for start in range(0, table.num_rows, size):
        yield table.slice(start, size)


def send_with_retries(send: Callable[[], Any], *, retries: int) -> Exception | None:
    """Run `send`, retrying with exponential backoff; return the last error, or None.

    Every exception is retried: the clients here do not agree on which of theirs are
    transient, and an idempotent upsert that fails twice for a permanent reason costs only
    the backoff. The caller decides whether retrying is safe at all.

    Args:
        send: The request, as a no-argument callable.
        retries: How many times to retry after the first attempt fails.

    Returns:
        None when an attempt succeeded, else the exception the last attempt raised.
    """
    error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            send()
        except Exception as exc:  # recorded per point by the caller, never swallowed
            error = exc
            if attempt < retries:
                time.sleep(min(_BACKOFF_SECONDS * 2**attempt, _BACKOFF_CAP_SECONDS))
            continue
        return None
    return error


def stable_point_id(value: Any) -> int | str:
    """A Qdrant-legal point id for `value`, the same on every run and every retry.

    Qdrant accepts an unsigned integer or a UUID and nothing else. An integer and a UUID
    string pass through; any other string becomes the UUID5 of itself, so ``"doc-17"`` maps
    to one UUID forever and a retried upsert lands on the same point instead of a new one.

    Args:
        value: The id from the frame.

    Returns:
        The id to send.

    Raises:
        PlanError: If `value` is a negative integer, which no UUID or unsigned id can hold.

    Examples:
        .. doctest::

            >>> from batcher.io.formats.vector.contract import stable_point_id
            >>> stable_point_id(7)
            7
            >>> stable_point_id("doc-1") == stable_point_id("doc-1")
            True
    """
    if isinstance(value, int):
        if value < 0:
            raise PlanError(f"point id {value} is negative; Qdrant ids are unsigned")
        return value
    text = str(value)
    try:
        return str(uuid.UUID(text))
    except ValueError:
        return str(uuid.uuid5(_ID_NAMESPACE, text))


class VectorSink(BulkSink):
    """Base for the vector-store sinks: validate, check the target, send in batches.

    A subclass names its metrics and default batch size and implements three calls against
    its client: `_describe` (what the target holds, or None), `_send` (one batch under
    `mode`), and `_client`. Everything else -- validation, the remote check, batching,
    retries, the per-point report -- is here, so every store does it the same way.

    ``upsert`` replaces points by id, which is what each store's write primitive does.
    ``delete`` removes the points whose ids the frame holds. Neither discards a point the
    write did not name, so both are safe on every shard of a distributed write.
    """

    #: The store's spelling of each portable metric it supports.
    native_metrics: ClassVar[Mapping[str, str]] = {}

    #: Points per request unless `batch_size` says otherwise.
    default_batch_size: int = 100

    supported_modes = ("upsert", "delete")
    destructive_modes: frozenset[str] = frozenset()

    #: Modes whose retry could duplicate a point; never retried.
    unretryable_modes: frozenset[str] = frozenset()

    __slots__ = ("batch_size", "dimension", "id_column", "max_retries", "metric", "vector_column")

    def __init__(
        self,
        *,
        id_column: str = DEFAULT_ID_COLUMN,
        vector_column: str = DEFAULT_VECTOR_COLUMN,
        dimension: int | None = None,
        metric: str | None = None,
        batch_size: int | None = None,
        max_retries: int = 3,
        mode: str = "upsert",
        **conn_kwargs: Any,
    ) -> None:
        super().__init__(key_field=id_column, mode=mode, **conn_kwargs)
        if batch_size is not None and batch_size < 1:
            raise PlanError(f"batch_size must be at least 1, got {batch_size}")
        if max_retries < 0:
            raise PlanError(f"max_retries must be 0 or more, got {max_retries}")
        if dimension is not None and dimension < 1:
            raise PlanError(f"dimension must be at least 1, got {dimension}")
        self.id_column = id_column
        self.vector_column = vector_column
        self.dimension = dimension
        self.metric = resolve_metric(metric, self.native_metrics, store=self.format_name)
        self.batch_size = batch_size or self.default_batch_size
        self.max_retries = max_retries

    def vector_columns(self) -> dict[str, str]:
        """Each vector column, keyed by the name the store holds it under ("" for unnamed)."""
        return {"": self.vector_column}

    def write(self, table: pa.Table, path: str) -> WrittenFile:
        """Validate the shard, check the target, then send it batch by batch.

        Args:
            table: The shard to write.
            path: The collection, index or namespace.

        Returns:
            A `WrittenFile` counting the points applied.

        Raises:
            PlanError: If a column is missing or mistyped, or the target disagrees.
            VectorWriteError: If any point was refused or not applied, naming each one.
        """
        from batcher.plan.types import logical_bytes

        if table.num_rows == 0:
            return WrittenFile(path=path, rows=0, bytes=0)
        vectors = self.vector_columns() if self.mode != "delete" else {}
        prepared = prepare(
            table,
            id_column=self.id_column,
            vector_columns=list(vectors.values()),
            dimension=self.dimension,
        )
        if vectors:
            self._check_payload(table, path)
        client = self._client()
        try:
            if vectors:
                check_remote(
                    self._describe(client, path, self._payload_columns(table)),
                    dimensions={name: prepared.dimensions[col] for name, col in vectors.items()},
                    metric=self.metric,
                    store=self.format_name,
                    target=path,
                )
            written, failures = self._send_all(client, path, prepared.table)
        finally:
            self._close(client)
        if failures:
            raise VectorWriteError(
                _failure_message(failures, written=written, total=table.num_rows),
                failures=failures,
                written=written,
            )
        return WrittenFile(path=path, rows=written, bytes=logical_bytes(table))

    def _send_all(self, client: Any, path: str, table: pa.Table) -> tuple[int, list[PointFailure]]:
        """Send every batch, recording the points of each batch that still failed."""
        retries = 0 if self.mode in self.unretryable_modes else self.max_retries
        written = 0
        failures: list[PointFailure] = []
        for chunk in chunks(table, self.batch_size):
            error = send_with_retries(lambda c=chunk: self._send(client, path, c), retries=retries)
            if error is None:
                written += chunk.num_rows
                continue
            reason = f"{type(error).__name__}: {error}"
            failures.extend(
                PointFailure(i, reason) for i in chunk.column(self.id_column).to_pylist()
            )
        return written, failures

    def _check_payload(self, table: pa.Table, path: str) -> None:
        """Refuse a payload column the store cannot hold, before anything is sent."""
        for name in self._payload_columns(table):
            dtype = table.schema.field(name).type
            if not self._payload_type_ok(dtype):
                raise PlanError(
                    f"{self.format_name} cannot store column {name!r} ({dtype}) as payload "
                    f"in {path!r}; nothing was written",
                    hint="Cast it to a supported type, encode it as a string, or drop it.",
                )

    def _payload_type_ok(self, dtype: pa.DataType) -> bool:  # noqa: ARG002 - overridden per store
        """Whether the store can hold a payload column of `dtype`; any type by default."""
        return True

    def _apply(self, rows: list[dict[str, Any]], path: str) -> None:  # noqa: ARG002
        """Never reached: a vector sink validates and sends Arrow batches in `write`."""
        raise TypeError("VectorSink writes Arrow batches through write(); _apply is unused")

    def _close(self, client: Any) -> None:
        """Release `client`; a client with nothing to close is left alone."""
        close = getattr(client, "close", None)
        if callable(close):
            close()

    def _payload_columns(self, table: pa.Table) -> list[str]:
        """Every column that is neither the id nor a vector."""
        skip = {self.id_column, *self.vector_columns().values()}
        return [name for name in table.column_names if name not in skip]

    @abstractmethod
    def _client(self) -> Any:
        """Open a client, resolving credentials here (on the worker), never earlier."""

    @abstractmethod
    def _describe(self, client: Any, path: str, payload: list[str]) -> RemoteTarget | None:
        """What the store says the target holds, or None when it does not exist yet.

        `payload` names the frame's payload columns, for a store whose target has a fixed
        schema to check them against. Raise `PlanError` here to refuse the write.
        """

    @abstractmethod
    def _send(self, client: Any, path: str, chunk: pa.Table) -> None:
        """Apply one batch under `mode`, raising if the store did not apply all of it."""


def _require_columns(table: pa.Table, columns: Sequence[str]) -> None:
    """Raise naming the first of `columns` the table does not have."""
    for column in columns:
        if column not in table.column_names:
            raise PlanError(
                f"the frame is missing column {column!r}",
                available=table.column_names,
                available_label="Columns",
                hint="Rename it, or name the right column with id_column= / vector_column=.",
            )


def _flag(bad: dict[int, str], rows: Any, reason: str) -> None:
    """Record `reason` for each row index in `rows`, keeping the first reason a row got."""
    for row in rows.tolist():
        bad.setdefault(int(row), reason)


def _flag_duplicates(bad: dict[int, str], ids: pa.ChunkedArray) -> None:
    """Flag every occurrence of an id that appears more than once in the shard.

    Two points with one id in one write are a race the store settles by request order,
    which batching and retries make arbitrary, so neither is the "right" one to keep.
    """
    counts = pc.value_counts(ids.drop_null())
    repeated = counts.filter(pc.greater(counts.field("counts"), 1)).field("values")
    if len(repeated):
        mask = np.asarray(pc.is_in(ids, value_set=repeated).fill_null(False))
        _flag(bad, np.flatnonzero(mask), "duplicate id in this write")


def _normalize_vectors(
    column: pa.ChunkedArray, name: str, dimension: int | None, bad: dict[int, str]
) -> tuple[pa.Array | None, int]:
    """Normalize one vector column, flagging bad rows; return (array or None, dimension).

    The array is None when a row was flagged, since a frame with bad rows is not sent.
    """
    arr = column.combine_chunks() if isinstance(column, pa.ChunkedArray) else column
    if isinstance(arr.type, pa.FixedShapeTensorType):
        if len(arr.type.shape) != 1:
            raise PlanError(
                f"vector column {name!r} holds {arr.type.shape}-shaped tensors; a vector "
                "store needs one-dimensional vectors"
            )
        arr = arr.storage
    value_type = getattr(arr.type, "value_type", None)
    if not (
        (
            pa.types.is_fixed_size_list(arr.type)
            or pa.types.is_list(arr.type)
            or pa.types.is_large_list(arr.type)
        )
        and value_type is not None
        and (pa.types.is_floating(value_type) or pa.types.is_integer(value_type))
    ):
        raise PlanError(
            f"vector column {name!r} is {arr.type}; a vector column is "
            "fixed_size_list<float32, dim>",
            hint="Produce it with ml.embed(..., output_type='fixed_size_list') or cast it.",
        )
    if pa.types.is_fixed_size_list(arr.type) and dimension not in (None, arr.type.list_size):
        raise PlanError(
            f"vector column {name!r} holds {arr.type.list_size}-dimensional vectors, but "
            f"dimension={dimension} was asked for; nothing was written"
        )
    valid = np.asarray(pc.is_valid(arr))
    rows = np.flatnonzero(valid)
    _flag(bad, np.flatnonzero(~valid), "null vector")
    present = arr.filter(pa.array(valid))
    lengths = np.asarray(pc.list_value_length(present).fill_null(0))
    dim = dimension if dimension is not None else (int(lengths[0]) if len(lengths) else 0)
    _flag(bad, rows[lengths != dim], f"vector does not have {dim} values")
    values = present.flatten().cast(pa.float32(), safe=False)
    finite = np.isfinite(values.to_numpy(zero_copy_only=False))
    parents = np.asarray(pc.list_parent_indices(present))
    _flag(bad, rows[np.unique(parents[~finite])], "vector holds a NaN, infinity or null")
    if bad:
        return None, dim
    return pa.FixedSizeListArray.from_arrays(values, dim), dim


def _failure_message(failures: Sequence[PointFailure], *, written: int, total: int) -> str:
    """One line naming how many points failed, and the first few with their reasons."""
    shown = "; ".join(f"{f.id!r}: {f.reason}" for f in failures[:_SHOWN_FAILURES])
    more = len(failures) - _SHOWN_FAILURES
    tail = f"; and {more} more" if more > 0 else ""
    return (
        f"{len(failures)} of {total} points were not written ({written} were): {shown}{tail}."
        " The full list is on the error's .failures."
    )
