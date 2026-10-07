"""Resumable incremental API ingestion: a durable watermark, a lookback, and dedup by key.

`Incremental` is the user's description; `IncrementalRun` is one read's working copy of it.

**What is persisted.** A small JSON document at `Incremental.state` (local or any
`resolve_filesystem` scheme): the *watermark* (the greatest value of `cursor_field` read so
far), the *cursor* (the pagination token of the last page whose records were all handed to
the consumer), and the dedup tokens still inside the window a later run can re-fetch.

**When it is persisted** mirrors the incremental file source (`streaming/autoloader.py`).
A read *stages* the new state only once every page has been consumed -- a read that dies
part-way stages nothing, so the next run starts from the old state and loses no record. With
``auto_commit=True`` (the default) the staged state is committed at that point, which is
what the incremental file source does too; under a streaming query that point follows the
publish of the last epoch, and the source also exposes `confirm`/`snapshot_position`/`seek`
for the checkpoint. A one-shot ``read -> write`` drains the read *before* the sink commits,
so a sink failure after that would lose the batch on resume: pass ``auto_commit=False`` and
call `Incremental.commit` after the write succeeds to close that window.

**Why a page boundary loses nothing.** A resumed read asks the API from the watermark minus
`lookback` *inclusive*, so a record sharing the boundary value, or updated late inside the
lookback, is fetched again. Each record version is identified by its key plus its cursor
value; a version already delivered is dropped, a new version of the same key passes. Tokens
are kept only for the window a later run can re-fetch (at or after the next lower bound), so
the state stays the size of the lookback window, not the size of the table.

All filtering is Arrow compute over a page, never a Python loop over rows.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc

from batcher._internal.errors import FormatError, PlanError

__all__ = ["Incremental", "IncrementalRun"]

_SEP = "\x1f"


@dataclass(frozen=True, slots=True)
class Incremental:
    """Resume an API read where the last one stopped, with a lookback and dedup by key.

    Give `cursor_field` (a monotonically updated field such as ``updated_at``) to resume by
    watermark: the next read sends the watermark minus `lookback` as the query parameter
    `param` (such as GitHub's ``since``) and drops records older than it. Leave
    `cursor_field` unset to resume by the pagination cursor instead: the next read starts at
    the last page the previous one accepted. Either way `key` deduplicates the overlap, so a
    record re-fetched at the boundary is not delivered twice.

    Examples:
        .. doctest::

            >>> import batcher as bt, tempfile, os
            >>> from datetime import timedelta
            >>> inc = bt.io.Incremental(
            ...     state=os.path.join(tempfile.mkdtemp(), "issues.json"),
            ...     cursor_field="updated_at",
            ...     key="id",
            ...     lookback=timedelta(minutes=5),
            ...     param="since",
            ... )
            >>> inc.load() is None
            True

    Args:
        state: Where the state document lives (a local path or an object-store URI).
        cursor_field: The record field whose greatest value is the watermark.
        key: The field, or fields, identifying a record for deduplication.
        lookback: How far before the watermark the next read starts: a `timedelta` for a
            timestamp cursor, a number for a numeric one.
        start: The lower bound for the very first read, before any state exists.
        param: The query parameter the lower bound is sent as, or None to filter only
            on the client side.
        auto_commit: Commit the new state as soon as the read is fully consumed. Set it
            to False to commit explicitly with `commit` after a write succeeds.
    """

    state: str
    cursor_field: str | None = None
    key: str | tuple[str, ...] | None = None
    lookback: timedelta | int | float | None = None
    start: Any = None
    param: str | None = None
    auto_commit: bool = True

    def __post_init__(self) -> None:
        if not self.state:
            raise PlanError("Incremental(state=...) must name where the state is kept")
        if self.lookback is not None and self.cursor_field is None:
            raise PlanError("Incremental(lookback=...) needs cursor_field= to look back along")

    @property
    def keys(self) -> tuple[str, ...]:
        """The dedup key fields, as a tuple (empty when no key is set).

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.io.Incremental(state="s.json", key="id").keys
                ('id',)

        Returns:
            The key field names.
        """
        if self.key is None:
            return ()
        return (self.key,) if isinstance(self.key, str) else tuple(self.key)

    def load(self) -> dict | None:
        """The committed state: ``watermark``, ``cursor`` (the last accepted page) and more.

        Examples:
            .. doctest::

                >>> import batcher as bt, tempfile, os
                >>> path = os.path.join(tempfile.mkdtemp(), "s.json")
                >>> bt.io.Incremental(state=path).load() is None
                True

        Returns:
            The committed state document, or None before the first commit.
        """
        return _read_json(self.state)

    def pending(self) -> dict | None:
        """The state a finished read staged but has not committed, or None.

        Examples:
            .. doctest::

                >>> import batcher as bt, tempfile, os
                >>> path = os.path.join(tempfile.mkdtemp(), "s.json")
                >>> bt.io.Incremental(state=path, auto_commit=False).pending() is None
                True

        Returns:
            The staged state document, or None.
        """
        return _read_json(self._pending_path)

    def commit(self) -> bool:
        """Promote the staged state to the committed one.

        Call it after the write that consumed the read has succeeded, when the read was
        built with ``auto_commit=False``.

        Examples:
            .. doctest::

                >>> import batcher as bt, tempfile, os
                >>> path = os.path.join(tempfile.mkdtemp(), "s.json")
                >>> bt.io.Incremental(state=path, auto_commit=False).commit()
                False

        Returns:
            True when a staged state was committed, False when there was none.
        """
        staged = self.pending()
        if staged is None:
            return False
        _write_json(self.state, staged)
        from batcher.io.filesystem import resolve_filesystem

        resolve_filesystem(self._pending_path).remove(self._pending_path)
        return True

    @property
    def _pending_path(self) -> str:
        return f"{self.state}.pending"

    def _stage(self, document: dict) -> None:
        """Write `document` as the staged state, committing it under ``auto_commit``."""
        _write_json(self._pending_path, document)
        if self.auto_commit:
            self.commit()


def _read_json(path: str) -> dict | None:
    from batcher.io.filesystem import resolve_filesystem

    fs = resolve_filesystem(path)
    if not fs.exists(path):
        return None
    with fs.open(path, "rb") as fh:
        raw = fh.read()
    try:
        document = json.loads(raw)
    except ValueError as exc:
        raise FormatError(f"incremental state at {path!r} is not valid JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise FormatError(f"incremental state at {path!r} is not a JSON object")
    return document


def _write_json(path: str, document: dict) -> None:
    from batcher.io.filesystem import resolve_filesystem

    fs = resolve_filesystem(path)
    parent = path.rstrip("/").rpartition("/")[0]
    if parent:
        fs.mkdirs(parent, exist_ok=True)
    with fs.atomic_writer(path) as fh:
        fh.write(json.dumps(document, sort_keys=True, default=str).encode())


def shift_back(value: Any, lookback: Any) -> Any:
    """`value` moved back by `lookback`, keeping an ISO string's ``Z`` style.

    Examples:
        .. doctest::

            >>> from datetime import timedelta
            >>> from batcher.io.formats.http.state import shift_back
            >>> shift_back("2024-01-02T00:10:00Z", timedelta(minutes=10))
            '2024-01-02T00:00:00Z'
            >>> shift_back(100, 5)
            95
    """
    if lookback is None or value is None:
        return value
    if isinstance(value, datetime):
        return value - _as_timedelta(lookback)
    if isinstance(value, str):
        if not isinstance(lookback, timedelta):
            raise PlanError("a string watermark needs a timedelta lookback")
        moved = (datetime.fromisoformat(value) - lookback).isoformat()
        return moved.replace("+00:00", "Z") if value.endswith("Z") else moved
    if isinstance(lookback, timedelta):
        raise PlanError("a numeric watermark needs a numeric lookback")
    return value - lookback


def _as_timedelta(lookback: Any) -> timedelta:
    if not isinstance(lookback, timedelta):
        raise PlanError("a timestamp watermark needs a timedelta lookback")
    return lookback


def _comparable(column: pa.Array) -> pa.Array:
    """A version of `column` that orders correctly: ISO strings as UTC timestamps."""
    if pa.types.is_string(column.type) or pa.types.is_large_string(column.type):
        try:
            return column.cast(pa.timestamp("us", tz="UTC"))
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
            return column
    return column


def _scalar_like(value: Any, dtype: pa.DataType) -> pa.Scalar:
    if pa.types.is_timestamp(dtype) and isinstance(value, str):
        return pa.scalar(datetime.fromisoformat(value)).cast(dtype)
    return pa.scalar(value).cast(dtype)


def _stored(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat().replace("+00:00", "Z")
    return value


class IncrementalRun:
    """One read's working copy of an `Incremental`: the bound, the filter, the next state.

    Args:
        spec: The user's `Incremental`.
    """

    __slots__ = (
        "_cursor",
        "_cursor_type",
        "_seen_cursors",
        "_seen_tokens",
        "_spec",
        "_start_state",
        "_watermark",
    )

    def __init__(self, spec: Incremental) -> None:
        self._spec = spec
        state = spec.load() or {}
        self._start_state = state
        self._watermark = state.get("watermark", spec.start)
        self._cursor = state.get("cursor")
        self._cursor_type: pa.DataType | None = None
        seen = state.get("seen") or []
        self._seen_tokens = pa.array([str(t) for t, _ in seen], pa.string())
        self._seen_cursors = pa.array([c for _, c in seen], pa.string())

    @property
    def lower_bound(self) -> Any:
        """The value the read starts from: the watermark minus the lookback."""
        return _stored(shift_back(self._watermark, self._spec.lookback))

    @property
    def resume_cursor(self) -> Any:
        """The pagination cursor to resume from, when resuming by cursor rather than field."""
        return self._cursor if self._spec.cursor_field is None else None

    @property
    def watermark(self) -> Any:
        """The greatest cursor-field value read so far."""
        return self._watermark

    def filter(self, batch: pa.RecordBatch) -> pa.RecordBatch:
        """Drop the rows below the lower bound and the versions already delivered.

        Args:
            batch: One page's batch.

        Returns:
            The rows to deliver; the watermark and dedup tokens advance past them.
        """
        spec = self._spec
        if spec.cursor_field is not None:
            batch = self._above_bound(batch)
        if spec.keys and batch.num_rows:
            batch = self._dedup(batch)
        if spec.cursor_field is not None and batch.num_rows:
            self._advance(batch.column(spec.cursor_field))
        return batch

    def accept(self, cursor: Any) -> None:
        """Record `cursor` as the last page whose records were all consumed."""
        if cursor is not None:
            self._cursor = cursor

    def finish(self) -> None:
        """Stage the new state, and commit it when the spec says ``auto_commit``."""
        document = {
            "version": 1,
            "watermark": _stored(self._watermark),
            "cursor": self._cursor,
            "seen": self._kept_tokens(),
        }
        self._spec._stage(document)

    # ---- the vectorized pieces -------------------------------------------------
    def _field(self, batch: pa.RecordBatch, name: str) -> pa.Array:
        if name not in batch.schema.names:
            raise PlanError(
                f"Incremental names field {name!r}, which the read's schema does not have "
                f"(it has {batch.schema.names}); declare it in schema="
            )
        return batch.column(name)

    def _above_bound(self, batch: pa.RecordBatch) -> pa.RecordBatch:
        bound = self.lower_bound
        if bound is None or not batch.num_rows:
            return batch
        column = _comparable(self._field(batch, str(self._spec.cursor_field)))
        keep = pc.greater_equal(column, _scalar_like(bound, column.type))
        # A record with no cursor value cannot be placed before or after the bound, so it
        # is delivered rather than silently dropped.
        return batch.filter(pc.fill_null(keep, True))

    def _tokens(self, batch: pa.RecordBatch) -> pa.Array:
        names = [*self._spec.keys]
        if self._spec.cursor_field is not None:
            names.append(self._spec.cursor_field)
        parts = []
        for name in names:
            try:
                parts.append(pc.fill_null(self._field(batch, name).cast(pa.string()), ""))
            except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as exc:
                raise PlanError(
                    f"Incremental key field {name!r} cannot be compared: {exc}"
                ) from exc
        if len(parts) == 1:
            return parts[0]
        return pc.binary_join_element_wise(*parts, _SEP)

    def _dedup(self, batch: pa.RecordBatch) -> pa.RecordBatch:
        tokens = self._tokens(batch)
        fresh = pc.invert(pc.is_in(tokens, value_set=self._seen_tokens))
        positions = pa.array(range(batch.num_rows), pa.int64())
        firsts = (
            pa.table({"t": tokens, "i": positions})
            .group_by("t", use_threads=False)
            .aggregate([("i", "min")])
            .column("i_min")
        )
        keep = pc.and_(fresh, pc.is_in(positions, value_set=firsts))
        kept = batch.filter(keep)
        new_tokens = tokens.filter(keep)
        if self._spec.cursor_field is None:
            # Resuming by page cursor re-reads only the last accepted page, so only that
            # page's tokens can recur: keep the latest page's, not the whole history.
            self._seen_tokens = new_tokens
            self._seen_cursors = pa.nulls(len(new_tokens), pa.string())
        else:
            cursors = kept.column(self._spec.cursor_field).cast(pa.string())
            self._seen_tokens = pa.concat_arrays([self._seen_tokens, new_tokens])
            self._seen_cursors = pa.concat_arrays([self._seen_cursors, cursors])
        return kept

    def _advance(self, column: pa.Array) -> None:
        self._cursor_type = column.type
        comparable = _comparable(column)
        top = pc.max(comparable)
        if not top.is_valid:
            return
        if self._watermark is not None:
            current = _scalar_like(self._watermark, comparable.type)
            if pc.less_equal(top, current).as_py():
                return
        index = pc.index(comparable, top).as_py()
        self._watermark = _stored(column[index].as_py())

    def _kept_tokens(self) -> list[list[Any]]:
        """The tokens still inside the window the next read re-fetches."""
        tokens, cursors = self._seen_tokens, self._seen_cursors
        if len(tokens) and self._cursor_type is not None and self._watermark is not None:
            # The cursor values were kept as text; read them back as the column's own type
            # so a numeric cursor orders numerically rather than lexically.
            typed = cursors if pa.types.is_string(self._cursor_type) else None
            if typed is None:
                typed = cursors.cast(self._cursor_type)
            comparable = _comparable(typed)
            bound = self.lower_bound
            keep = pc.fill_null(
                pc.greater_equal(comparable, _scalar_like(bound, comparable.type)), False
            )
            tokens, cursors = tokens.filter(keep), cursors.filter(keep)
        return [list(pair) for pair in zip(tokens.to_pylist(), cursors.to_pylist(), strict=True)]
