"""Apache Hudi format — read-only via `hudi` (hudi-rs).

`HudiSource` reads a Hudi table as Arrow: a snapshot read of the current table
state, time travel to an instant, and an incremental read between two instants.
Hudi writes require the Spark/Flink write stack and are out of scope for the
Rust/Arrow data plane, so `HudiSink` exists only to raise a clear `BackendError`.

All `hudi` imports are deferred — importing this module never requires the
optional dependency. A missing dependency raises `BackendError` with a
``pip install 'batcher[hudi]'`` hint.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pyarrow as pa

from batcher._internal.errors import BackendError
from batcher.io.formats.base import SINKS, SOURCES
from batcher.io.splits import Split, WholeSourceSplit

__all__ = ["HudiSink", "HudiSource"]


def _require_hudi() -> Any:
    """Import and return the hudi-rs `HudiTable` class or raise `BackendError`."""
    try:
        from hudi import HudiTable
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise BackendError(
            "Hudi read support requires hudi-rs: pip install 'batcher[hudi]'"
        ) from exc
    return HudiTable


_HUDI_OP = {"eq": "=", "ne": "!=", "lt": "<", "le": "<=", "gt": ">", "ge": ">="}
_HUDI_FLIP = {"lt": "gt", "le": "ge", "gt": "lt", "ge": "le", "eq": "eq", "ne": "ne"}


def _hudi_filters(predicate: dict | None) -> list[tuple[str, str, Any]]:
    """Translate an AND-of-comparisons predicate to hudi-rs ``(col, op, value)``
    filter tuples, or ``[]`` if the predicate isn't fully pushable (OR / computed
    terms) — the caller then reads unfiltered and the engine's `Filter` re-checks.
    """
    if predicate is None:
        return []
    out: list[tuple[str, str, Any]] = []

    def walk(node: dict) -> bool:
        if node.get("e") != "binary":
            return False
        op = node["op"]
        if op == "and":
            return walk(node["left"]) and walk(node["right"])
        if op in _HUDI_OP:
            left, right = node["left"], node["right"]
            if left.get("e") == "col" and right.get("e") == "lit":
                out.append((left["name"], _HUDI_OP[op], next(iter(right["value"].values()))))
                return True
            if left.get("e") == "lit" and right.get("e") == "col":
                flipped = _HUDI_OP[_HUDI_FLIP[op]]
                out.append((right["name"], flipped, next(iter(left["value"].values()))))
                return True
        return False

    return out if walk(predicate) else []


@SOURCES.register("hudi")
class HudiSource:
    """A read-only Apache Hudi table read as Arrow.

    Args:
        table_uri: The table root (local path or cloud URI).
        as_of_instant: Optional Hudi instant timestamp for snapshot time travel.
        options: Optional hudi-rs reader options (incl. cloud storage options).
    """

    # Predicate pushdown: a pushed predicate becomes hudi-rs ``(col, op, value)``
    # filter tuples, best-effort — if the backend rejects them the read retries
    # unfiltered and the engine's `Filter` re-checks, so it is always safe.
    supports_predicate = True

    __slots__ = ("_as_of_instant", "_options", "_table_uri")

    def __init__(
        self,
        table_uri: str,
        *,
        as_of_instant: str | None = None,
        options: dict[str, str] | None = None,
    ) -> None:
        self._table_uri = table_uri
        self._as_of_instant = as_of_instant
        self._options = options or {}

    def _table(self) -> Any:
        hudi_table = _require_hudi()
        try:
            return hudi_table(self._table_uri, options=self._options)
        except Exception as exc:
            raise BackendError(f"failed to open Hudi table {self._table_uri!r}: {exc}") from exc

    def _snapshot(self, table: Any, filters: list[Any]) -> list[pa.RecordBatch]:
        if self._as_of_instant is not None:
            return table.read_snapshot_as_of(self._as_of_instant, filters)
        return table.read_snapshot(filters)

    def _read_table(self, projection: list[str] | None, predicate: dict | None = None) -> pa.Table:
        table = self._table()
        filters = _hudi_filters(predicate)
        try:
            try:
                batches = self._snapshot(table, filters)
            except Exception:
                # Backend rejected the pushed filters (version/format mismatch) →
                # read unfiltered; the engine's Filter still produces the right rows.
                batches = self._snapshot(table, [])
        except Exception as exc:
            raise BackendError(f"Hudi snapshot read failed for {self._table_uri!r}: {exc}") from exc
        result = pa.Table.from_batches(batches)
        return result.select(projection) if projection is not None else result

    def schema(self) -> pa.Schema:
        return self._table().get_schema()

    def read(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> list[pa.RecordBatch]:
        return self._read_table(projection, predicate).to_batches()

    def iter_batches(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> Iterator[pa.RecordBatch]:
        yield from self._read_table(projection, predicate).to_batches()

    def read_incremental(self, start_instant: str, end_instant: str | None = None) -> pa.Table:
        """Read rows changed between two Hudi instants as an Arrow table."""
        table = self._table()
        try:
            if end_instant is not None:
                batches = table.read_incremental_records(start_instant, end_instant)
            else:
                batches = table.read_incremental_records(start_instant)
            return pa.Table.from_batches(batches)
        except Exception as exc:
            raise BackendError(
                f"Hudi incremental read failed for {self._table_uri!r}: {exc}"
            ) from exc

    def row_count(self) -> int | None:
        return None  # hudi-rs exposes no cheap exact count without a scan.

    def identity(self) -> str:
        ref = self._as_of_instant or "latest"
        return f"hudi:{self._table_uri}@{ref}"

    def splits(self, target_size: int | None = None) -> list[Split]:  # noqa: ARG002
        return [WholeSourceSplit(self)]


@SINKS.register("hudi")
class HudiSink:
    """Placeholder Hudi sink — writes require the Spark/Flink write stack."""

    __slots__ = ()

    def __init__(self, *_: Any, **__: Any) -> None:
        raise BackendError("Hudi writes require Spark/Flink; Batcher supports Hudi reads only")
