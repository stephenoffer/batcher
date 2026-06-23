"""Lance format — columnar, random-access read + write via `pylance` (lance).

Lance is a columnar table format built for fast random access and vector search.
`LanceSource` reads a `lance.LanceDataset` as Arrow with projection and predicate
pushdown, exposing *fragment-level* splits — one `LanceFragmentSplit` per dataset
fragment — so a distributed read scans only its assigned fragment. Unlike Parquet,
Lance is designed for cheap random row access (``take``), which Batcher uses for
point lookups; bulk scans here are the streaming path.

All `lance` imports are deferred — importing this module never requires the
optional dependency. A missing dependency raises `BackendError` with a
``pip install 'batcher[lance]'`` hint.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import pyarrow as pa

from batcher._internal.errors import BackendError
from batcher.io.formats.base import SINKS, SOURCES
from batcher.io.manifest import WrittenFile
from batcher.io.splits import Split

__all__ = ["LanceFragmentSplit", "LanceSink", "LanceSource"]


def _require_lance() -> Any:
    """Import and return the `lance` module or raise `BackendError`."""
    try:
        import lance
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise BackendError("Lance support requires pylance: pip install 'batcher[lance]'") from exc
    return lance


# Per-process cache: uri -> (dataset, {fragment_id: fragment}). A worker opens a
# Lance dataset once and reuses the id→fragment index across the splits it reads,
# instead of reopening + linear-scanning per read (which would be O(fragments^2)).
_LANCE_FRAGMENTS: dict[str, tuple[Any, dict[int, Any]]] = {}
_LANCE_CACHE_MAX = 8


def _lance_fragments(uri: str) -> tuple[Any, dict[int, Any]]:
    cached = _LANCE_FRAGMENTS.get(uri)
    if cached is None:
        dataset = _require_lance().LanceDataset(uri)
        index = {frag.fragment_id: frag for frag in dataset.get_fragments()}
        if len(_LANCE_FRAGMENTS) >= _LANCE_CACHE_MAX:
            _LANCE_FRAGMENTS.clear()
        cached = (dataset, index)
        _LANCE_FRAGMENTS[uri] = cached
    return cached


def _lance_filter(base: str | None, predicate: dict | None) -> str | None:
    """Combine a ctor SQL filter string with a Kyber-pushed predicate IR.

    Lance scanners take a SQL-style filter string, so the pushable subset of the
    predicate translates via `to_sql_where`; both filters are AND-combined.
    """
    extra = None
    if predicate is not None:
        from batcher.io.predicate import to_sql_where

        extra = to_sql_where(predicate)
    parts = [p for p in (base, extra) if p]
    return " AND ".join(f"({p})" for p in parts) if parts else None


@dataclass(frozen=True, slots=True)
class LanceFragmentSplit:
    """One fragment of a Lance dataset, scanned in isolation on a worker.

    Carries only ``(uri, fragment_id, predicate)`` so it pickles cheaply; `read`
    reopens the dataset and scans just that fragment with the pushed-down filter.
    """

    uri: str
    fragment_id: int
    predicate: str | None = None

    def _fragment(self) -> Any:
        # Open the dataset + index its fragments by id ONCE per worker (cached),
        # not per read — a per-read reopen+linear-scan would be O(fragments^2).
        _, index = _lance_fragments(self.uri)
        frag = index.get(self.fragment_id)
        if frag is None:
            raise BackendError(f"Lance fragment {self.fragment_id} not found in {self.uri!r}")
        return frag

    def schema(self) -> pa.Schema:
        return self._fragment().schema

    def read(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> list[pa.RecordBatch]:
        flt = _lance_filter(self.predicate, predicate)
        return self._fragment().scanner(columns=projection, filter=flt).to_table().to_batches()

    def iter_batches(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> Iterator[pa.RecordBatch]:
        flt = _lance_filter(self.predicate, predicate)
        yield from self._fragment().scanner(columns=projection, filter=flt).to_batches()

    def row_count(self) -> int | None:
        return self._fragment().count_rows()

    def identity(self) -> str:
        return f"lance:{self.uri}:frag{self.fragment_id}"


@SOURCES.register("lance")
class LanceSource:
    """A Lance dataset read as Arrow with projection + predicate pushdown.

    Args:
        uri: The dataset root (local path or cloud URI).
        predicate: Optional SQL-style filter string pushed down to the Lance
            scanner (e.g. ``"x > 10 AND y = 'a'"``).
    """

    # Predicate pushdown: Kyber's pushed predicate → a Lance SQL filter string
    # (combined with any constructor `predicate`).
    supports_predicate = True

    __slots__ = ("_predicate", "_uri")

    def __init__(self, uri: str, *, predicate: str | None = None) -> None:
        self._uri = uri
        self._predicate = predicate

    def _dataset(self) -> Any:
        lance = _require_lance()
        try:
            return lance.LanceDataset(self._uri)
        except Exception as exc:
            raise BackendError(f"failed to open Lance dataset {self._uri!r}: {exc}") from exc

    def schema(self) -> pa.Schema:
        return self._dataset().schema

    def read(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> list[pa.RecordBatch]:
        flt = _lance_filter(self._predicate, predicate)
        return self._dataset().scanner(columns=projection, filter=flt).to_table().to_batches()

    def iter_batches(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> Iterator[pa.RecordBatch]:
        flt = _lance_filter(self._predicate, predicate)
        yield from self._dataset().scanner(columns=projection, filter=flt).to_batches()

    def row_count(self) -> int | None:
        return self._dataset().count_rows()

    def identity(self) -> str:
        return f"lance:{self._uri}"

    def splits(self, target_size: int | None = None) -> list[Split]:  # noqa: ARG002
        fragments = self._dataset().get_fragments()
        return [LanceFragmentSplit(self._uri, f.fragment_id, self._predicate) for f in fragments]


@SINKS.register("lance")
class LanceSink:
    """Write a Lance dataset.

    Args:
        mode: ``"create"`` (default), ``"append"``, or ``"overwrite"`` — passed
            through to ``lance.write_dataset``.
    """

    __slots__ = ("mode",)

    def __init__(self, mode: str = "create") -> None:
        self.mode = mode

    def write(self, table: pa.Table, path: str) -> WrittenFile:
        """Write `table` as a Lance dataset rooted at `path`."""
        lance = _require_lance()
        try:
            lance.write_dataset(table, path, mode=self.mode)
        except Exception as exc:
            raise BackendError(f"failed to write Lance dataset {path!r}: {exc}") from exc
        return WrittenFile(path=path, rows=table.num_rows, bytes=0)
