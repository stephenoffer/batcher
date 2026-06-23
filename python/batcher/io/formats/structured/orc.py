"""ORC format — lazy, projection-pushdown read + write via `pyarrow.orc`.

ORC support ships with core pyarrow (no extra), but the import is still deferred
so importing this module never forces the ORC reader to load. Reads expose
*stripe-level* splits — one `ORCStripeSplit` per ``(file, stripe)`` — so a
distributed read parallelizes within a file, reading only its assigned stripe via
``pyarrow.orc.ORCFile.read_stripe``. Row counts come from the footer (no scan).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import IO, Any

import pyarrow as pa

from batcher._internal.errors import BackendError
from batcher.io.base import FileSink, FileSource
from batcher.io.filesystem import resolve_filesystem
from batcher.io.formats.base import SINKS, SOURCES
from batcher.io.splits import Split

__all__ = ["ORCSink", "ORCSource", "ORCStripeSplit"]


def _require_orc() -> Any:
    """Import and return `pyarrow.orc` or raise `BackendError`."""
    try:
        import pyarrow.orc as orc
    except ImportError as exc:  # pragma: no cover - pyarrow ships orc by default
        raise BackendError(
            "ORC support requires pyarrow built with ORC: pip install 'batcher[all]'"
        ) from exc
    return orc


@dataclass(frozen=True, slots=True)
class ORCStripeSplit:
    """One stripe of a single ORC file, read in isolation on a worker.

    Carries only ``(path, stripe)`` so it pickles cheaply; `read` reopens the file
    and pulls just that stripe via ``ORCFile.read_stripe``.
    """

    path: str
    stripe: int

    def _file(self) -> Any:
        orc = _require_orc()
        fs = resolve_filesystem(self.path)
        return orc.ORCFile(fs.open(self.path))

    def read(self, projection: list[str] | None = None) -> list[pa.RecordBatch]:
        batch = self._file().read_stripe(self.stripe, columns=projection)
        return [batch]

    def schema(self) -> pa.Schema:
        return self._file().schema

    def iter_batches(self, projection: list[str] | None = None) -> Iterator[pa.RecordBatch]:
        yield from self.read(projection)

    def row_count(self) -> int | None:
        return self._file().read_stripe(self.stripe).num_rows

    def identity(self) -> str:
        return f"orc:{self.path}:stripe{self.stripe}"


@SOURCES.register("orc")
class ORCSource(FileSource):
    """One or more ORC files (single file, directory, or glob)."""

    suffix = ".orc"
    format_name = "orc"
    # Predicate pushdown: a pushed predicate → a pyarrow.dataset ORC filter, which
    # prunes stripes via their column statistics.
    supports_predicate = True

    __slots__ = ()

    def _read_schema(self, fh: IO[Any]) -> pa.Schema:
        orc = _require_orc()
        return orc.ORCFile(fh).schema

    def _read_file(self, fh: IO[Any], projection: list[str] | None) -> list[pa.RecordBatch]:
        orc = _require_orc()
        return orc.ORCFile(fh).read(columns=projection).to_batches()

    @staticmethod
    def _pa_filter(predicate: dict | None) -> Any:
        if predicate is None:
            return None
        from batcher.io.predicate import to_pyarrow_expression

        return to_pyarrow_expression(predicate)

    def read(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> list[pa.RecordBatch]:
        flt = self._pa_filter(predicate)
        if flt is None:
            return super().read(projection)
        import pyarrow.dataset as pads

        dataset = pads.dataset(self._files(), format="orc")
        return dataset.to_table(columns=projection, filter=flt).to_batches()

    def iter_batches(
        self, projection: list[str] | None = None, predicate: dict | None = None
    ) -> Iterator[pa.RecordBatch]:
        flt = self._pa_filter(predicate)
        if flt is None:
            yield from super().iter_batches(projection)
            return
        import pyarrow.dataset as pads

        dataset = pads.dataset(self._files(), format="orc")
        yield from dataset.to_batches(columns=projection, filter=flt)

    def _file_row_count(self, path: str) -> int | None:
        orc = _require_orc()
        with self._fs.open(path) as fh:
            return orc.ORCFile(fh).nrows

    def _file_splits(self, path: str, target_size: int | None) -> list[Split]:  # noqa: ARG002
        orc = _require_orc()
        with self._fs.open(path) as fh:
            nstripes = orc.ORCFile(fh).nstripes
        return [ORCStripeSplit(path, i) for i in range(nstripes)]


@SINKS.register("orc")
class ORCSink(FileSink):
    """Write an ORC file."""

    suffix = ".orc"
    format_name = "orc"

    __slots__ = ("compression",)

    def __init__(self, compression: str = "zstd") -> None:
        self.compression = compression

    def _write_file(self, table: pa.Table, fh: IO[Any]) -> None:
        orc = _require_orc()
        orc.write_table(table, fh, compression=self.compression)
