"""Shared helpers for the Iceberg connector: the dependency gate and write tokens."""

from __future__ import annotations

import pyarrow as pa

from batcher._internal.errors import BackendError

__all__ = ["_new_write_token", "_require_pyiceberg", "_staged_schema"]


def _require_pyiceberg() -> None:
    """Raise `BackendError` if pyiceberg is not importable."""
    try:
        import pyiceberg  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise BackendError(
            "Iceberg support requires pyiceberg: pip install 'batcher-engine[iceberg]'"
        ) from exc


def _new_write_token() -> str:
    """A short token identifying one write, so staged file names differ between writes."""
    import uuid

    return uuid.uuid4().hex[:12]


def _staged_schema(path: str) -> pa.Schema:
    """The Arrow schema of a staged Parquet file (used to create the table if missing)."""
    import pyarrow.parquet as pq

    from batcher.io.filesystem import resolve_filesystem

    fs = resolve_filesystem(path)
    target = fs.native_read_target(path)
    if target is not None:
        pafs, in_path = target
        return pq.ParquetFile(in_path, filesystem=pafs).schema_arrow
    with fs.open(path) as fh:
        return pq.ParquetFile(fh).schema_arrow
