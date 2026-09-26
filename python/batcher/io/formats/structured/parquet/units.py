"""Hand a Parquet scan to the engine to read row group by row group, when that reads the same rows.

The resident read decodes every file on the control plane's threads before the engine starts,
then hands the engine the whole relation. The engine can instead read the row groups itself,
each worker decoding its own and pushing them through its pipeline as they land
(`core.execute_local_parquet`). That overlaps decoding with computing and never holds the
relation, but it skips everything `FileSource.read` does per file, so it is offered only where
none of that could change a row:

* every file local and reachable by the native reader, the one both paths use to decode. Local
  because the path's concurrency is one read per engine worker: that is every core on a local
  disk, where a read is decode-bound, but on object storage a read is latency-bound and the
  resident read keeps hundreds of requests in flight (`bc_io`'s file and row-group concurrency),
  so a remote source keeps it until the row-group read has been measured there;
* no row cap (`n_rows`) and no tolerance for unreadable files (`on_error="raise"`): a capped read
  stops early and a tolerant one drops files, neither of which a row-group read reproduces;
* every file's schema identical to the one the source declares, so the per-file conformance
  step (`FileSource._normalize`, schema evolution's typed-null fill and casts) is the identity;
* a pushed predicate only when it has a native translation — the engine keeps its `Filter`
  either way, so an untranslatable one is simply not pushed.

Anything else returns `None` and the query reads the source as it always has.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from batcher.io.formats.structured.parquet.source import ParquetSource

__all__ = ["ParquetUnitRead", "unit_read"]

#: Past this many files the eligibility check reads a footer schema per file, which is a
#: round trip each on object storage; such a source keeps the resident read, whose batched
#: multi-file path overlaps those round trips.
MAX_FILES = 4096


@dataclass(frozen=True, slots=True)
class ParquetUnitRead:
    """What the engine needs to read a Parquet source row group by row group.

    Attributes:
        uris: The files, in the order the resident read concatenates them.
        columns: The pushed projection, or None for every column.
        predicate: The pushed predicate's native translation (JSON), or None.
        batch_size: Rows to decode at once.
    """

    uris: list[str]
    columns: list[str] | None
    predicate: str | None
    batch_size: int


def unit_read(
    source: ParquetSource, projection: list[str] | None, predicate: dict | None
) -> ParquetUnitRead | None:
    """The row-group read for `source`, or None when it could return different rows.

    Args:
        source: The Parquet source the scan reads.
        projection: The scan's pushed projection.
        predicate: The scan's pushed predicate, in the control plane's form.

    Returns:
        The read, or None when any condition in the module note does not hold.
    """
    import json

    from batcher.io.formats.structured import _parquet_native
    from batcher.io.predicate import to_native_predicate

    if source._n_rows is not None or source._errors.mode != "raise":
        return None
    files = source._files()
    if not files or len(files) > MAX_FILES:
        return None
    if not all("://" not in f and source._native_uri_is_addressable(f) for f in files):
        return None
    columns = source._effective_projection(projection)
    declared = source.schema()
    wanted = declared if columns is None else _select(declared, columns)
    if wanted is None:
        return None
    for f in files:
        on_disk = _select(source._file_schema(f), wanted.names)
        if on_disk is None or not on_disk.equals(wanted):
            return None
    native = to_native_predicate(predicate) if predicate else None
    return ParquetUnitRead(
        uris=list(files),
        columns=None if columns is None else list(columns),
        predicate=None if native is None else json.dumps(native),
        batch_size=_parquet_native.native_read_batch(declared, columns),
    )


def _select(schema, names):
    """`schema` narrowed to `names`, in that order, or None when one is missing."""
    import pyarrow as pa

    try:
        return pa.schema([schema.field(n) for n in names])
    except KeyError:
        return None
