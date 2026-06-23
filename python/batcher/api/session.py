"""Session entry points that create `Dataset`s.

The generic read dispatch (`read`/`read_table`, used by the `bt.read` namespace),
in-memory ingestion (`from_arrow`/`from_pydict`/`from_batches`), and the
framework-interop constructors (`from_pandas`/`from_polars`/…). All build a
single-`Scan` `Dataset` over a lazy `Source`; cloud and streaming sources slot in
behind the same `Source` protocol.

The `from_*` framework adapters wrap the `Source`-building functions in
`batcher.io.interop` and lift the result into a `Dataset` — that module stays
`Dataset`-free (no import cycle, no optional framework pulled in on import).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import Any

import pyarrow as pa

from batcher.api.dataset import Dataset
from batcher.io import interop
from batcher.io.detect import detect_format
from batcher.io.formats.base import SOURCES
from batcher.io.source import (
    InMemorySource,
    IteratorSource,
    Source,
)
from batcher.plan.logical import Scan
from batcher.plan.schema import SchemaRef

__all__ = [
    "engine_version",
    "from_arrow",
    "from_batches",
    "from_dask",
    "from_huggingface",
    "from_numpy",
    "from_pandas",
    "from_polars",
    "from_pydict",
    "from_spark",
    "from_tf",
    "from_torch",
    "read",
    "read_table",
    "sql",
]


def engine_version() -> str:
    """Return the version reported by the compiled Rust engine."""
    from batcher import _native

    return _native.__engine_version__


class _Catalog:
    """A process-local registry of named tables for SQL and cross-query reuse.

    ``bt.catalog.register("customers", ds)`` then ``bt.sql("SELECT * FROM customers")``
    or ``bt.catalog.table("customers")``. Names passed explicitly to `bt.sql` override
    registered ones. This is control-plane metadata only — registering does not execute.
    """

    __slots__ = ("_tables",)

    def __init__(self) -> None:
        self._tables: dict[str, Dataset] = {}

    def register(self, name: str, dataset: Dataset) -> Dataset:
        """Register `dataset` under `name` (replacing any existing entry)."""
        self._tables[name] = dataset
        return dataset

    def table(self, name: str) -> Dataset:
        """Return the dataset registered as `name` (raises `PlanError` if absent)."""
        if name not in self._tables:
            from batcher._internal.errors import PlanError

            raise PlanError(f"no table {name!r} in catalog; registered: {self.list()}")
        return self._tables[name]

    def list(self) -> list[str]:
        """The sorted names of all registered tables."""
        return sorted(self._tables)

    def drop(self, name: str) -> None:
        """Remove `name` from the catalog (no error if absent)."""
        self._tables.pop(name, None)

    def clear(self) -> None:
        """Remove every registered table."""
        self._tables.clear()


catalog = _Catalog()


def sql(query: str, **tables: Any) -> Dataset:
    """Run a SQL query over named tables (`Dataset`s or pyarrow tables).

    Unqualified table names resolve from the `bt.catalog` registry when not passed
    explicitly here, so ``bt.catalog.register("t", ds); bt.sql("SELECT * FROM t")`` works.
    """
    from batcher._sql import sql as _sql_fn

    resolved = {**catalog._tables, **tables}
    return _sql_fn(query, **resolved)


def _scan(source: Source) -> Dataset:
    plan = Scan(source_id=0, schema=SchemaRef.from_arrow(source.schema()))
    return Dataset(plan, sources=[source])


def read(path: str, *, format: str | None = None, **opts: Any) -> Dataset:
    """Read a file/object-store dataset, dispatching on `format` or the path.

    With no `format`, it is inferred from the URI scheme (``delta://``…) or the
    file extension. ``read("s3://b/*.parquet")`` → Parquet; ``read("data/",
    format="csv")``. For database/catalog sources use `read_table` or the typed
    ``read_*`` helpers.
    """
    fmt = detect_format(path, format)
    return _scan(SOURCES.get(fmt)(path, **opts))


def read_table(format: str, *args: Any, **opts: Any) -> Dataset:
    """Read a registered non-file source by name (lakehouse/SQL/NoSQL/streaming).

    ``read_table("delta", "s3://bucket/table", version=3)`` constructs the
    registered ``delta`` source. The typed ``read_*`` helpers wrap this for the
    common backends.
    """
    return _scan(SOURCES.get(format)(*args, **opts))


def from_arrow(data: pa.Table | pa.RecordBatch | Sequence[pa.RecordBatch]) -> Dataset:
    """Create a `Dataset` from an Arrow table, record batch, or list of batches.

    An empty (zero-row) table or batch is allowed — its schema is preserved via a
    single empty morsel, so an empty input flows through the engine like any other.
    A bare empty sequence of batches carries no schema and is rejected.
    """
    if isinstance(data, pa.Table):
        # A zero-row Table yields no batches; keep its schema with one empty morsel.
        batches = data.to_batches() or [_empty_batch(data.schema)]
    elif isinstance(data, pa.RecordBatch):
        batches = [data]
    else:
        batches = list(data)
        if not batches:
            raise ValueError(
                "from_arrow() requires at least one record batch (a bare empty "
                "sequence carries no schema; pass an empty pa.Table instead)"
            )
    return _scan(InMemorySource(batches))


def _empty_batch(schema: pa.Schema) -> pa.RecordBatch:
    """A zero-row RecordBatch carrying `schema` (so empty inputs keep their types)."""
    return pa.RecordBatch.from_arrays([pa.array([], type=f.type) for f in schema], schema=schema)


def from_pydict(mapping: dict[str, list[Any]]) -> Dataset:
    """Create a `Dataset` from a column-oriented Python dict."""
    return from_arrow(pa.table(mapping))


def compact(
    path: str,
    *,
    target_size_mb: float = 128.0,
    num_files: int | None = None,
    by: str | list[str] | None = None,
    format: str | None = None,
    **opts: Any,
) -> Any:
    """Compact a dataset in place — rewrite many small files into fewer, larger ones.

    The fix for the small-files problem (tiny part files from incremental writes):
    reads `path`, repartitions to ~`target_size_mb` files (or exactly `num_files`,
    optionally Hive-partitioned `by` column), writes the result back, and deletes the
    now-stale part-files the rewrite replaced. The data is fully materialized before
    the overwrite, so the rewrite is safe. Single-writer only. Returns the
    `WriteManifest` of the compacted output.
    """
    import os

    from batcher.io.detect import detect_format
    from batcher.io.filesystem import resolve_filesystem
    from batcher.io.formats.base import SOURCES

    fmt = detect_format(path, format)
    fs = resolve_filesystem(path)
    suffix = getattr(SOURCES.get(fmt), "suffix", "")
    try:
        old_files = list(fs.expand(path, suffix=suffix))
    except OSError:
        old_files = []

    spec: dict[str, Any] = {"by": by} if by is not None else {}
    if num_files is not None:
        spec["num_files"] = num_files
    else:
        spec["target_size_mb"] = target_size_mb
    manifest = (
        read(path, format=fmt).repartition(**spec).write(path, format=fmt, mode="overwrite", **opts)
    )

    new_names = {os.path.basename(f.path) for f in manifest.files}
    for f in old_files:
        if os.path.basename(f) not in new_names:
            fs.remove(f)
    return manifest


def range(start: int, stop: int, step: int = 1, *, name: str = "value") -> Dataset:
    """A one-column `Dataset` of the integers ``[start, stop)`` stepped by `step`.

    The generator source for synthetic keys and joins; for date dimensions see
    `date_range`. ``bt.range(0, 5)`` → ``value = 0,1,2,3,4``.
    """
    import builtins

    values = list(builtins.range(start, stop, step))
    return from_arrow(pa.table({name: pa.array(values, pa.int64())}))


def date_range(start: str, end: str, *, interval_days: int = 1, name: str = "date") -> Dataset:
    """A one-column `Dataset` of dates from `start` to `end` (both ISO ``YYYY-MM-DD``,
    inclusive) stepped by `interval_days` — the calendar/date-dimension generator.

    ``bt.date_range("2024-01-01", "2024-12-31")`` builds a daily date dimension.
    """
    import builtins
    from datetime import date, timedelta

    s, e = date.fromisoformat(start), date.fromisoformat(end)
    if interval_days < 1:
        raise ValueError("date_range(): interval_days must be >= 1")
    days = [s + timedelta(days=i) for i in builtins.range(0, (e - s).days + 1, interval_days)]
    return from_arrow(pa.table({name: pa.array(days, pa.date32())}))


def from_batches(
    factory: Callable[[], Iterator[pa.RecordBatch]],
    schema: pa.Schema,
    *,
    bounded: bool = True,
) -> Dataset:
    """Create a streaming `Dataset` from a re-iterable batch factory.

    `factory()` must return a fresh iterator of `pyarrow.RecordBatch` each call.
    Combined with `Dataset.iter_batches()`, a breaker-free pipeline (filter /
    project / map_batches) over this source is consumed one batch at a time in
    bounded memory — the path for unbounded or larger-than-memory inputs.

    Pass ``bounded=False`` for a genuinely infinite stream so terminal operations
    that must materialize (`collect`, `count`, `to_*`) fail fast instead of hanging.
    """
    return _scan(IteratorSource(factory, schema, bounded=bounded))


# --- Framework-interop constructors (foreign object → Dataset) -------------
def from_numpy(ndarray: Any, *, column: str = "data") -> Dataset:
    """Create a `Dataset` from a NumPy array (1-D scalar or 2-D fixed-size-list)."""
    return _scan(interop.from_numpy(ndarray, column=column))


def from_pandas(df: Any) -> Dataset:
    """Create a `Dataset` from a pandas `DataFrame` (zero-copy via Arrow)."""
    return _scan(interop.from_pandas(df))


def from_polars(df: Any) -> Dataset:
    """Create a `Dataset` from a Polars `DataFrame` (zero-copy Arrow export)."""
    return _scan(interop.from_polars(df))


def from_huggingface(hf_dataset: Any) -> Dataset:
    """Create a `Dataset` from a HuggingFace `datasets.Dataset` (Arrow-backed)."""
    return _scan(interop.from_huggingface(hf_dataset))


def from_torch(dataset_or_tensors: Any) -> Dataset:
    """Create a `Dataset` from a PyTorch tensor, tuple of tensors, or `Dataset`."""
    return _scan(interop.from_torch(dataset_or_tensors))


def from_tf(tf_dataset: Any) -> Dataset:
    """Create a `Dataset` from a ``tf.data.Dataset`` (materialized to Arrow)."""
    return _scan(interop.from_tf(tf_dataset))


def from_spark(spark_df: Any) -> Dataset:
    """Create a `Dataset` from a Spark `DataFrame` (Arrow collection)."""
    return _scan(interop.from_spark(spark_df))


def from_dask(ddf: Any) -> Dataset:
    """Create a streaming `Dataset` from a Dask `DataFrame` (one partition per batch)."""
    return _scan(interop.from_dask(ddf))
