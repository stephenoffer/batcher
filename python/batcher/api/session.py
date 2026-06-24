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
from batcher.api.sql_session import Session
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
    "from_items",
    "from_numpy",
    "from_pandas",
    "from_polars",
    "from_pydict",
    "from_pylist",
    "from_ray_dataset",
    "from_spark",
    "from_tf",
    "from_torch",
    "read",
    "read_table",
    "register_function",
    "sql",
]


def engine_version() -> str:
    """Return the version string reported by the compiled Rust data plane.

    The version of the native ``bc_py`` extension, distinct from the Python
    package version. Useful for confirming which engine build is loaded.

    Returns:
        The engine version, e.g. ``"0.1.0"``.
    """
    from batcher import _native

    return _native.__engine_version__


# The process-global default SQL session. `bt.catalog` exposes it (so the
# established `bt.catalog.register/table/drop/clear/list` surface keeps working),
# and the module-level `sql` / `register_function` below delegate to it.
catalog = Session()


def sql(query: str, *, dialect: str | None = None, **tables: Any) -> Dataset:
    """Run a SQL query over named tables, returning a lazy `Dataset`.

    Each keyword binds a table name used in the query to a `Dataset` or a pyarrow
    table. The query is parsed and optimized through the same engine as the
    DataFrame API, so the two interoperate freely: the result is itself a lazy
    `Dataset` you can keep building on (``.filter``, ``.with_columns``, another
    ``sql``) before a terminal operation runs the whole plan.

    Unqualified names that are not passed here resolve from the `bt.catalog`
    registry, so ``bt.catalog.register("t", ds)`` lets later ``bt.sql("... FROM t")``
    calls omit the binding. Functions registered with `bt.register_function` are
    callable from the query. ``CREATE TABLE/VIEW AS`` and ``DROP TABLE`` register and
    unregister tables in `bt.catalog`. For an isolated catalog use `bt.Session`.

    Args:
        query: A SQL statement. Table names refer to the bound keywords.
        dialect: Override the sqlglot read dialect for this call (default ``duckdb``).
        **tables: Named inputs, each a `Dataset` or pyarrow table.

    Returns:
        A lazy `Dataset` of the query result.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> sales = bt.from_pydict({"region": ["w", "e", "w"], "amount": [10, 20, 30]})
            >>> out = bt.sql(
            ...     "SELECT region, SUM(amount) AS total "
            ...     "FROM sales GROUP BY region ORDER BY region",
            ...     sales=sales,
            ... )
            >>> out.to_pydict()
            {'region': ['e', 'w'], 'total': [20, 40]}
    """
    session = catalog if dialect is None else catalog._with_dialect(dialect)
    return session._run(query, tables)


def register_function(name: str, fn: Callable, **options: Any) -> None:
    """Register a Python function callable from `bt.sql` (the default session).

    Sugar for ``bt.catalog.register_function``; see `Session.register_function` for
    the call forms (scalar ``SELECT f(x)`` vs table ``SELECT * FROM f(t)``) and
    options.

    Args:
        name: The SQL name the function is called by.
        fn: The Python callable.
        **options: Forwarded to `Session.register_function`.
    """
    catalog.register_function(name, fn, **options)


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


def read_memory(name: str) -> Dataset:
    """Read the in-memory table written by a ``ds.write.memory(name, ...)`` query.

    The streaming `memory` sink accumulates each micro-batch under `name`; this
    snapshots the current contents as a `Dataset`. Raises `PlanError` if no query
    has written to `name`.
    """
    from batcher._internal.errors import PlanError
    from batcher.io.formats.streaming.sinks import memory_table

    try:
        table = memory_table(name)
    except KeyError:
        raise PlanError(f"no in-memory streaming sink named {name!r}") from None
    return from_arrow(table)


def streams() -> list[Any]:
    """List the currently-active streaming queries (Spark ``spark.streams.active``).

    Each entry is a handle to a query started by a streaming write that is still
    running, so you can track or stop it. Empty when no stream is active.

    Returns:
        The active streaming-query handles.
    """
    from batcher.api.streaming import active_streams

    return active_streams()


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
    """Create a `Dataset` from a column-oriented ``{name: values}`` dict.

    Each key is a column and each value its list of cells (all the same length);
    types are inferred by Arrow. The most direct way to get small in-memory data
    into the engine. Returns a lazy `Dataset` — no work runs until a terminal op.

    Args:
        mapping: Column name to its list of values.

    Returns:
        A lazy `Dataset` over the data.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"region": ["w", "e"], "amount": [10, 20]})
            >>> ds.to_pydict()
            {'region': ['w', 'e'], 'amount': [10, 20]}
    """
    return from_arrow(pa.table(mapping))


def from_pylist(rows: list[dict[str, Any]]) -> Dataset:
    """Create a `Dataset` from a row-oriented list of ``{column: value}`` dicts.

    The row-major counterpart to `from_pydict` (e.g. JSON records); the union of keys
    is the schema and missing keys are null. ``bt.from_pylist([{"a": 1}, {"a": 2}])``.
    """
    return from_arrow(pa.Table.from_pylist(rows))


def from_items(items: list[Any], *, column: str = "item") -> Dataset:
    """Create a `Dataset` from a list of items, one row per item (Ray Data style).

    Dict items expand to columns (like `from_pylist`); scalar/other items become a
    single `column`. ``bt.from_items([1, 2, 3])`` / ``bt.from_items([{"a": 1}])``.
    """
    return _scan(interop.from_items(items, column=column))


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
    """Create a single-column `Dataset` from a NumPy array under name `column`.

    The leading axis is the row axis: a 1-D array becomes a scalar column, an
    ``(n, dim)`` array a fixed-size-list column (the embedding convention), and a
    higher-rank array a fixed-shape-tensor column. Needs only ``numpy`` (core).

    Args:
        ndarray: The array to ingest; its first axis indexes rows.
        column: The name of the single output column.

    Returns:
        A lazy `Dataset` with one column over the array.
    """
    return _scan(interop.from_numpy(ndarray, column=column))


def from_pandas(df: Any) -> Dataset:
    """Create a `Dataset` from a pandas `DataFrame` via its Arrow bridge.

    Needs pandas (``pip install 'batcher-engine[pandas]'``); raises `BackendError`
    if it is absent. Goes through ``pyarrow.Table.from_pandas`` — no per-row Python.
    """
    return _scan(interop.from_pandas(df))


def from_polars(df: Any) -> Dataset:
    """Create a `Dataset` from a Polars `DataFrame` via its zero-copy Arrow export.

    Polars is Arrow-backed, so the buffers are referenced directly, not copied.
    Needs polars (``pip install 'batcher-engine[polars]'``); raises `BackendError`
    if it is absent.
    """
    return _scan(interop.from_polars(df))


def from_huggingface(hf_dataset: Any) -> Dataset:
    """Create a `Dataset` from a HuggingFace ``datasets.Dataset`` (Arrow-backed).

    HuggingFace datasets are Arrow tables underneath, so the table is taken
    directly. Needs ``datasets`` (``pip install 'batcher-engine[huggingface]'``);
    raises `BackendError` if it is absent.
    """
    return _scan(interop.from_huggingface(hf_dataset))


def from_torch(dataset_or_tensors: Any) -> Dataset:
    """Create a `Dataset` from a PyTorch tensor, tuple of tensors, or `Dataset`.

    Tensors are moved to CPU and adapted via their NumPy buffers (one column per
    tensor); only bulk buffers cross into the engine, never per-row Python. Needs
    ``torch`` (``pip install 'batcher-engine[torch]'``); raises `BackendError` if
    it is absent.
    """
    return _scan(interop.from_torch(dataset_or_tensors))


def from_tf(tf_dataset: Any) -> Dataset:
    """Create a `Dataset` from a ``tf.data.Dataset``, materializing it to Arrow.

    Each element's tensors are converted to NumPy and concatenated column-wise.
    Needs ``tensorflow`` (``pip install 'batcher-engine[tensorflow]'``); raises
    `BackendError` if it is absent.
    """
    return _scan(interop.from_tf(tf_dataset))


def from_spark(spark_df: Any) -> Dataset:
    """Create a `Dataset` from a Spark `DataFrame` by collecting it through Arrow.

    The Spark frame is collected to the driver via its Arrow bridge, so this
    materializes the data — for large frames write to a shared store and `read` it
    instead. Needs ``pyspark`` (``pip install 'batcher-engine[spark]'``); raises
    `BackendError` if it is absent.
    """
    return _scan(interop.from_spark(spark_df))


def from_dask(ddf: Any) -> Dataset:
    """Create a streaming `Dataset` from a Dask `DataFrame`, one partition per batch.

    Partitions stream lazily into the engine in bounded memory rather than being
    materialized at once. Needs ``dask`` (``pip install 'batcher-engine[dask]'``);
    raises `BackendError` if it is absent.
    """
    return _scan(interop.from_dask(ddf))


def from_ray_dataset(ray_dataset: Any) -> Dataset:
    """Create a streaming `Dataset` from a Ray Dataset (one Arrow block per batch).

    The migration on-ramp from Ray Data: blocks stream lazily into the engine in
    bounded memory. Requires `ray`.
    """
    return _scan(interop.from_ray_dataset(ray_dataset))
