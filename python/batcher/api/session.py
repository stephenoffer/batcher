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
from batcher.plan.logical import LogicalPlan, Scan
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

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> isinstance(bt.engine_version(), str)
            True
    """
    from batcher import _native

    return _native.__engine_version__


# The process-global default SQL session, backing the module-level `sql` /
# `register_function` below. It is intentionally private: `bt.sql(...)` is the one
# obvious entry point for the default catalog, and `bt.Session` is the public handle
# for an isolated one.
_catalog = Session()


def sql(query: str, *, dialect: str | None = None, **tables: Any) -> Dataset:
    """Run a SQL query over named tables, returning a lazy `Dataset`.

    Each keyword binds a table name used in the query to a `Dataset` or a pyarrow
    table. The query is parsed and optimized through the same engine as the
    DataFrame API, so the two interoperate freely: the result is itself a lazy
    `Dataset` you can keep building on (``.filter``, ``.with_columns``, another
    ``sql``) before a terminal operation runs the whole plan.

    Names not passed here resolve from the default catalog, which ``CREATE
    TABLE/VIEW AS`` populates and ``DROP TABLE`` clears, so a later ``bt.sql("...
    FROM t")`` can omit the binding. Functions registered with `bt.register_function`
    are callable from the query. For an isolated catalog use `bt.Session`.

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
    session = _catalog if dialect is None else _catalog._with_dialect(dialect)
    return session._run(query, tables)


def register_function(name: str, fn: Callable, **options: Any) -> None:
    """Register a Python function callable from `bt.sql` (the default session).

    Registers on the default catalog; see `Session.register_function` for the call
    forms (scalar ``SELECT f(x)`` vs table ``SELECT * FROM f(t)``) and options. For an
    isolated registry use `bt.Session`.

    Args:
        name: The SQL name the function is called by.
        fn: The Python callable.
        **options: Forwarded to `Session.register_function`.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> import pyarrow.compute as pc
            >>> bt.register_function("dbl", lambda a: pc.multiply(a, 2), result_type="int64")
            >>> t = bt.from_pydict({"x": [1, 2, 3]})
            >>> bt.sql("SELECT dbl(x) AS y FROM t", t=t).to_pydict()
            {'y': [2, 4, 6]}
    """
    _catalog.register_function(name, fn, **options)


def _scan(source: Source) -> Dataset:
    """Build the `Dataset` for `source`, governed by the active security policy.

    The single place a source becomes a plan, and therefore the single place governance
    has to be applied for it to be unbypassable — see `api.security`.
    """
    from batcher.api.security import govern_scan

    plan: LogicalPlan = Scan(source_id=0, schema=SchemaRef.from_arrow(source.schema()))
    return Dataset(govern_scan(plan, source), sources=[source])


def read(path: str, *, format: str | None = None, **opts: Any) -> Dataset:
    """Read a file/object-store dataset, dispatching on `format` or the path.

    With no `format`, it is inferred from the URI scheme (``delta://``…) or the
    file extension. ``read("s3://b/*.parquet")`` → Parquet; ``read("data/",
    format="csv")``. For database/catalog sources use `read_table` or the typed
    ``read_*`` helpers.

    Examples:
        .. doctest::

            >>> import tempfile, os
            >>> import batcher as bt
            >>> path = os.path.join(tempfile.mkdtemp(), "t.parquet")
            >>> _ = bt.from_pydict({"x": [1, 2, 3]}).write(path, format="parquet")
            >>> bt.read(path).count()
            3
    """
    fmt = detect_format(path, format)
    return _scan(SOURCES.get(fmt)(path, **opts))


def read_table(format: str, *args: Any, **opts: Any) -> Dataset:
    """Read a registered non-file source by name (lakehouse/SQL/NoSQL/streaming).

    ``read_table("delta", "s3://bucket/table", version=3)`` constructs the
    registered ``delta`` source. The typed ``read_*`` helpers wrap this for the
    common backends.

    Examples:
        .. code-block:: python

            import batcher as bt

            ds = bt.read.table("delta", "s3://bucket/table", version=3)
    """
    return _scan(SOURCES.get(format)(*args, **opts))


def read_memory(name: str) -> Dataset:
    """Read the in-memory table written by a ``ds.write.memory(name, ...)`` query.

    The streaming `memory` sink accumulates each micro-batch under `name`; this
    snapshots the current contents as a `Dataset`. Raises `PlanError` if no query
    has written to `name`.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> query = bt.from_pydict({"x": [1, 2, 3]}).write.memory("demo")
            >>> _ = query.await_termination()
            >>> bt.read_memory("demo").count()
            3

    Args:
        name: The in-memory sink name a streaming write accumulated into.

    Returns:
        A `Dataset` snapshotting the current contents of the named sink.

    Raises:
        PlanError: If no query has written to `name`.
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

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> bt.streams()
            []
    """
    from batcher.api.streaming import active_streams

    return active_streams()


def await_any_termination(timeout: float | None = None) -> bool:
    """Block until any active streaming query stops (Spark ``awaitAnyTermination``).

    Waits for the first currently-running query to terminate, re-raising its exception
    if it failed. Returns immediately when no query is active.

    Args:
        timeout: Maximum seconds to wait; ``None`` waits indefinitely.

    Returns:
        ``True`` if a query stopped (or none were active), ``False`` on timeout.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> bt.await_any_termination(timeout=0.0)
            True
    """
    from batcher.api.streaming import await_any_termination as _await_any

    return _await_any(timeout)


def from_arrow(data: pa.Table | pa.RecordBatch | Sequence[pa.RecordBatch]) -> Dataset:
    """Create a `Dataset` from an Arrow table, record batch, or list of batches.

    An empty (zero-row) table or batch is allowed — its schema is preserved via a
    single empty morsel, so an empty input flows through the engine like any other.
    A bare empty sequence of batches carries no schema and is rejected.

    Examples:
        .. doctest::

            >>> import pyarrow as pa
            >>> import batcher as bt
            >>> bt.from_arrow(pa.table({"x": [1, 2]})).to_pydict()
            {'x': [1, 2]}

    Args:
        data: An Arrow table, record batch, or sequence of record batches.

    Returns:
        A lazy `Dataset` over the Arrow data.

    Raises:
        ValueError: If `data` is an empty sequence carrying no schema.
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

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> bt.from_pylist([{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]).to_pydict()
            {'a': [1, 2], 'b': ['x', 'y']}

    Args:
        rows: A list of ``{column: value}`` dicts; the union of keys is the schema.

    Returns:
        A lazy `Dataset` over the rows.
    """
    return from_arrow(pa.Table.from_pylist(rows))


def from_items(items: list[Any], *, column: str = "item") -> Dataset:
    """Create a `Dataset` from a list of items, one row per item (Ray Data style).

    Dict items expand to columns (like `from_pylist`); scalar/other items become a
    single `column`. ``bt.from_items([1, 2, 3])`` / ``bt.from_items([{"a": 1}])``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> bt.from_items([1, 2, 3]).to_pydict()
            {'item': [1, 2, 3]}

    Args:
        items: The items, one row each; dict items expand to columns.
        column: The single-column name used for scalar (non-dict) items.

    Returns:
        A lazy `Dataset` with one row per item.
    """
    return _scan(interop.from_items(items, column=column))


def compact(
    path: str,
    *,
    target_size_mb: float = 128.0,
    num_files: int | None = None,
    by: str | list[str] | None = None,
    z_order: list[str] | None = None,
    where: Any = None,
    format: str | None = None,
    **opts: Any,
) -> Any:
    """Compact a dataset in place — rewrite many small files into fewer, larger ones.

    The fix for the small-files problem (tiny part files from incremental writes). What it
    *does* depends on what `path` is, and the difference matters:

    * A **transactional table** (Delta) is compacted as a transaction. Small files are
      bin-packed and a new version retires the old ones *from the log*, leaving them on
      storage — so every existing version still reads and time travel survives. Nothing is
      deleted here; `vacuum` is what reclaims. Pass `z_order=[...]` to sort the rewritten
      rows along a Z-curve over those columns, which narrows each file's min/max bounds and
      so multiplies what the *next* query can skip from the log alone. `where` restricts
      the work to matching partitions.
    * A **plain file directory** (Parquet, CSV, ...) has no log, so it is read,
      repartitioned to ~`target_size_mb` files (or exactly `num_files`, optionally
      Hive-partitioned by `by`), written back, and the replaced part-files removed. Nothing
      references the old files, so removing them is safe. Single-writer only.

    Examples:
        .. doctest::

            >>> import tempfile, os, glob
            >>> import batcher as bt
            >>> d = tempfile.mkdtemp()
            >>> _ = bt.from_pydict({"x": [1, 2, 3, 4]}).repartition(num_files=2).write(
            ...     d, format="parquet"
            ... )
            >>> _ = bt.compact(d, num_files=1, format="parquet")
            >>> len(glob.glob(os.path.join(d, "*.parquet")))
            1

    Args:
        path: The dataset location to compact in place.
        target_size_mb: Approximate target size per output file (ignored if
            `num_files` is given).
        num_files: Exact number of output files to rewrite to (file directories only).
        by: Column(s) to Hive-partition the rewritten output by (file directories only).
        z_order: Columns to Z-order the rewritten rows by (transactional tables only).
        where: Partition filters limiting the scope (transactional tables only).
        format: The dataset format; inferred from `path` when omitted.
        **opts: Extra options forwarded to the writer / maintenance backend.

    Returns:
        The `WriteManifest` for a file directory; the backend's optimize metrics for a
        transactional table.
    """
    import os

    from batcher.io.detect import detect_format
    from batcher.io.filesystem import resolve_filesystem
    from batcher.io.formats.base import SOURCES
    from batcher.io.formats.lakehouse.maintenance import table_maintenance

    fmt = detect_format(path, format)

    # A transactional table must be maintained transactionally. The file rewrite below
    # deletes what it replaces, and a table's older versions still *reference* those files
    # — doing it to a Delta table silently destroys time travel (and destroys it
    # invisibly, since `count()` keeps answering from the log after the data is gone).
    maintenance = table_maintenance(fmt)
    if maintenance is not None:
        target = None if num_files is not None else int(target_size_mb * 1024 * 1024)
        return maintenance.compact(
            path, target_size_bytes=target, z_order=z_order, where=where, **opts
        )
    if z_order is not None or where is not None:
        from batcher._internal.errors import PlanError

        raise PlanError(
            f"compact(): z_order/where need a transactional table; {fmt!r} is a file "
            "directory with no transaction log. Use num_files/target_size_mb, or write "
            "to a Delta table."
        )

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


def vacuum(
    path: str,
    *,
    retention_hours: float | None = None,
    dry_run: bool = True,
    format: str | None = None,
    **opts: Any,
) -> list[str]:
    """Reclaim the data files of a transactional table that no live version references.

    The counterpart to `compact`. Compaction never deletes — it rewrites small files and
    retires the old ones from the log, leaving them on storage so time travel still works.
    This is the operation that eventually removes them, and it is the only one allowed to.

    It **defaults to a dry run**, reporting what it would delete and deleting nothing,
    because the files it removes are precisely the ones older versions and any in-flight
    reader depend on. The retention window is the safety argument: a file is only removed
    once it has been unreferenced for longer than any reader could still be using it.
    Shortening the window below the table's configured minimum means an active reader can
    have its files deleted mid-scan, so the backend refuses unless you waive the check
    explicitly.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> would_delete = bt.vacuum("s3://lake/events")  # doctest: +SKIP
            >>> bt.vacuum("s3://lake/events", dry_run=False)  # doctest: +SKIP

    Args:
        path: The table root.
        retention_hours: How long an unreferenced file is kept before it can be
            reclaimed. Defaults to the format's own default (7 days for Delta).
        dry_run: When True (the default), report the files but delete nothing.
        format: The table format; inferred from `path` when omitted.
        **opts: Backend options (e.g. ``storage_options``).

    Returns:
        The files deleted — or, on a dry run, the files that would be.

    Raises:
        PlanError: If `path` is not a transactional table (a plain file directory has no
            log, so nothing is unreferenced and there is nothing to reclaim).
    """
    from batcher.io.detect import detect_format
    from batcher.io.formats.lakehouse.maintenance import table_maintenance

    fmt = detect_format(path, format)
    maintenance = table_maintenance(fmt)
    if maintenance is None:
        from batcher._internal.errors import PlanError

        raise PlanError(
            f"vacuum() needs a transactional table; {fmt!r} is a plain file directory "
            "with no transaction log, so no file is unreferenced and there is nothing "
            "to reclaim."
        )
    return maintenance.vacuum(path, retention_hours=retention_hours, dry_run=dry_run, **opts)


def range(start: int, stop: int, step: int = 1, *, name: str = "value") -> Dataset:
    """A one-column `Dataset` of the integers ``[start, stop)`` stepped by `step`.

    The generator source for synthetic keys and joins; for date dimensions see
    `date_range`. ``bt.range(0, 5)`` → ``value = 0,1,2,3,4``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> bt.range(0, 5).to_pydict()
            {'value': [0, 1, 2, 3, 4]}

    Args:
        start: The first integer (inclusive).
        stop: The end integer (exclusive).
        step: The stride between successive integers.
        name: The output column name.

    Returns:
        A one-column lazy `Dataset` of the integer range.
    """
    import builtins

    values = list(builtins.range(start, stop, step))
    return from_arrow(pa.table({name: pa.array(values, pa.int64())}))


def date_range(start: str, end: str, *, interval_days: int = 1, name: str = "date") -> Dataset:
    """A one-column `Dataset` of dates from `start` to `end`, stepped by `interval_days`.

    Both bounds are inclusive ISO ``YYYY-MM-DD`` strings — the calendar /
    date-dimension generator. ``bt.date_range("2024-01-01", "2024-12-31")`` builds a
    daily date dimension.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> bt.date_range("2024-01-01", "2024-01-03").count()
            3

    Args:
        start: The first date (inclusive), ISO ``YYYY-MM-DD``.
        end: The last date (inclusive), ISO ``YYYY-MM-DD``.
        interval_days: The stride in days between successive dates.
        name: The output column name.

    Returns:
        A one-column lazy `Dataset` of the date range.

    Raises:
        ValueError: If `interval_days` is less than 1.
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

    Examples:
        .. doctest::

            >>> import pyarrow as pa
            >>> import batcher as bt
            >>> schema = pa.schema([("x", pa.int64())])
            >>> ds = bt.from_batches(lambda: iter([pa.record_batch({"x": [1, 2, 3]})]), schema)
            >>> ds.count()
            3

    Args:
        factory: A callable returning a fresh iterator of record batches each call.
        schema: The Arrow schema of the produced batches.
        bounded: Whether the stream is finite; ``False`` makes materializing terminal
            operations fail fast rather than hang.

    Returns:
        A streaming lazy `Dataset` over the factory's batches.
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

    Examples:
        .. doctest::

            >>> import numpy as np
            >>> import batcher as bt
            >>> bt.from_numpy(np.array([1, 2, 3])).to_pydict()
            {'data': [1, 2, 3]}
    """
    return _scan(interop.from_numpy(ndarray, column=column))


def from_pandas(df: Any) -> Dataset:
    """Create a `Dataset` from a pandas `DataFrame` via its Arrow bridge.

    Needs pandas (``pip install 'batcher-engine[pandas]'``); raises `BackendError`
    if it is absent. Goes through ``pyarrow.Table.from_pandas`` — no per-row Python.

    Examples:
        .. doctest::

            >>> import pandas as pd
            >>> import batcher as bt
            >>> bt.from_pandas(pd.DataFrame({"a": [1, 2], "b": [3, 4]})).to_pydict()
            {'a': [1, 2], 'b': [3, 4]}

    Args:
        df: The pandas `DataFrame` to ingest.

    Returns:
        A lazy `Dataset` over the frame.

    Raises:
        BackendError: If pandas is not installed.
    """
    return _scan(interop.from_pandas(df))


def from_polars(df: Any) -> Dataset:
    """Create a `Dataset` from a Polars `DataFrame` via its zero-copy Arrow export.

    Polars is Arrow-backed, so the buffers are referenced directly, not copied.
    Needs polars (``pip install 'batcher-engine[polars]'``); raises `BackendError`
    if it is absent.

    Examples:
        .. doctest::

            >>> import polars as pl
            >>> import batcher as bt
            >>> bt.from_polars(pl.DataFrame({"a": [1, 2, 3]})).to_pydict()
            {'a': [1, 2, 3]}

    Args:
        df: The Polars `DataFrame` to ingest (referenced zero-copy).

    Returns:
        A lazy `Dataset` over the frame.

    Raises:
        BackendError: If polars is not installed.
    """
    return _scan(interop.from_polars(df))


def from_huggingface(hf_dataset: Any) -> Dataset:
    """Create a `Dataset` from a HuggingFace ``datasets.Dataset`` (Arrow-backed).

    HuggingFace datasets are Arrow tables underneath, so the table is taken
    directly. Needs ``datasets`` (``pip install 'batcher-engine[huggingface]'``);
    raises `BackendError` if it is absent.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from datasets import load_dataset  # doctest: +SKIP
            >>> hf = load_dataset("imdb", split="train")  # doctest: +SKIP
            >>> ds = bt.from_huggingface(hf)  # doctest: +SKIP

    Args:
        hf_dataset: The HuggingFace ``datasets.Dataset`` to ingest.

    Returns:
        A lazy `Dataset` over the underlying Arrow table.

    Raises:
        BackendError: If ``datasets`` is not installed.
    """
    return _scan(interop.from_huggingface(hf_dataset))


def from_torch(dataset_or_tensors: Any) -> Dataset:
    """Create a `Dataset` from a PyTorch tensor, tuple of tensors, or `Dataset`.

    Tensors are moved to CPU and adapted via their NumPy buffers (one column per
    tensor); only bulk buffers cross into the engine, never per-row Python. Needs
    ``torch`` (``pip install 'batcher-engine[torch]'``); raises `BackendError` if
    it is absent.

    Examples:
        .. doctest::

            >>> import torch  # doctest: +SKIP
            >>> import batcher as bt
            >>> bt.from_torch(torch.tensor([1, 2, 3])).to_pydict()  # doctest: +SKIP
            {'data': [1, 2, 3]}

    Args:
        dataset_or_tensors: A PyTorch tensor, tuple of tensors, or ``Dataset``.

    Returns:
        A lazy `Dataset`, one column per tensor.

    Raises:
        BackendError: If ``torch`` is not installed.
    """
    return _scan(interop.from_torch(dataset_or_tensors))


def from_tf(tf_dataset: Any) -> Dataset:
    """Create a `Dataset` from a ``tf.data.Dataset``, materializing it to Arrow.

    Each element's tensors are converted to NumPy and concatenated column-wise.
    Needs ``tensorflow`` (``pip install 'batcher-engine[tensorflow]'``); raises
    `BackendError` if it is absent.

    Examples:
        .. doctest::

            >>> import tensorflow as tf  # doctest: +SKIP
            >>> import batcher as bt  # doctest: +SKIP
            >>> tf_ds = tf.data.Dataset.from_tensor_slices({"x": [1, 2, 3]})  # doctest: +SKIP
            >>> ds = bt.from_tf(tf_ds)  # doctest: +SKIP

    Args:
        tf_dataset: The ``tf.data.Dataset`` to materialize.

    Returns:
        A lazy `Dataset` over the converted data.

    Raises:
        BackendError: If ``tensorflow`` is not installed.
    """
    return _scan(interop.from_tf(tf_dataset))


def from_spark(spark_df: Any) -> Dataset:
    """Create a `Dataset` from a Spark `DataFrame` by collecting it through Arrow.

    The Spark frame is collected to the driver via its Arrow bridge, so this
    materializes the data — for large frames write to a shared store and `read` it
    instead. Needs ``pyspark`` (``pip install 'batcher-engine[spark]'``); raises
    `BackendError` if it is absent.

    Examples:
        .. doctest::

            >>> import batcher as bt  # doctest: +SKIP
            >>> from pyspark.sql import SparkSession  # doctest: +SKIP
            >>> spark = SparkSession.builder.getOrCreate()  # doctest: +SKIP
            >>> sdf = spark.createDataFrame([(1,), (2,), (3,)], ["x"])  # doctest: +SKIP
            >>> ds = bt.from_spark(sdf)  # doctest: +SKIP

    Args:
        spark_df: The Spark `DataFrame` to collect through Arrow.

    Returns:
        A lazy `Dataset` over the collected data.

    Raises:
        BackendError: If ``pyspark`` is not installed.
    """
    return _scan(interop.from_spark(spark_df))


def from_dask(ddf: Any) -> Dataset:
    """Create a streaming `Dataset` from a Dask `DataFrame`, one partition per batch.

    Partitions stream lazily into the engine in bounded memory rather than being
    materialized at once. Needs ``dask`` (``pip install 'batcher-engine[dask]'``);
    raises `BackendError` if it is absent.

    Examples:
        .. doctest::

            >>> import dask.dataframe as dd  # doctest: +SKIP
            >>> import pandas as pd  # doctest: +SKIP
            >>> import batcher as bt  # doctest: +SKIP
            >>> pdf = pd.DataFrame({"x": [1, 2, 3]})  # doctest: +SKIP
            >>> ddf = dd.from_pandas(pdf, npartitions=2)  # doctest: +SKIP
            >>> ds = bt.from_dask(ddf)  # doctest: +SKIP

    Args:
        ddf: The Dask `DataFrame` to stream in, one partition per batch.

    Returns:
        A streaming lazy `Dataset` over the partitions.

    Raises:
        BackendError: If ``dask`` is not installed.
    """
    return _scan(interop.from_dask(ddf))


def from_ray_dataset(ray_dataset: Any) -> Dataset:
    """Create a streaming `Dataset` from a Ray Dataset (one Arrow block per batch).

    The migration on-ramp from Ray Data: blocks stream lazily into the engine in
    bounded memory. Requires `ray`.

    Examples:
        .. doctest::

            >>> import ray  # doctest: +SKIP
            >>> import batcher as bt  # doctest: +SKIP
            >>> rds = ray.data.range(100)  # doctest: +SKIP
            >>> ds = bt.from_ray_dataset(rds)  # doctest: +SKIP

    Args:
        ray_dataset: The Ray Dataset to stream in, one Arrow block per batch.

    Returns:
        A streaming lazy `Dataset` over the blocks.

    Raises:
        BackendError: If ``ray`` is not installed.
    """
    return _scan(interop.from_ray_dataset(ray_dataset))
