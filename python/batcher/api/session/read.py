"""The generic read dispatch plus the top-level ``read_*`` shorthands.

`read` sniffs the format from the URI scheme or file extension; `read_table`
constructs a registered non-file source by name. The `read_csv` / `read_parquet` /
… functions are the pandas and Polars spellings of the same thing, so a ported
script keeps reading the way it was written; the fuller per-format option surface
lives on the `bt.read` namespace (``bt.read.csv``), which these delegate to.
"""

from __future__ import annotations

from typing import Any

from batcher.api.dataset import Dataset
from batcher.api.session._scan import _scan
from batcher.io.detect import detect_format
from batcher.io.formats.base import SOURCES

__all__ = [
    "read",
    "read_avro",
    "read_csv",
    "read_database",
    "read_delta",
    "read_excel",
    "read_iceberg",
    "read_ipc",
    "read_json",
    "read_memory",
    "read_ndjson",
    "read_orc",
    "read_parquet",
    "read_table",
]


def _namespace() -> Any:
    """The `bt.read` accessor, imported lazily (it imports this module in turn)."""
    from batcher.api.io_namespace import read as reader

    return reader


def read(path: str, *, format: str | None = None, **opts: Any) -> Dataset:
    """Read a file/object-store dataset, dispatching on `format` or the path.

    With no `format`, it is inferred from the URI scheme (``delta://``…) or the
    file extension. ``read("s3://b/*.parquet")`` → Parquet; ``read("data/",
    format="csv")``. For database/catalog sources use `read_table` or the typed
    ``read_*`` helpers.

    Args:
        path: A file, directory, glob, or URI to read.
        format: Force a format instead of inferring one from `path`.
        **opts: Format-specific reader options forwarded to the source.

    Returns:
        A lazy `Dataset` over the source.

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

    Args:
        format: The registered source name, e.g. ``"delta"`` or ``"kafka"``.
        *args: Positional arguments forwarded to that source's constructor.
        **opts: Keyword options forwarded to that source's constructor.

    Returns:
        A lazy `Dataset` over the source.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.read_table("delta", "s3://bucket/table", version=3)  # doctest: +SKIP
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
    from batcher.api.session.frames import from_arrow
    from batcher.io.formats.streaming.sinks import memory_table

    try:
        table = memory_table(name)
    except KeyError:
        raise PlanError(f"no in-memory streaming sink named {name!r}") from None
    return from_arrow(table)


def read_csv(path: str, **opts: Any) -> Dataset:
    r"""Read a CSV file, directory, or glob (pandas/Polars ``read_csv``).

    Shorthand for ``bt.read.csv(path, ...)``. The header row and column types are
    inferred from the first block; pass ``schema=`` to declare them instead.

    Args:
        path: A CSV file, directory, or glob to read.
        **opts: Reader options forwarded to the CSV source, notably ``schema``
            and ``on_error``.

    Returns:
        A lazy `Dataset` over the CSV source.

    Examples:
        .. doctest::

            >>> import batcher as bt, tempfile, os
            >>> p = os.path.join(tempfile.mkdtemp(), "t.csv")
            >>> _ = open(p, "w").write("a,b\n1,2\n")
            >>> bt.read_csv(p).to_pydict()
            {'a': [1], 'b': [2]}
    """
    return read(path, format="csv", **opts)


def read_parquet(path: str, **opts: Any) -> Dataset:
    """Read a Parquet file, directory, or glob (pandas/Polars ``read_parquet``).

    Shorthand for ``bt.read.parquet(path, ...)``. Projection and predicates are
    pushed into the read, and Hive-style partition directories are pruned.

    Args:
        path: A Parquet file, directory, or glob to read.
        **opts: Reader options forwarded to the Parquet source.

    Returns:
        A lazy `Dataset` over the Parquet source.

    Examples:
        .. doctest::

            >>> import batcher as bt, tempfile, os
            >>> p = os.path.join(tempfile.mkdtemp(), "t.parquet")
            >>> _ = bt.from_pydict({"x": [1, 2]}).write.parquet(p)
            >>> bt.read_parquet(p).sort("x").to_pydict()
            {'x': [1, 2]}
    """
    return read(path, format="parquet", **opts)


def read_json(path: str, **opts: Any) -> Dataset:
    r"""Read newline-delimited JSON (pandas ``read_json`` with ``lines=True``).

    Shorthand for ``bt.read.json(path, ...)``. `read_ndjson` is the Polars name for
    the same reader.

    Args:
        path: A JSON file, directory, or glob to read.
        **opts: Reader options forwarded to the JSON source.

    Returns:
        A lazy `Dataset` over the JSON source.

    Examples:
        .. doctest::

            >>> import batcher as bt, tempfile, os
            >>> p = os.path.join(tempfile.mkdtemp(), "t.json")
            >>> _ = open(p, "w").write('{"a": 1}\n{"a": 2}\n')
            >>> bt.read_json(p).to_pydict()
            {'a': [1, 2]}
    """
    return read(path, format="json", **opts)


def read_ndjson(path: str, **opts: Any) -> Dataset:
    r"""Read newline-delimited JSON (the Polars ``read_ndjson`` spelling).

    An alias of `read_json`, which already reads one JSON object per line.

    Args:
        path: An NDJSON file, directory, or glob to read.
        **opts: Reader options forwarded to the JSON source.

    Returns:
        A lazy `Dataset` over the NDJSON source.

    Examples:
        .. doctest::

            >>> import batcher as bt, tempfile, os
            >>> p = os.path.join(tempfile.mkdtemp(), "t.ndjson")
            >>> _ = open(p, "w").write('{"a": 1}\n')
            >>> bt.read_ndjson(p).to_pydict()
            {'a': [1]}
    """
    return read(path, format="json", **opts)


def read_ipc(path: str, **opts: Any) -> Dataset:
    """Read an Arrow IPC / Feather file (Polars ``read_ipc``).

    Shorthand for ``bt.read.arrow(path, ...)``.

    Args:
        path: An Arrow IPC file, directory, or glob to read.
        **opts: Reader options forwarded to the Arrow IPC source.

    Returns:
        A lazy `Dataset` over the Arrow IPC source.

    Examples:
        .. doctest::

            >>> import batcher as bt, tempfile, os
            >>> p = os.path.join(tempfile.mkdtemp(), "t.arrow")
            >>> _ = bt.from_pydict({"x": [1]}).write(p, format="arrow")
            >>> bt.read_ipc(p).to_pydict()
            {'x': [1]}
    """
    return read(path, format="arrow", **opts)


def read_orc(path: str, **opts: Any) -> Dataset:
    """Read an ORC file, directory, or glob (pandas ``read_orc``).

    Shorthand for ``bt.read.orc(path, ...)``.

    Args:
        path: An ORC file, directory, or glob to read.
        **opts: Reader options forwarded to the ORC source.

    Returns:
        A lazy `Dataset` over the ORC source.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.read_orc("s3://bucket/events/")  # doctest: +SKIP
    """
    return read(path, format="orc", **opts)


def read_avro(path: str, **opts: Any) -> Dataset:
    """Read an Avro file, directory, or glob.

    Shorthand for ``bt.read.avro(path, ...)``.

    Args:
        path: An Avro file, directory, or glob to read.
        **opts: Reader options forwarded to the Avro source.

    Returns:
        A lazy `Dataset` over the Avro source.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.read_avro("s3://bucket/events/")  # doctest: +SKIP
    """
    return read(path, format="avro", **opts)


def read_excel(path: str, **opts: Any) -> Dataset:
    """Read an Excel workbook (pandas/Polars ``read_excel``).

    Shorthand for ``bt.read.excel(path, ...)``.

    Args:
        path: The workbook to read.
        **opts: Reader options forwarded to the Excel source, e.g. ``sheet``.

    Returns:
        A lazy `Dataset` over the sheet.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.read_excel("report.xlsx", sheet="Q1")  # doctest: +SKIP
    """
    return read(path, format="excel", **opts)


def read_delta(path: str, **opts: Any) -> Dataset:
    """Read a Delta Lake table (Polars ``read_delta``).

    Shorthand for ``bt.read.delta(path, ...)``. Pass ``version=`` or
    ``timestamp=`` to time travel.

    Args:
        path: The Delta table root.
        **opts: Reader options forwarded to the Delta source, e.g. ``version``.

    Returns:
        A lazy `Dataset` over the table snapshot.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.read_delta("s3://lake/events", version=3)  # doctest: +SKIP
    """
    return _namespace().delta(path, **opts)


def read_iceberg(table: str, **opts: Any) -> Dataset:
    """Read an Apache Iceberg table (Polars ``scan_iceberg``).

    Shorthand for ``bt.read.iceberg(table, ...)``.

    Args:
        table: The Iceberg table identifier or location.
        **opts: Reader options forwarded to the Iceberg source, e.g. ``catalog``.

    Returns:
        A lazy `Dataset` over the table snapshot.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.read_iceberg("db.events", catalog="glue")  # doctest: +SKIP
    """
    return _namespace().iceberg(table, **opts)


def read_database(query: str, uri: str | None = None, **opts: Any) -> Dataset:
    """Read a SQL database from a connection URI (Polars ``read_database_uri``).

    Shorthand for ``bt.read.sql(query, uri=...)``. The URI vocabulary is
    SQLAlchemy's — the same string ``pandas.read_sql`` or ``$DATABASE_URL`` uses —
    and the projection and filter are pushed into the submitted SQL.

    Args:
        query: The SQL statement to run, or a table name.
        uri: The SQLAlchemy-style connection URI.
        **opts: Options forwarded to the SQL source, e.g. ``connection`` or
            ``partition_on``.

    Returns:
        A lazy `Dataset` over the query result.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.read_database(  # doctest: +SKIP
            ...     "SELECT * FROM orders", uri="postgresql://host/db"
            ... )
    """
    return _namespace().sql(query, uri=uri, **opts)
