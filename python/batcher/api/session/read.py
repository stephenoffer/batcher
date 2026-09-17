"""The generic read dispatch behind the `bt.read` namespace.

`read` sniffs the format from the URI scheme or file extension; `_read_table` constructs a
registered non-file source by name, and is what `bt.read.table` and every typed reader on
the namespace call. There are no top-level ``read_*`` shorthands: `bt.read.csv(...)` is the
one spelling, and `python -m batcher.migrate` rewrites `bt.read_csv(...)` onto it.
"""

from __future__ import annotations

from typing import Any

from batcher.api.dataset import Dataset
from batcher.api.session._scan import _scan
from batcher.io.detect import detect_format, partition_aware_format
from batcher.io.filesystem import require_success_marker
from batcher.io.formats.base import SOURCES

__all__ = [
    "read",
    "read_memory",
]


def _namespace() -> Any:
    """The `bt.read` accessor, imported lazily (it imports this module in turn)."""
    from batcher.api.io_namespace import read as reader

    return reader


def read(path: str, *, format: str | None = None, **opts: Any) -> Dataset:
    """Read a file/object-store dataset, dispatching on `format` or the path.

    With no `format`, it is inferred from the URI scheme (``delta://``…) or the
    file extension. ``read("s3://b/*.parquet")`` → Parquet; ``read("data/",
    format="csv")``. For database/catalog sources use `bt.read.table` or the typed
    readers on the `bt.read` namespace.

    ``require_success=True`` refuses a directory whose producing write never published a
    ``_SUCCESS`` marker. Every data file is written atomically, so none is ever half-written
    — but a run that died partway leaves a directory of *valid* files that reads back
    cleanly and silently short, and the marker is the only thing that distinguishes the two.
    Off by default, because a directory Batcher did not write has no marker and is not
    thereby incomplete; turn it on for a path another job produces.

    Args:
        path: A file, directory, glob, or URI to read.
        format: Force a format instead of inferring one from `path`.
        **opts: Format-specific reader options forwarded to the source, plus
            ``require_success`` (see above), which is consumed here.

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
    if opts.pop("require_success", False):
        require_success_marker(path)
    fmt = partition_aware_format(path, detect_format(path, format), opts)
    return _scan(SOURCES.get(fmt)(path, **opts))


def _read_table(format: str, *args: Any, **opts: Any) -> Dataset:
    """Read a registered non-file source by name (lakehouse/SQL/NoSQL/streaming).

    ``bt.read.table("delta", "s3://bucket/table", version=3)`` constructs the
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
            >>> ds = bt.read.table("delta", "s3://bucket/table", version=3)  # doctest: +SKIP
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
