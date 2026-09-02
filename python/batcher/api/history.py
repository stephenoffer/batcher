"""`query_history()` — the queries this deployment has run, as a `Dataset`.

Every executed query already leaves a structured document behind
(`api.terminal.event_log`): its measured wall time, rows, spill, memory and per-operator
profile, written to `$BATCHER_HOME/logs`. That artifact is complete and it was, until
this module, only reachable by opening JSON files by hand. Snowflake answers the same
question with `SNOWFLAKE.ACCOUNT_USAGE.QUERY_HISTORY` and Databricks with
`system.query.history`, and in both the answer is a *table* — because the questions an
operator actually has are relational ones: which queries spilled last night, what the p95
was per shape, which shape got slower after a release.

So this reads the documents that already exist and returns a `Dataset`. It adds no second
write path and no new artifact to keep in sync: turn the event log off and the history
stops growing, which is the honest coupling.

**What it deliberately does not expose.** The document carries the whole plan, literal
predicate constants included, so a `WHERE ssn = '123-45-6789'` is in it verbatim — which
is why `event_log` writes into an owner-only directory. The columns here are the
*measurements*, never the plan. `profile_path` names the document for a caller that has
decided it wants the rest; making that a deliberate second step is the difference between
an operator metric and a table that quietly republishes every literal any query ever
filtered on.

The module is named `history` rather than `query_history` on purpose: a module under
`batcher.api` whose name matches the function it exports shadows that function on the
package, so the lazy export table would route the name to the module instead of to the
callable (`tests/unit/test_lazy_exports.py` catches it).

**It records completed queries.** A query that raised closes out on the event bus and in
the trace, and writes no document, so it is not here. An operator watching for failures
wants the bus (`bt.subscribe`) or the trace backend, not this.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any

import pyarrow as pa

from batcher._internal.errors import PlanError

if TYPE_CHECKING:
    from batcher.api.dataset import Dataset

__all__ = ["query_history"]

#: One output column per entry: the column's name, its Arrow type, and the document key
#: it is read from.
#:
#: Names follow the warehouses' own vocabulary where the measurement is the same one —
#: `total_elapsed_ms` and `rows_produced` are Snowflake's `TOTAL_ELAPSED_TIME` and
#: `ROWS_PRODUCED` — and Batcher's own where it is not, because inventing a Snowflake-ish
#: name for something Snowflake does not measure claims a parity that is not there.
#:
#: The types are declared rather than inferred, and that is what makes an *empty* history
#: usable: inference over no rows types every column `null`, so the dashboard query
#: `filter(col("total_elapsed_ms") > 1000)` failed on a fresh deployment — the one place
#: it most needed to return no rows. Declaring them means the query plans identically
#: whether or not anything has run yet.
#:
#: A missing key yields None rather than dropping the row: a document written by an older
#: build is still a query that ran, and losing it would silently shorten the history at
#: exactly the moment (an upgrade) an operator is comparing across one.
_COLUMNS: tuple[tuple[str, pa.DataType, str], ...] = (
    ("query_id", pa.string(), "query_id"),
    ("total_elapsed_ms", pa.float64(), "total_ms"),
    ("rows_produced", pa.int64(), "rows"),
    ("distributed", pa.bool_(), "distributed"),
    ("measured", pa.bool_(), "measured"),
    ("spilled", pa.bool_(), "spilled"),
    ("bytes_spilled", pa.int64(), "total_spill_bytes"),
    ("peak_memory_bytes", pa.int64(), "peak_rss_bytes"),
    ("memory_budget_bytes", pa.int64(), "memory_budget_bytes"),
    ("cpu_utilization", pa.float64(), "cpu_utilization"),
    ("admission", pa.string(), "carbonite_summary"),
    ("machine", pa.string(), "machine"),
)

#: Columns read out of the document's nested `usage` block rather than its top level.
_USAGE_COLUMNS: tuple[tuple[str, pa.DataType, str], ...] = (
    ("cpu_ms", pa.float64(), "cpu_ms"),
    ("wall_ms", pa.float64(), "wall_ms"),
    ("cores_busy", pa.float64(), "cores_busy"),
)

#: Columns this module derives rather than reads.
_DERIVED: tuple[tuple[str, pa.DataType], ...] = (
    ("operator_count", pa.int64()),
    ("profile_path", pa.string()),
)


def _schema() -> pa.Schema:
    """The history's schema, identical whether or not any query has been recorded."""
    fields = [(name, kind) for name, kind, _ in _COLUMNS]
    fields += [(name, kind) for name, kind, _ in _USAGE_COLUMNS]
    fields += list(_DERIVED)
    return pa.schema(fields)


def query_history(path: str | None = None, *, limit: int | None = None) -> Dataset:
    """Return the completed queries this deployment has recorded, most recent first.

    One row per query, with the measurements the engine took. `query_id` sorts
    chronologically, so ``sort("query_id")`` orders the history without parsing a
    timestamp.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> _ = bt.from_pydict({"a": [1, 2, 3]}).agg(s=bt.col("a").sum()).collect()
            >>> history = bt.query_history()
            >>> "total_elapsed_ms" in history.columns
            True

            The point of it being a `Dataset` is that the operator's real questions are
            relational:

            >>> slow = history.filter(bt.col("total_elapsed_ms") > 1000.0)
            >>> slow.count() >= 0
            True

    Args:
        path: The directory to read documents from. Defaults to the one
            `observability.event_log_dir` names, which is where the engine writes them.
        limit: Read only the most recent this many documents. Documents sort
            chronologically by name, so this is the cheap way to ask about recent
            activity without parsing the whole retention window.

    Returns:
        A `Dataset` with one row per recorded query. Empty, with the full schema, when
        nothing has been recorded — so a dashboard query against a fresh deployment
        returns no rows rather than failing on a missing column.

    Raises:
        PlanError: If `limit` is not a positive integer, or `path` names something that
            is not a directory. A typo'd path returning an empty history would read as
            "nothing ran", which is the wrong answer to act on.
    """
    from batcher.api.session import from_arrow

    if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit < 1):
        raise PlanError(
            f"query_history(limit=...) must be a positive integer, got {limit!r}.",
            hint="Omit it to read the whole retention window.",
        )
    columns = _read(_directory(path), limit)
    schema = _schema()
    return from_arrow(pa.table([columns[f.name] for f in schema], schema=schema))


def _default_directory() -> str:
    """Where the event log writes, resolved **without creating it**.

    `event_log._resolve_dir` is the writer's resolution and it `mkdir`s, owner-only, as
    part of resolving — correct for a writer and wrong here twice over. A reader that
    creates a directory as a side effect of being called turns "has anything run?" into a
    filesystem mutation, and against a configured `event_log_dir` the process cannot
    create (`/data/...` on a machine where it is not root) it turns the question into a
    `PermissionError` rather than the honest answer, which is "nothing".

    The path convention is therefore stated in two places, and
    `tests/unit/test_query_history.py` holds them equal so they cannot drift.

    Returns:
        The directory the event log writes into, whether or not it exists.
    """
    from batcher.config import active_config

    configured = active_config().observability.event_log_dir
    if configured:
        return configured
    home = os.environ.get("BATCHER_HOME")
    return os.path.join(home or os.path.join(os.path.expanduser("~"), ".batcher"), "logs")


def _directory(path: str | None) -> str:
    """The directory to read, defaulting to the event log's own and validating a given one."""
    if path is None:
        return _default_directory()
    if not isinstance(path, str) or not path:
        raise PlanError(
            f"query_history(path=...) must be a non-empty string, got {path!r}.",
            hint="Omit it to read where the engine writes its event log.",
        )
    if os.path.exists(path) and not os.path.isdir(path):
        raise PlanError(
            f"query_history(path={path!r}) is not a directory.",
            hint="It reads a directory of event-log documents, not one document.",
        )
    return path


def _read(directory: str, limit: int | None) -> dict[str, list[Any]]:
    """Read the documents in `directory` into one column-per-field dict.

    Column-oriented from the start rather than a list of row dicts that something else
    transposes: the output is Arrow, so building rows first would allocate a dict per
    query only to take it apart again.
    """
    columns: dict[str, list[Any]] = {name: [] for name in _schema().names}
    for name in _document_names(directory, limit):
        full = os.path.join(directory, name)
        document = _load(full)
        if document is None:
            continue
        for column, _, key in _COLUMNS:
            columns[column].append(document.get(key))
        usage = document.get("usage")
        usage = usage if isinstance(usage, dict) else {}
        for column, _, key in _USAGE_COLUMNS:
            columns[column].append(usage.get(key))
        ops = document.get("ops")
        columns["operator_count"].append(len(ops) if isinstance(ops, list) else None)
        columns["profile_path"].append(full)
    return columns


def _document_names(directory: str, limit: int | None) -> list[str]:
    """The document names to read, most recent first.

    Sorted by name, which is chronological: `event_log._query_id` mints
    ``YYYYmmdd-HHMMSS-<pid>-<seq>`` precisely so the ordering is free here.
    """
    try:
        names = sorted(
            (e.name for e in os.scandir(directory) if e.name.endswith(".json") and e.is_file()),
            reverse=True,
        )
    except (FileNotFoundError, NotADirectoryError, PermissionError):
        # A deployment that has never written a document, or one whose log directory this
        # process may not read. Both are "no history", not a failure of the query the
        # caller is trying to run against it.
        return []
    return names if limit is None else names[:limit]


def _load(full: str) -> dict[str, Any] | None:
    """One document, or None if it cannot be read as one.

    Skipped rather than raised, and this is the one place that judgement is right: the
    engine prunes this directory while it writes it, so a document can be deleted between
    the scan and the open, and a half-written one is what a crash mid-write leaves. One
    unreadable file must not fail a history query over two hundred good ones.
    """
    try:
        with open(full, encoding="utf-8") as handle:
            document = json.load(handle)
    except (OSError, ValueError):
        return None
    return document if isinstance(document, dict) else None
