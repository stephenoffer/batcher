"""The engine-adapter contract every comparator implements.

A benchmark compares the same query across several engines. Each engine differs in
how it loads data, whether it speaks SQL, and how it returns a result, so the rest
of the harness talks to engines only through this small interface:

- ``handle(table)`` / ``read_parquet(uri)`` produce the engine's *native* object (a
  Batcher ``Dataset``, a Polars frame, a DuckDB relation, ...). Operator-mix cases
  build their query directly on that handle and return a ``pyarrow.Table``.
- ``sql_runner(tables)`` pre-registers a set of named tables once and returns a
  ``query -> pyarrow.Table`` callable, or ``None`` when the engine has no SQL
  surface. The SQL-first standard suites (TPC-H / TPC-DS / ClickBench) fan a single
  query string out across every engine whose ``sql_runner`` is not ``None``.

Capability flags (``tier``, ``supports_sql``) let the harness skip an engine on a
workload it cannot express instead of failing it. ``available()`` reports whether
the engine's package is importable in this environment.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pyarrow as pa
import pyarrow.fs as pafs

# A pre-registered SQL executor: query string -> result table.
SqlRunner = Callable[[str], pa.Table]

# Per-table ``{stored column name -> canonical name}`` for the scan path (see
# ``sources.scan_rename``): the public TPC-H parquet is positionally named.
Rename = dict[str, dict[str, str]]


def sql_projection(cols: dict[str, str] | None) -> str:
    """The SELECT list that renames a scanned table's columns — ``*`` when there is none.

    Shared by the SQL engines whose scan binding is a view (DuckDB, Spark): a rename is
    expressed as ``stored AS canonical``, which every SQL planner folds into the parquet
    read rather than materializing.
    """
    if not cols:
        return "*"
    return ", ".join(f'"{stored}" AS "{canonical}"' for stored, canonical in cols.items())


class Engine:
    """Base adapter. Subclasses set the class attributes and override what they support.

    Concrete adapters live one-per-file in this package and register themselves in
    ``engines/__init__.py``. The default method bodies raise / return ``None`` so an
    engine only implements the surfaces it actually has.
    """

    name: str = ""
    # "single" (single-node only), "multi" (distributed), or "both".
    tier: str = "single"
    supports_sql: bool = False

    @classmethod
    def available(cls) -> bool:
        """Whether this engine's package can be imported here (override per engine)."""
        return False

    def prepare(self) -> None:
        """One-time setup this engine needs **before any engine runs**. Default: nothing.

        It exists for one real ordering problem. Ray Data's workers need the `benchmarks/`
        directory on their `PYTHONPATH` — the TPC-H pipelines are module-level functions, so
        cloudpickle sends them by reference and a worker that cannot import `suites` dies
        before running a batch. A job-level `runtime_env` can only be set by whoever calls
        `ray.init`, and the adapter only did so `if not ray.is_initialized()`. In the
        `--tier multi` lineup Batcher leads and initializes Ray first, so that branch never
        ran and every Ray Data query failed with `ModuleNotFoundError: No module named
        'suites'` — the comparison the tier exists for, reported as `ERR` on all 22 queries.

        Calling this for every selected engine before the first case lets the engine that
        needs to own `ray.init` take it.
        """

    def release(self) -> None:
        """Give up cluster-wide resources this engine is holding. Default: nothing.

        A multi-engine run times each engine in turn on one cluster, so an engine that keeps
        the cluster reserved between its own calls starves whichever engine is timed next.
        Batcher is the one that does: its session fleet is a placement group reserving ~99%
        of the cores and is deliberately kept warm across `collect()` calls. Daft then could
        not start at all — every query failed with `No flotilla workers became available
        within 120s (4 attempted)` — so the comparison the `--tier multi` lineup exists for
        produced no competitor column.

        Called after an engine's runs for a case, and only when it shares the lineup, so a
        single-engine run keeps the warm fleet it would have in production.
        """

    def handle(self, table: pa.Table) -> Any:
        """Native handle wrapping an in-memory Arrow table (for operator-mix cases)."""
        raise NotImplementedError(f"{self.name} has no in-memory handle")

    def read_parquet(self, uri: str) -> Any:
        """Native handle reading parquet from ``uri`` (local path, ``s3://``, ``https://``)."""
        raise NotImplementedError(f"{self.name} cannot read parquet")

    def sql_runner(self, _tables: dict[str, pa.Table]) -> SqlRunner | None:
        """A ``query -> pa.Table`` callable with the tables pre-registered, or ``None``.

        The base returns ``None`` — the engine has no SQL surface, so the SQL suites
        omit it (it shows as ``n/a``, never a failure). SQL engines override this.
        """
        return None

    def sql_runner_scan(
        self, _uris: dict[str, str], _rename: Rename | None = None
    ) -> SqlRunner | None:
        """A ``query -> pa.Table`` callable with each named table bound to a *lazy
        native parquet scan* of ``uris[name]`` (a glob), or ``None``.

        This is the large-scale counterpart to :meth:`sql_runner`: instead of
        pre-materializing every table into shared Arrow (~100GB at sf100), each engine
        reads parquet natively and lazily through its own scan — the representative way
        these engines run at scale, and the only way that fits in memory. SQL engines
        override it; the base returns ``None`` (the suite omits the engine).

        ``_rename`` (from :func:`sources.scan_rename`) maps each table's stored column
        names to the canonical ones the queries use, applied inside the scan binding.
        The public TPC-H parquet is named positionally, so the scan path needs it; it is
        the identical pure-metadata projection for every engine.
        """
        return None

    # ----------------------------------------------------------------------- #
    # Scan benchmark: bind the scan *inside* the timed call
    # ----------------------------------------------------------------------- #
    # `sql_runner_scan` binds its tables once, up front, so file listing and metadata
    # opening land outside the timed region. That is right for TPC-H (where the scan is
    # a fixed setup cost shared by 22 queries) and wrong for the file-layout benchmark,
    # whose entire subject *is* that setup cost. These two hooks therefore rebuild the
    # scan on every invocation, so the measurement covers list -> open -> read -> compute.

    def scan_sql_runner(self, _glob: str) -> SqlRunner | None:
        """A ``query -> pa.Table`` callable binding table ``t`` to a *fresh* scan of ``_glob``.

        Re-planned on every call, unlike :meth:`sql_runner_scan`. Returns ``None`` when
        the engine has no SQL surface (the case then shows ``n/a`` for it).
        """
        return None

    def scan_handle(self, _filesystem: pafs.FileSystem, _paths: list[str]) -> Any:
        """A native lazy handle over an explicit parquet file list.

        The non-SQL engines (PyArrow, Ray Data) take a file list rather than a glob, so
        the scan suite lists the corpus itself and hands the paths over. Called inside
        the timed region.
        """
        raise NotImplementedError(f"{self.name} cannot scan a parquet file list")
