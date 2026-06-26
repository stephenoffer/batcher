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

# A pre-registered SQL executor: query string -> result table.
SqlRunner = Callable[[str], pa.Table]


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
