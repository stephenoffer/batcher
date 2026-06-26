"""The per-benchmark data context: loads the public tables once, serves every engine.

A context loads a benchmark's tables from their established public parquet source
(``sources.load_tables``) a single time, then exposes exactly what the two suite
styles need:

- ``sql_runners()`` — engine name -> pre-registered ``query -> pa.Table`` callable,
  for the SQL-first standard suites (one query, fanned across SQL engines).
- ``handle(table, engine)`` — the engine's native object for a named table, cached,
  for the operator-mix cases (which build their query on it directly).

There is no data generation here and no per-engine duplication of the load: the same
normalized Arrow tables back every engine, which is what lets the correctness gate
compare them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pyarrow as pa

from engines import Engine
from sources import load_tables

# A benchmark dataset name -> the public source benchmark it reads from. The
# operator-mix runs over the TPC-H tables (a real lineitem/orders join, real dates
# and strings) instead of any synthetic substrate.
SOURCE_FOR = {
    "tpch": "tpch",
    "tpcds": "tpcds",
    "clickbench": "clickbench",
    "operators": "tpch",
}


@dataclass
class Context:
    """Loaded tables for one benchmark plus the per-engine handles cases ask for."""

    benchmark: str
    tables: dict[str, pa.Table]
    engines: list[Engine]
    _runners: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _handles: dict[tuple[str, str], Any] = field(default_factory=dict, init=False, repr=False)

    @classmethod
    def build(
        cls,
        benchmark: str,
        scale: float,
        engines: list[Engine],
        source: str | None = None,
    ) -> Context:
        tables = load_tables(SOURCE_FOR[benchmark], scale, source)
        return cls(benchmark=benchmark, tables=tables, engines=engines)

    def table(self, name: str) -> pa.Table:
        """The normalized Arrow table registered under ``name``."""
        return self.tables[name]

    def names(self) -> list[str]:
        """The active engine lineup, by name (report order)."""
        return [e.name for e in self.engines]

    def sql_runners(self) -> dict[str, Any]:
        """Engine name -> SQL executor, built once for every SQL-capable engine here."""
        if not self._runners:
            for engine in self.engines:
                if not engine.supports_sql:
                    continue
                runner = engine.sql_runner(self.tables)
                if runner is not None:
                    self._runners[engine.name] = runner
        return self._runners

    def handle(self, table: str, engine: str) -> Any:
        """The native handle for ``table`` in ``engine`` (cached across cases)."""
        key = (table, engine)
        if key not in self._handles:
            self._handles[key] = _engine(self.engines, engine).handle(self.tables[table])
        return self._handles[key]


def _engine(engines: list[Engine], name: str) -> Engine:
    for engine in engines:
        if engine.name == name:
            return engine
    raise KeyError(f"engine {name!r} not in the active lineup")
