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

The scan suite is the one exception (``build_corpus``): it benchmarks scan planning, so
it loads nothing and instead exposes the parquet ``corpora`` for each case to open
inside its own timed call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pyarrow as pa

from engines import Engine
from sources import (
    ImageCorpus,
    ScanCorpus,
    image_corpus,
    load_tables,
    scan_corpora,
    scan_rename,
    table_uris,
)

# A benchmark dataset name -> the public source benchmark it reads from. The
# operator-mix runs over the TPC-H tables (a real lineitem/orders join, real dates
# and strings) instead of any synthetic substrate.
SOURCE_FOR = {
    "tpch": "tpch",
    "tpcds": "tpcds",
    "clickbench": "clickbench",
    "operators": "tpch",
    "json": "json",
}

# Datasets built from a file *corpus* rather than named tables: the cases construct each
# engine's reader themselves (inside the timed call), so no table is loaded or bound here.
# ``scan`` is structured parquet in three layouts; ``images`` is unstructured JPEGs.
CORPUS_BENCHMARKS = frozenset({"scan", "images"})


@dataclass
class Context:
    """Loaded tables for one benchmark plus the per-engine handles cases ask for."""

    benchmark: str
    tables: dict[str, pa.Table]
    engines: list[Engine]
    uris: dict[str, str] = field(default_factory=dict)
    rename: dict[str, dict[str, str]] = field(default_factory=dict)
    corpora: dict[str, ScanCorpus] = field(default_factory=dict)
    images: ImageCorpus | None = None
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

    @classmethod
    def build_scan(
        cls,
        benchmark: str,
        scale: float,
        engines: list[Engine],
        source: str | None = None,
    ) -> Context:
        """Scan-mode context: no Arrow preload — each table is a lazy parquet glob.

        Used at large scale (sf100+), where materializing every table into shared
        Arrow would not fit in memory. Only the SQL fanout suites are supported here
        (operator-mix ``handle``/``table`` need in-memory Arrow and raise).
        """
        uris = table_uris(SOURCE_FOR[benchmark], scale, source)
        rename = scan_rename(SOURCE_FOR[benchmark], uris)
        return cls(benchmark=benchmark, tables={}, engines=engines, uris=uris, rename=rename)

    @classmethod
    def build_corpus(
        cls,
        benchmark: str,
        scale: float,
        engines: list[Engine],
        source: str | None = None,
    ) -> Context:
        """Corpus context: a file corpus at ``scale``, with nothing loaded or bound.

        The corpus suites measure the read/decode path itself, so each case builds its
        engine's reader inside the timed call — there is no table to preload and no runner
        to pre-register. ``scan`` fills ``corpora`` (three parquet layouts); ``images``
        fills ``images`` (one JPEG corpus).
        """
        if benchmark == "images":
            return cls(
                benchmark=benchmark,
                tables={},
                engines=engines,
                images=image_corpus(scale, source),
            )
        return cls(
            benchmark=benchmark,
            tables={},
            engines=engines,
            corpora=scan_corpora(scale, source),
        )

    def table(self, name: str) -> pa.Table:
        """The normalized Arrow table registered under ``name``."""
        return self.tables[name]

    def names(self) -> list[str]:
        """The active engine lineup, by name (report order)."""
        return [e.name for e in self.engines]

    def sql_runners(self) -> dict[str, Any]:
        """Engine name -> SQL executor, built once for every SQL-capable engine here.

        In scan mode (``uris`` set) each table binds to a lazy native parquet scan;
        otherwise the pre-loaded Arrow tables are registered.
        """
        if not self._runners:
            scan = bool(self.uris)
            for engine in self.engines:
                if not engine.supports_sql:
                    continue
                runner = (
                    engine.sql_runner_scan(self.uris, self.rename)
                    if scan
                    else engine.sql_runner(self.tables)
                )
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
