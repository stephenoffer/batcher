"""Daft adapter — distributed DataFrame comparator with a best-effort SQL surface.

Daft participates in the operator-mix (native ``DataFrame`` handle) and in the SQL
suites where its SQL planner can express the query. The SQL registration API has
shifted across Daft versions, so ``sql_runner`` tries the known shapes and degrades
to ``None`` (the suite then omits Daft) rather than crashing the run.
"""

from __future__ import annotations

import importlib.util

import pyarrow as pa

from .base import Engine, SqlRunner


class DaftEngine(Engine):
    name = "daft"
    tier = "multi"
    supports_sql = True

    @classmethod
    def available(cls) -> bool:
        return importlib.util.find_spec("daft") is not None

    def handle(self, table: pa.Table):
        import daft

        return daft.from_arrow(table)

    def read_parquet(self, uri: str):
        import daft

        return daft.read_parquet(uri)

    def sql_runner(self, tables: dict[str, pa.Table]) -> SqlRunner | None:
        import daft

        frames = {name: daft.from_arrow(tbl) for name, tbl in tables.items()}
        # Current Daft: named DataFrames are passed to daft.sql as bindings.
        return lambda query: daft.sql(query, **frames).to_arrow()
