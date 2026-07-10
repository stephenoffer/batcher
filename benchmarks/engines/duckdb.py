"""DuckDB adapter — the correctness oracle and the primary single-node comparator.

DuckDB reads parquet natively (``read_parquet`` over local/``s3://``/``https://``
paths) and runs every standard-suite query as SQL.
"""

from __future__ import annotations

import importlib.util

import pyarrow as pa

from .base import Engine, SqlRunner


class DuckDBEngine(Engine):
    name = "duckdb"
    tier = "single"
    supports_sql = True

    @classmethod
    def available(cls) -> bool:
        return importlib.util.find_spec("duckdb") is not None

    def handle(self, table: pa.Table):
        import duckdb

        con = duckdb.connect()
        con.register("t", table)
        return con  # operator-mix cases query the registered "t"

    def read_parquet(self, uri: str):
        import duckdb

        return duckdb.connect(), uri

    def sql_runner(self, tables: dict[str, pa.Table]) -> SqlRunner:
        import duckdb

        con = duckdb.connect()
        # Ingest into DuckDB's NATIVE columnar storage — how every official
        # TPC-H/ClickBench result runs it. Merely *registering* the Arrow table forces
        # DuckDB's zero-copy Arrow scan, which is ~100x slower on joins (e.g. TPC-H Q12
        # at sf10: ~6.4s registered vs ~66ms native) — a large, unfair handicap that
        # made Batcher look far ahead of a DuckDB that was never running its real path.
        # Ingestion is one-time and untimed, exactly as Batcher's Arrow input is already
        # in memory and untimed; the timed query then runs on each engine's native form.
        for name, tbl in tables.items():
            con.register(f"__arrow_{name}", tbl)
            con.execute(f'CREATE TABLE "{name}" AS SELECT * FROM "__arrow_{name}"')
            con.unregister(f"__arrow_{name}")
        return lambda query: con.sql(query).to_arrow_table()

    def sql_runner_scan(self, uris: dict[str, str]) -> SqlRunner:
        import duckdb

        con = duckdb.connect()
        con.sql("INSTALL httpfs; LOAD httpfs;")
        for name, uri in uris.items():
            con.sql(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet('{uri}')")
        return lambda query: con.sql(query).to_arrow_table()
