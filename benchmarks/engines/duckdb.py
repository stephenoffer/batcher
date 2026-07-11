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
        # TPC-H/ClickBench result runs it, and the "DuckDB at its best" bar. Ingestion is
        # one-time and untimed; the timed query then runs on DuckDB's compressed, dictionary-
        # encoded, zone-mapped native form.
        #
        # This measures DuckDB's *storage engine plus* its execution engine against Batcher's
        # execution engine over raw Arrow — not a like-for-like execution comparison. On the
        # SAME in-memory Arrow that Batcher runs on, DuckDB is 1.3-2.6x slower (see the
        # ``duckdb_arrow`` adapter), so the native-vs-Arrow gap is DuckDB's storage advantage,
        # which Batcher's `Arrow is the only columnar contract` invariant precludes matching.
        # (An earlier note here claimed registered-Arrow was "~100x slower on joins"; that was
        # true of older DuckDB — on 1.5.x it is ~1.5-3x, so ``duckdb_arrow`` is now the viable
        # like-for-like bar. Report both to keep the claim honest.)
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

    def scan_sql_runner(self, glob: str) -> SqlRunner:
        import duckdb

        con = duckdb.connect()
        con.sql("INSTALL httpfs; LOAD httpfs;")

        def run(query: str) -> pa.Table:
            # A view over `read_parquet` re-binds — and so re-lists the glob — on every
            # execution, which is what puts scan planning inside the timed region.
            con.execute(f"CREATE OR REPLACE VIEW t AS SELECT * FROM read_parquet('{glob}')")
            return con.sql(query).to_arrow_table()

        return run
