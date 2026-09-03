"""DuckDB adapter — the correctness oracle and the primary single-node comparator.

DuckDB reads parquet natively (``read_parquet`` over local/``s3://``/``https://``
paths) and runs every standard-suite query as SQL.
"""

from __future__ import annotations

import importlib.util
import os

import pyarrow as pa

from .base import Engine, Rename, SqlRunner, sql_projection


def match_batcher_budget(con: object) -> None:
    """Give DuckDB the same CPU and memory budget Batcher gives itself.

    Left alone the two engines size themselves independently and the difference favours
    Batcher. Measured on this box: DuckDB defaults `memory_limit` to **80% of RAM** (147.1
    GiB of 184), while Batcher auto-senses the whole machine and applies a `hard_limit`
    fraction of 0.9, giving it **165.6 GiB — 13% more headroom before it spills.**

    Thirteen percent decides nothing at sf1, where neither engine spills. It decides whether
    a query spills *at all* somewhere above 10M rows, which is exactly the regime the project
    concedes it loses in — so an undisclosed 13% sits right on the boundary of the honest
    claim, and on the flattering side of it.

    Threads are pinned for a different reason. The two engines agree at 92 on this box today,
    but they arrive there by separate auto-detections — Batcher's is cgroup-aware, DuckDB's
    is its own — and parity that holds by coincidence is parity nobody will notice losing.
    Pinning makes it a property of the harness rather than of the host.

    Both budgets come from Batcher's config, so the comparator is matched *to* the system
    under test rather than the reverse. That is the direction that removes an advantage.
    """
    from batcher._internal.hardware import available_cpu_count, machine_memory_bytes
    from batcher.config import active_config

    cfg = active_config()
    cap = cfg.memory.max_memory_bytes or machine_memory_bytes()
    effective = int(cap * (cfg.memory.hard_limit or 1.0))
    threads = cfg.execution.parallelism or available_cpu_count()
    con.execute(f"SET memory_limit='{effective}B'")
    con.execute(f"SET threads={max(1, int(threads))}")


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
        match_batcher_budget(con)
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

    def sql_runner_scan(self, uris: dict[str, str], rename: Rename | None = None) -> SqlRunner:
        import duckdb

        con = duckdb.connect()
        match_batcher_budget(con)
        con.sql("INSTALL httpfs; LOAD httpfs;")
        region = os.environ.get("BENCH_S3_REGION")
        if region:
            con.sql(f"SET s3_region='{region}'")
        for name, uri in uris.items():
            cols = sql_projection((rename or {}).get(name))
            con.sql(f"CREATE OR REPLACE VIEW {name} AS SELECT {cols} FROM read_parquet('{uri}')")
        return lambda query: con.sql(query).to_arrow_table()

    def scan_sql_runner(self, glob: str) -> SqlRunner:
        import duckdb

        con = duckdb.connect()
        match_batcher_budget(con)
        con.sql("INSTALL httpfs; LOAD httpfs;")

        def run(query: str) -> pa.Table:
            # A view over `read_parquet` re-binds — and so re-lists the glob — on every
            # execution, which is what puts scan planning inside the timed region.
            con.execute(f"CREATE OR REPLACE VIEW t AS SELECT * FROM read_parquet('{glob}')")
            return con.sql(query).to_arrow_table()

        return run
