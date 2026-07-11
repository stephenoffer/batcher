"""DuckDB-on-Arrow adapter — the same-input execution comparison.

The default :class:`DuckDBEngine` ingests every table into DuckDB's *native* columnar
storage (an untimed ``CREATE TABLE``) before the timed query. That measures DuckDB's
storage engine — compression, dictionary encoding, min/max zone maps — *plus* its
execution engine, against Batcher's execution engine over raw Arrow. It is the right
bar for "DuckDB at its best," but it conflates two layers: on the same in-memory Arrow
that Batcher runs on, DuckDB is markedly slower (measured 1.3-2.6x on TPC-H sf1), so the
native-vs-Arrow gap is DuckDB's storage advantage, not an execution deficit in Batcher.

This adapter is the honest execution-parity comparator: it binds each table as a
*zero-copy Arrow view* (``con.register``) — exactly the input Batcher receives — so the
timed query exercises DuckDB's execution engine over identical bytes. The original
adapter's rationale for avoiding this (that an Arrow scan is "~100x slower on joins") was
true of older DuckDB; on 1.5.x a registered-Arrow join is ~1.5-3x native, not 100x, so
the fair comparison is now viable and is what this measures.

Batcher's ``Arrow is the only columnar contract`` invariant means it has no native
compressed store to switch to — so ``duckdb_arrow`` is the like-for-like bar, and
``duckdb`` (native) is the storage-advantaged one. Report both to keep the claim honest.
"""

from __future__ import annotations

import pyarrow as pa

from .base import SqlRunner
from .duckdb import DuckDBEngine


class DuckDBArrowEngine(DuckDBEngine):
    name = "duckdb_arrow"

    def sql_runner(self, tables: dict[str, pa.Table]) -> SqlRunner:
        import duckdb

        con = duckdb.connect()
        # Zero-copy Arrow views — the same in-memory bytes Batcher executes over, with no
        # untimed ingest/compression step. This is the execution-engine comparison.
        for name, tbl in tables.items():
            con.register(name, tbl)
        return lambda query: con.sql(query).to_arrow_table()
