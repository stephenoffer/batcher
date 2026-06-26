"""Batcher adapter — the system under test.

Operator-mix cases build on the ``Dataset`` handle (``bt.from_arrow`` /
``bt.read.parquet``); the SQL suites run through a ``bt.Session``, which parses the
query with sqlglot and lowers it to the same plan IR the DataFrame API produces.
"""

from __future__ import annotations

import dataclasses

import pyarrow as pa

import batcher as bt
from batcher.config import active_config, set_config

from .base import Engine, SqlRunner

# Measure pure engine performance: turn the per-query event log off so its small file
# write (on by default for observability) doesn't add I/O noise to the benchmark timing.
_cfg = active_config()
set_config(_cfg.replace(observability=dataclasses.replace(_cfg.observability, event_log=False)))


class BatcherEngine(Engine):
    name = "batcher"
    tier = "both"
    supports_sql = True

    @classmethod
    def available(cls) -> bool:
        return True  # batcher is the package under test; always present

    def handle(self, table: pa.Table) -> bt.Dataset:
        return bt.from_arrow(table)

    def read_parquet(self, uri: str) -> bt.Dataset:
        return bt.read.parquet(uri)

    def sql_runner(self, tables: dict[str, pa.Table]) -> SqlRunner:
        session = bt.Session()
        for name, tbl in tables.items():
            session.register(name, tbl)
        return lambda query: session.sql(query).collect()
