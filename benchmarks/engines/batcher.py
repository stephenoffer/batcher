"""Batcher adapter — the system under test.

Operator-mix cases build on the ``Dataset`` handle (``bt.from_arrow`` /
``bt.read.parquet``); the SQL suites run through a ``bt.Session``, which parses the
query with sqlglot and lowers it to the same plan IR the DataFrame API produces.
"""

from __future__ import annotations

import dataclasses
import os

import pyarrow as pa

import batcher as bt
from batcher.config import active_config, set_config

from .base import Engine, Rename, SqlRunner

# Measure pure engine performance: turn the per-query event log off so its small file
# write (on by default for observability) doesn't add I/O noise to the benchmark timing.
_cfg = active_config()
set_config(_cfg.replace(observability=dataclasses.replace(_cfg.observability, event_log=False)))

# Execution mode. Default single-node: Batcher's in-process path is its low-overhead
# strength, and for an in-memory operator-mix it is the honest counterpart to Ray Data
# running distributed on the cluster.
#
# ``BENCH_BATCHER_DISTRIBUTED=1`` forces the distributed path — the multi-node tier, where
# the comparators are Daft-on-Ray and Ray Data. It must be an explicit ``True``, not
# ``"auto"``: ``resolve_distributed`` requires an ALREADY-initialized Ray, and nothing in
# the benchmark process initializes one before the first collect, so ``"auto"`` silently
# resolved to single-node and the flag measured nothing.
#
# ``BENCH_BATCHER_PARTITIONS`` sets the shuffle fan-out. Unset, ``collect`` defaults it to
# the *driver's* core count, which on a head node smaller than the workers under-fans the
# cluster; the benchmark pins it to the cluster's total CPUs instead.
_DISTRIBUTED: bool = os.environ.get("BENCH_BATCHER_DISTRIBUTED") == "1"
_PARTITIONS: int | None = (
    int(os.environ["BENCH_BATCHER_PARTITIONS"])
    if os.environ.get("BENCH_BATCHER_PARTITIONS")
    else None
)


def _collect(dataset):
    """Collect under the benchmark's execution mode (single-node, or distributed fan-out)."""
    if not _DISTRIBUTED:
        return dataset.collect()
    return dataset.collect(distributed=True, num_partitions=_PARTITIONS)


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
        return lambda query: _collect(session.sql(query))

    def sql_runner_scan(self, uris: dict[str, str], rename: Rename | None = None) -> SqlRunner:
        session = bt.Session()
        for name, uri in uris.items():
            scan = bt.read.parquet(uri)
            cols = (rename or {}).get(name)
            if cols:
                scan = scan.rename(cols)
            session.register(name, scan)
        return lambda query: _collect(session.sql(query))

    def scan_sql_runner(self, glob: str) -> SqlRunner:
        def run(query: str) -> pa.Table:
            session = bt.Session()
            session.register("t", bt.read.parquet(glob))
            return _collect(session.sql(query))

        return run
