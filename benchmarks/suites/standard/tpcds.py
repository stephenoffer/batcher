"""TPC-DS — the full 99-query decision-support benchmark.

Every query is registered, not a curated subset: TPC-DS is the broadest standard
workload there is (roll-ups and grouping sets, correlated and scalar subqueries,
window functions, set operations, snowflake joins across all three sales channels),
so a partial suite hides exactly the shapes that are hardest to get right.

The statements are **not written here**. They are vendored verbatim from DuckDB's
``tpcds`` extension into :data:`QUERY_FILE` by ``tools/vendor_tpcds_queries.py`` — the
same extension whose ``dsdgen`` materializes the tables (``sources.tables``), so the
queries and the data come from one source. This module only splits that file on its
``-- @query <name>`` delimiters and fans each statement across the SQL-capable engines.

A query an engine cannot yet run reports as an error for that engine and the case as
``PARTIAL``; the others are still compared and timed. That is the point of registering
all 99 — the gaps are visible per query rather than absent from the suite.
"""

from __future__ import annotations

import os

from registry import suite
from suites.standard._vendored import register_vendored

tpcds = suite("tpcds", dataset="tpcds")

QUERY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tpcds_queries.sql")

# TPC-DS is 99 queries by definition; anything else means the vendored file is wrong.
QUERY_COUNT = 99

# Engines a query kills rather than slows. Each entry is a measurement: re-check it when the
# engine's version changes, and delete it when the engine survives.
_REFUSE = {
    "tpcds-q64": {
        # Measured 2026-09-28, daft 0.7.25, sf1, alone in its own process: SIGKILLed by the
        # cgroup OOM killer at 44 GB RSS (Batcher peaks at 4.0 GB, DuckDB at 5.6 GB). Without
        # this refusal the kill takes every other engine's q64 result with it under
        # `--isolate`, and the rest of the suite without it.
        "daft": "OOM: daft is SIGKILLed on q64 at sf1 (44 GB RSS, daft 0.7.25)",
    },
}

QUERIES = register_vendored(
    tpcds,
    QUERY_FILE,
    count=QUERY_COUNT,
    vendor_tool="tools/vendor_tpcds_queries.py",
    refuse=_REFUSE,
)
