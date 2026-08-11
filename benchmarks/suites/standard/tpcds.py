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

tpcds = suite("tpcds", dataset="tpcds")

QUERY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tpcds_queries.sql")

# The delimiter `tools/vendor_tpcds_queries.py` writes before each statement.
_MARKER = "-- @query "
# TPC-DS is 99 queries by definition; anything else means the vendored file is wrong.
QUERY_COUNT = 99


def load_queries(path: str = QUERY_FILE) -> dict[str, str]:
    """Split the vendored ``.sql`` file into ``{case name -> statement}``.

    Args:
        path: The vendored query file.

    Returns:
        Each query's case name mapped to its SQL text, in file order.
    """
    queries: dict[str, str] = {}
    name: str | None = None
    lines: list[str] = []
    with open(path) as fh:
        text = fh.read()
    for line in text.splitlines():
        if line.startswith(_MARKER):
            if name is not None:
                queries[name] = "\n".join(lines).strip()
            name = line[len(_MARKER) :].strip()
            lines = []
        elif name is not None:
            lines.append(line)
    if name is not None:
        queries[name] = "\n".join(lines).strip()
    return queries


QUERIES = load_queries()

# A truncated or half-written vendored file would otherwise shrink the benchmark silently:
# fewer cases register, every gate stays green, and the suite quietly stops being TPC-DS.
if len(QUERIES) != QUERY_COUNT:
    raise RuntimeError(
        f"{QUERY_FILE} holds {len(QUERIES)} queries, expected {QUERY_COUNT} — "
        f"re-run `python tools/vendor_tpcds_queries.py`"
    )

for _name, _query in QUERIES.items():
    tpcds.sql(_name, _query)
