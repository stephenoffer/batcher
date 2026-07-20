"""Polars adapter — single-node DataFrame comparator with a SQL surface.

Operator-mix cases build on an eager ``pl.DataFrame``; the standard suites run
through ``pl.SQLContext`` (Polars covers a large SQL subset — queries it cannot
parse surface as ``n/a``/``PARTIAL``, never a wrong answer).

Two limits of that SQL surface are worth knowing before adding a suite to it. It
rejects implicit ``FROM a, b WHERE a.x = b.x`` joins, ``EXISTS`` subqueries, and
scalar-subquery comparisons — which is most of TPC-H. And its constant folding of
decimal literals is lossy: ``0.06 + 0.01`` folds to the ``f64`` one ulp *below*
``0.07``, so TPC-H q6's ``BETWEEN 0.06 - 0.01 AND 0.06 + 0.01`` silently dropped
every ``l_discount = 0.07`` row and returned a wrong revenue. TPC-H therefore runs
Polars through native lazy-DataFrame pipelines (``suites/standard/tpch_polars``),
the way Polars' own published TPC-H benchmark does; this SQL path still serves
TPC-DS, ClickBench, and the scan suite.
"""

from __future__ import annotations

import importlib.util
import re

import pyarrow as pa

from .base import Engine, Rename, SqlRunner

# Polars' SQL parser accepts the combined ANSI interval literal ``INTERVAL '90 days'``
# but rejects the equally-standard split form ``INTERVAL '90' DAY`` the TPC-H text
# uses. The two are identical in meaning, so rewriting the split form into the
# combined one lets Polars run the query on the *same* workload (a dialect
# adaptation, like the harness's date→timestamp normalization — never a result
# change). Queries Polars genuinely cannot express (scalar/correlated subquery
# comparisons) still surface as PARTIAL rather than a wrong answer.
_SPLIT_INTERVAL = re.compile(
    r"interval\s+'(\d+)'\s+(year|month|week|day|hour|minute|second)s?\b", re.I
)


def _polars_sql_dialect(query: str) -> str:
    """Rewrite split interval literals to the combined form Polars' parser accepts."""
    return _SPLIT_INTERVAL.sub(lambda m: f"INTERVAL '{m.group(1)} {m.group(2).lower()}s'", query)


class PolarsEngine(Engine):
    name = "polars"
    tier = "single"
    supports_sql = True

    @classmethod
    def available(cls) -> bool:
        return importlib.util.find_spec("polars") is not None

    def handle(self, table: pa.Table):
        import polars as pl

        return pl.from_arrow(table)

    def read_parquet(self, uri: str):
        import polars as pl

        # scan_parquet keeps it lazy; collect happens inside the case.
        return pl.scan_parquet(uri)

    def sql_runner(self, tables: dict[str, pa.Table]) -> SqlRunner:
        import polars as pl

        ctx = pl.SQLContext(eager=True)
        for name, tbl in tables.items():
            ctx.register(name, pl.from_arrow(tbl))
        return lambda query: ctx.execute(_polars_sql_dialect(query)).to_arrow()

    def sql_runner_scan(self, uris: dict[str, str], rename: Rename | None = None) -> SqlRunner:
        import polars as pl

        ctx = pl.SQLContext(eager=True)
        for name, uri in uris.items():
            scan = pl.scan_parquet(uri)
            cols = (rename or {}).get(name)
            if cols:
                scan = scan.rename(cols)
            ctx.register(name, scan)
        return lambda query: ctx.execute(_polars_sql_dialect(query)).to_arrow()

    def scan_sql_runner(self, glob: str) -> SqlRunner:
        import polars as pl

        def run(query: str) -> pa.Table:
            # `scan_parquet` expands the glob when it is built, so it must be rebuilt per
            # call for the listing cost to be measured rather than amortized away.
            ctx = pl.SQLContext(eager=True)
            ctx.register("t", pl.scan_parquet(glob))
            return ctx.execute(_polars_sql_dialect(query)).to_arrow()

        return run
