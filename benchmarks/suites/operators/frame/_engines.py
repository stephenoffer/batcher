"""One builder for a DataFrame-API case: each engine's own spelling over the same input.

A case states up to four things: a Batcher callable over a `Dataset`, a Polars callable over a
`LazyFrame`, one SQL string for the SQL engines that are not themselves under API test
(DuckDB on its native storage, `duckdb_arrow`, Spark), and a PyArrow callable over the Arrow
table. Anything left out reports `n/a`. Every handle is built here, outside the timed region,
so the timed callable is the operation and the result conversion and nothing else.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import pyarrow as pa

from registry import EngineQueries

if TYPE_CHECKING:
    from context import Context

# The SQL engines a DataFrame-API case still fans its SQL to. Batcher and Polars are the
# engines under API test, so they never take the SQL string here; Daft's SQL dialect lacks
# most of what these families exercise (PIVOT, ROLLUP, the regexp family), and a row of
# dialect errors would read as engine failures.
_SQL_ENGINES = ("duckdb", "duckdb_arrow", "spark")


def frame_case(
    ctx: Context,
    *,
    data: pa.Table | None = None,
    batcher: Callable[[Any], pa.Table] | None = None,
    polars: Callable[[Any], Any] | None = None,
    sql: str | None = None,
    pyarrow: Callable[[pa.Table], pa.Table] | None = None,
) -> EngineQueries:
    """Each active engine's callable for one case.

    Args:
        ctx: The run context.
        data: A derived input built from the TPC-H tables. `None` means `lineitem` itself.
        batcher: Receives a Batcher `Dataset` over the input; returns an Arrow table.
        polars: Receives a Polars `LazyFrame` over the input; returns a `LazyFrame` or
            `DataFrame`, which is collected and converted inside the timed call.
        sql: One query whose input table is written `{t}` (it is `lineitem` when `data` is
            `None`, and `t` over a derived input), so literal braces must be doubled.
        pyarrow: Receives the Arrow input; returns an Arrow table.

    Returns:
        Engine name to a zero-argument callable returning `pa.Table`.
    """
    names = set(ctx.names())
    table = ctx.table("lineitem") if data is None else data
    fns: EngineQueries = {}
    if batcher is not None and "batcher" in names:
        import batcher as bt

        ds = bt.from_arrow(table)
        fns["batcher"] = lambda: batcher(ds)
    if polars is not None and "polars" in names:
        import polars as pl

        lazy = pl.from_arrow(table).lazy()

        def run_polars() -> pa.Table:
            out = polars(lazy)
            return (out.collect() if isinstance(out, pl.LazyFrame) else out).to_arrow()

        fns["polars"] = run_polars
    if pyarrow is not None and "pyarrow" in names:
        fns["pyarrow"] = lambda: pyarrow(table)
    if sql is not None:
        fns.update(_sql_engines(ctx, names, sql, data))
    return fns


def _sql_engines(ctx: Context, names: set[str], sql: str, data: pa.Table | None) -> EngineQueries:
    """The SQL engines' callables: the engine's own runner, or one over the derived input."""
    fns: EngineQueries = {}
    if data is None:
        runners = ctx.sql_runners()
        query = sql.format(t="lineitem")
        for name in _SQL_ENGINES:
            if name in runners:
                fns[name] = lambda run=runners[name], q=query: run(q)
        return fns
    if "duckdb" in names:
        import duckdb

        from engines.duckdb import match_batcher_budget

        # Native storage and the same CPU/memory budget, as the `duckdb` engine runs
        # everywhere else (see `joins.py` and `engines/duckdb.py`).
        con = duckdb.connect()
        match_batcher_budget(con)
        con.register("t_arrow", data)
        con.execute("CREATE TABLE t AS SELECT * FROM t_arrow")
        con.unregister("t_arrow")
        query = sql.format(t="t")
        fns["duckdb"] = lambda: con.sql(query).to_arrow_table()
    if "duckdb_arrow" in names:
        import duckdb

        from engines.duckdb import match_batcher_budget

        arrow_con = duckdb.connect()
        match_batcher_budget(arrow_con)
        arrow_con.register("t", data)
        fns["duckdb_arrow"] = lambda: arrow_con.sql(sql.format(t="t")).to_arrow_table()
    return fns
