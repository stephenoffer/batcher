"""Shared scaffolding for the operator-mix cases.

The operator-mix runs single relational operators over the real TPC-H ``lineitem`` /
``orders`` tables, the dataframe-API counterpart to the SQL-first standard suites. Its
job is to give the **non-SQL** engines — PyArrow (Acero) and Ray Data — a place to
compete, since they cannot run the standard SQL queries. The SQL-capable engines
(Batcher, DuckDB, Polars, Spark, Daft) express each case through the same one SQL
string, fanned out via the context's pre-registered runners.

So a case is: one SQL string (for every SQL engine) plus optional native callables
for PyArrow and Ray. The harness's correctness gate then checks them all agree.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import pyarrow as pa

from registry import EngineQueries

if TYPE_CHECKING:
    from context import Context


def sql_fanout(ctx: Context, sql: str) -> EngineQueries:
    """One SQL string fanned across every SQL-capable engine in the active lineup."""
    return {name: (lambda run=run: run(sql)) for name, run in ctx.sql_runners().items()}


def ray_to_arrow(ds: object) -> pa.Table:
    """Materialize a Ray Dataset to Arrow without a pandas round trip.

    Every native Ray case here used to end `pa.Table.from_pandas(ds.to_pandas())`, which for
    a full-relation result — `op-sort-string` returns 6M `l_comment` values — converts Arrow
    to a NumPy array of Python objects and back, inside the timed region, for one engine
    only. Every other engine in the lineup returns Arrow directly. The conversion can cost
    more than the operator it wraps, so the Ray column was measuring the harness.

    `to_arrow_refs()` is Ray Data's own Arrow accessor: the blocks are already Arrow, so this
    is a fetch and a concat rather than a format change.
    """
    import ray

    blocks = [b for b in ray.get(ds.to_arrow_refs()) if b is not None]
    if not blocks:
        return pa.table({})
    return pa.concat_tables(blocks) if len(blocks) > 1 else blocks[0]


def cannot_run(fns: EngineQueries, engine: str, reason: str) -> EngineQueries:
    """Replace `engine`'s runner with one that reports `reason` instead of executing.

    For an engine that does not merely lose a case but **cannot survive it**. The harness
    catches an exception and prints the row; it cannot catch a `SIGKILL`, and one engine
    dying that way takes the whole suite's process with it — the `operators` run died at
    case 8 of 11 having printed no table at all, so four working engines reported nothing.

    Raising the reason keeps that fact in the report (the row shows the engine failing, with
    this text in the `[PARTIAL]` line) instead of trading it for a silent kill. It is not a
    way to hide a loss: an engine that merely runs slowly must keep its runner and be timed.
    Every use states what was measured, so it can be re-checked when the engine changes.
    """
    if engine in fns:

        def refuse() -> object:
            raise RuntimeError(reason)

        fns[engine] = refuse
    return fns


def with_native(
    ctx: Context,
    fns: EngineQueries,
    *,
    pyarrow: Callable[[pa.Table], pa.Table] | None = None,
    ray: Callable[[object], pa.Table] | None = None,
) -> EngineQueries:
    """Add PyArrow / Ray native callables for ``lineitem`` when those engines are active.

    ``pyarrow`` receives the ``lineitem`` Arrow table; ``ray`` receives its Ray
    Dataset handle. Either is omitted (engine shows ``n/a``) when the engine is not in
    the lineup or no implementation is supplied for the case.
    """
    active = ctx.names()
    if pyarrow is not None and "pyarrow" in active:
        table = ctx.table("lineitem")
        fns["pyarrow"] = lambda: pyarrow(table)
    if ray is not None and "ray" in active:
        handle = ctx.handle("lineitem", "ray")
        fns["ray"] = lambda: ray(handle)
    return fns
