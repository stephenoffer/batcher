"""Build an h2o ``groupby`` case: the SQL fanout, plus Ray Data's native pipeline.

Mirrors ``suites.standard.tpch_ray.runner``. Ray Data has no SQL surface, so without this
the whole suite reported ``n/a`` for it -- a blank column that reads like a loss and is
actually an absence. The SQL engines keep the SQL string; Ray gets the pipeline.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from registry import EngineQueries, sql_case

if TYPE_CHECKING:
    from context import Context

__all__ = ["case_with_ray", "ray_impl"]

#: The one table the groupby task uses.
GROUPBY_TABLES = ("x",)


def ray_impl(name: str) -> Callable[[dict[str, Any]], Any] | None:
    """The Ray Data pipeline for benchmark case `name`, or `None` when there is none.

    Args:
        name: The benchmark case name, such as ``h2o-gb-q1``.

    Returns:
        The registered pipeline, or `None` when Ray is absent or the case has no pipeline.
    """
    if not importlib.util.find_spec("ray"):
        return None
    from suites.h2o.groupby_ray import queries  # noqa: F401  (self-registering)
    from suites.h2o.groupby_ray.base import IMPLS

    return IMPLS.get(name)


def case_with_ray(name: str, query: str) -> Callable[[Context], EngineQueries]:
    """The SQL fanout for `query`, plus Ray Data's pipeline for `name`.

    Args:
        name: The benchmark case name.
        query: The SQL the case runs, which every SQL engine gets unchanged.

    Returns:
        A builder producing one callable per participating engine.
    """
    sql_build = sql_case(query)
    impl = ray_impl(name)

    def build(ctx: Context) -> EngineQueries:
        fns = sql_build(ctx)
        # `ctx.tables` is empty in scan mode, where each table is a lazy parquet scan and a
        # handle map would be empty -- the pipeline would `KeyError` and shadow the SQL
        # runner that does work there. Same guard as TPC-H's.
        if impl is not None and "ray" in ctx.names() and ctx.tables:
            handles = {t: ctx.handle(t, "ray") for t in GROUPBY_TABLES if t in ctx.tables}
            fns["ray"] = lambda: impl(handles)
        return fns

    return build
