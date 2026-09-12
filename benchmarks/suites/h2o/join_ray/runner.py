"""Build an h2o ``join`` case: the SQL fanout, plus Ray Data's native pipeline."""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from registry import EngineQueries, sql_case

if TYPE_CHECKING:
    from context import Context

__all__ = ["JOIN_TABLES", "case_with_ray", "ray_impl"]

#: Every table the join task's five questions draw from.
JOIN_TABLES = ("x", "small", "medium", "big")


def ray_impl(name: str) -> Callable[[dict[str, Any]], Any] | None:
    """The Ray Data pipeline for benchmark case `name`, or `None` when there is none.

    Args:
        name: The benchmark case name, such as ``h2o-join-q1``.

    Returns:
        The registered pipeline, or `None` when Ray is absent or the case has no pipeline.
    """
    if not importlib.util.find_spec("ray"):
        return None
    from suites.h2o.join_ray import queries  # noqa: F401  (self-registering)
    from suites.h2o.join_ray.base import IMPLS

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
        if impl is not None and "ray" in ctx.names() and ctx.tables:
            handles = {t: ctx.handle(t, "ray") for t in JOIN_TABLES if t in ctx.tables}
            fns["ray"] = lambda: impl(handles)
        return fns

    return build
