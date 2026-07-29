"""Build a TPC-H case: the SQL fanout, plus the native DataFrame pipelines.

Mirrors the operator-mix ``with_native`` mechanism but threads *all* TPC-H table
handles (a query joins several) so the DataFrame engines compete on the full query:
Ray Data (no SQL surface) via its ``ray.data.Dataset`` pipeline, batcher via its
native ``bt.Dataset`` pipeline (a parse-free, apples-to-apples counterpart to Ray
Data), and Polars via its lazy DataFrame API -- the way Polars' own published TPC-H
benchmark is written. SQL engines keep the SQL string.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from registry import EngineQueries, sql_case

if TYPE_CHECKING:
    from context import Context

__all__ = ["case_with_ray", "ray_impl"]

# Tables a Ray pipeline may ask for (handles built lazily, only when ray is active).
TPCH_TABLES = (
    "lineitem",
    "orders",
    "customer",
    "part",
    "supplier",
    "partsupp",
    "nation",
    "region",
)


def ray_impl(name: str) -> Callable[[dict[str, Any]], Any] | None:
    """The Ray Data pipeline for benchmark case ``name``, or ``None`` if there is none."""
    if not importlib.util.find_spec("ray"):
        return None

    # Importing the query modules is what populates `IMPLS` (they self-register).
    from . import queries_a, queries_b, queries_c  # noqa: F401
    from .base import IMPLS

    return IMPLS.get(name)


def case_with_ray(name: str, query: str) -> Callable[[Context], EngineQueries]:
    """The SQL fanout for ``query``, plus every native pipeline registered for ``name``."""
    from suites.standard.tpch_dataframe import batcher_impl
    from suites.standard.tpch_polars import polars_impl

    sql_build = sql_case(query)
    bt_impl = batcher_impl(name)
    pl_impl = polars_impl(name)

    def build(ctx: Context) -> EngineQueries:
        fns = sql_build(ctx)

        def native(engine: str, impl: Callable[[dict[str, Any]], Any] | None) -> None:
            # The native (DataFrame / Ray) impls build on in-memory table handles. In scan
            # mode there are none (`ctx.tables` is empty -- each table is a lazy parquet
            # scan), so registering the impl here would hand it an empty handle map and it
            # would `KeyError` on its first table lookup, shadowing the SQL runner that does
            # work in scan mode. Skip native entirely when there are no in-memory tables.
            if impl is not None and engine in ctx.names() and ctx.tables:
                handles = {t: ctx.handle(t, engine) for t in TPCH_TABLES if t in ctx.tables}
                fns[engine] = lambda: impl(handles)

        native("ray", ray_impl(name))
        native("batcher", bt_impl)
        native("polars", pl_impl)
        return fns

    return build
