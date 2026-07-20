"""The registry the TPC-H Polars query modules write into, plus their shared pieces.

Kept separate from ``__init__`` so the package façade stays a re-export shim and the
query modules have one obvious place to register themselves.
"""

from __future__ import annotations

from collections.abc import Callable

import polars as pl

__all__ = ["IMPLS", "PolarsQuery", "impl", "revenue"]

# One TPC-H query: table name -> lazy frame, returning the unmaterialized result.
PolarsQuery = Callable[[dict[str, "pl.LazyFrame"]], "pl.LazyFrame"]

# Benchmark case name (``tpch-q1``, ...) -> its Polars implementation.
IMPLS: dict[str, PolarsQuery] = {}


def impl(name: str) -> Callable[[PolarsQuery], PolarsQuery]:
    """Register a query implementation under its benchmark case name."""

    def register(fn: PolarsQuery) -> PolarsQuery:
        IMPLS[name] = fn
        return fn

    return register


def revenue(alias: str = "revenue") -> pl.Expr:
    """``sum(l_extendedprice * (1 - l_discount))`` — the TPC-H revenue aggregate.

    Eight of the 22 queries sum exactly this expression, so it lives here rather than
    being retyped (and mistyped) per query.
    """
    return (pl.col("l_extendedprice") * (1.0 - pl.col("l_discount"))).sum().alias(alias)
