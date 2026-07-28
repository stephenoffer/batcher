"""The registry the TPC-H Ray Data query modules write into, plus their shared pieces.

Kept separate from ``__init__`` so the package façade stays a re-export shim and the
query modules have one obvious place to register themselves. Mirrors the layout of
``tpch_polars``.

Every pipeline is written against the same two primitives Ray Data gives a user with
no SQL surface: ``map_batches`` over PyArrow (the format its blocks already hold) for
projection/filter/arithmetic, and its native ``groupby``/``join`` for the shuffles.
The final ordering and ``LIMIT`` of each query run on the already-aggregated result,
which TPC-H keeps small -- the heavy work stays in Ray.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc

__all__ = [
    "IMPLS",
    "JOIN_PARTITIONS",
    "RayQuery",
    "impl",
    "join",
    "mb",
    "revenue",
    "take",
    "year_of",
]

# Ray Data's hash-join fans each side into this many partitions joined independently.
JOIN_PARTITIONS = 64

# One TPC-H query: {table name -> ray.data.Dataset}, returning the result as Arrow.
RayQuery = Callable[[dict[str, Any]], pa.Table]

# Benchmark case name (``tpch-q1``, ...) -> its Ray Data implementation.
IMPLS: dict[str, RayQuery] = {}


def impl(name: str) -> Callable[[RayQuery], RayQuery]:
    """Register a query implementation under its benchmark case name."""

    def register(fn: RayQuery) -> RayQuery:
        IMPLS[name] = fn
        return fn

    return register


def mb(ds: Any, fn: Callable[[pa.Table], pa.Table]) -> Any:
    """``map_batches`` in PyArrow format -- the format Ray Data's blocks already use."""
    return ds.map_batches(fn, batch_format="pyarrow")


def join(
    left: Any,
    right: Any,
    on: str | tuple[str, ...],
    right_on: str | tuple[str, ...] | None = None,
    how: str = "inner",
    **kwargs: Any,
) -> Any:
    """A Ray Data hash join with the suite's partition count, keys given as plain names.

    Ray's ``Dataset.join`` takes tuples and a required ``num_partitions``; wrapping it
    keeps the 20-odd join sites in the query modules readable and stops the partition
    count from being retyped (and mistyped) at each one.

    **The result is materialized**, which is what makes the multi-join queries finish.
    Left lazy, a query like q3 hands Ray's streaming executor two joins and a grouped
    aggregate as one plan, and it runs all three shuffles *concurrently* -- three pools
    of ``JOIN_PARTITIONS`` aggregators competing for the same cores. On a 96-core box
    that thrashes: q3 did not complete in 15 minutes, and Ray logged ``Cluster
    resources are not enough to run any task``. Materializing each join runs one
    shuffle at a time, and the same q3 finishes in about 22 seconds. Measured
    per-stage, nothing here is slow; only the overlap was.
    """
    keys = (on,) if isinstance(on, str) else on
    rkeys = (right_on,) if isinstance(right_on, str) else right_on
    return left.join(
        right,
        join_type=how,
        num_partitions=JOIN_PARTITIONS,
        on=keys,
        right_on=rkeys,
        **kwargs,
    ).materialize()


def take(ds: Any) -> list[dict]:
    """Materialize a (small, post-aggregation) Ray Dataset to a list of row dicts."""
    return ds.take_all()


def revenue(b: pa.Table) -> pa.Array:
    """``l_extendedprice * (1 - l_discount)`` -- the TPC-H revenue term.

    Eight of the 22 queries compute exactly this, so it lives here rather than being
    retyped per query.
    """
    return pc.multiply(b["l_extendedprice"], pc.subtract(1.0, b["l_discount"]))


def year_of(column: pa.Array) -> pa.Array:
    """``extract(year FROM ...)`` as int64, matching the SQL engines' output type."""
    return pc.cast(pc.year(column), pa.int64())
