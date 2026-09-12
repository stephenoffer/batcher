"""Shared pieces for the h2o ``groupby`` task's Ray Data pipelines.

Ray Data has no SQL surface, so every engine comparison on this suite was a blank column
until these existed. Each pipeline is written against the two primitives Ray Data actually
gives a user -- ``groupby(...).aggregate(...)`` for the shuffles, and ``map_batches`` over
PyArrow (the format its blocks already hold) for projection and arithmetic -- so the
measurement is of Ray Data doing the work, not of a wrapper doing it for Ray Data.

The final projection and column renaming run on the *aggregated* result, which this suite
keeps small in every question but q10; the heavy work stays inside Ray.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pyarrow as pa

__all__ = ["IMPLS", "RayQuery", "impl", "rename", "to_arrow"]

#: One h2o groupby question: {table name -> ray.data.Dataset} -> the result as Arrow.
RayQuery = Callable[[dict[str, Any]], pa.Table]

#: Benchmark case name (``h2o-gb-q1``, ...) -> its Ray Data implementation.
IMPLS: dict[str, RayQuery] = {}


def impl(name: str) -> Callable[[RayQuery], RayQuery]:
    """Register a query implementation under its benchmark case name.

    Args:
        name: The benchmark case name, such as ``h2o-gb-q1``.

    Returns:
        A decorator that records the function and returns it unchanged.
    """

    def register(fn: RayQuery) -> RayQuery:
        IMPLS[name] = fn
        return fn

    return register


def to_arrow(ds: Any) -> pa.Table:
    """Materialize a Ray Data dataset as one Arrow table.

    Args:
        ds: The dataset to collect.

    Returns:
        Every block concatenated into a single table.
    """
    blocks = [pa.Table.from_pydict(b) if isinstance(b, dict) else b for b in ds.to_arrow_refs()]
    import ray

    tables = [t if isinstance(t, pa.Table) else ray.get(t) for t in blocks]
    tables = [t for t in tables if t.num_rows]
    if not tables:
        return pa.table({})
    return pa.concat_tables(tables, promote_options="default")


def rename(table: pa.Table, mapping: dict[str, str]) -> pa.Table:
    """Rename columns and drop everything not named, in the mapping's order.

    Ray Data's aggregates name their outputs after the aggregate and column (``sum(v1)``),
    which is not what the benchmark's SQL projects; the correctness gate compares column
    names, so the result has to carry the query's own.

    Args:
        table: The aggregated result.
        mapping: Ray Data's column name -> the name the benchmark expects.

    Returns:
        A table of exactly the mapped columns, renamed.
    """
    return pa.table({new: table.column(old) for old, new in mapping.items()})
