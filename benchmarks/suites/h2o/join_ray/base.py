"""The h2o ``join`` task as Ray Data pipelines.

Five joins of one 10-million-row left side against progressively larger right sides, which
is the shape the task exists to measure. Ray Data's ``Dataset.join`` is a hash join that
fans both sides into ``JOIN_PARTITIONS`` independently-joined partitions; everything else
here is column selection, which runs as ``map_batches`` over the PyArrow blocks Ray already
holds.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pyarrow as pa

__all__ = ["IMPLS", "JOIN_PARTITIONS", "RayQuery", "impl", "join", "select", "to_arrow"]

#: Ray Data's hash join fans each side into this many independently joined partitions.
#: The same count TPC-H's pipelines use, for the same reason: it is the knob that decides
#: how much of the box a single join gets.
JOIN_PARTITIONS = 64

#: One h2o join question: {table name -> ray.data.Dataset} -> the result as Arrow.
RayQuery = Callable[[dict[str, Any]], pa.Table]

#: Benchmark case name (``h2o-join-q1``, ...) -> its Ray Data implementation.
IMPLS: dict[str, RayQuery] = {}


def impl(name: str) -> Callable[[RayQuery], RayQuery]:
    """Register a query implementation under its benchmark case name.

    Args:
        name: The benchmark case name, such as ``h2o-join-q1``.

    Returns:
        A decorator that records the function and returns it unchanged.
    """

    def register(fn: RayQuery) -> RayQuery:
        IMPLS[name] = fn
        return fn

    return register


def join(left: Any, right: Any, on: str, how: str = "inner") -> Any:
    """A Ray Data hash join on one key, with the suite's partition count.

    Args:
        left: The left dataset.
        right: The right dataset.
        on: The single join key, present in both sides under the same name.
        how: ``"inner"`` or ``"left_outer"``.

    Returns:
        The joined dataset.
    """
    return left.join(right, join_type=how, num_partitions=JOIN_PARTITIONS, on=(on,))


def select(ds: Any, columns: list[str]) -> Any:
    """Keep `columns`, in order, dropping everything else.

    Args:
        ds: The dataset to project.
        columns: The column names to keep.

    Returns:
        A dataset carrying only those columns.
    """

    def take(batch: pa.Table) -> pa.Table:
        return batch.select([c for c in columns if c in batch.schema.names])

    return ds.map_batches(take, batch_format="pyarrow")


def to_arrow(ds: Any) -> pa.Table:
    """Materialize a Ray Data dataset as one Arrow table.

    Args:
        ds: The dataset to collect.

    Returns:
        Every non-empty block concatenated into a single table.
    """
    import ray

    tables = [ray.get(ref) for ref in ds.to_arrow_refs()]
    tables = [t for t in tables if t.num_rows]
    if not tables:
        return pa.table({})
    return pa.concat_tables(tables, promote_options="default")
