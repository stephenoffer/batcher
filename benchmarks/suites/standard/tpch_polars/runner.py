"""Look a TPC-H case up in the registry and adapt it to the suite's calling convention.

The suite hands a native implementation ``{table name -> engine handle}`` and expects
a ``pyarrow.Table`` back; the query modules speak ``LazyFrame`` in and out. This is the
one place those two shapes meet.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from typing import Any

import pyarrow as pa

__all__ = ["polars_impl"]


def polars_impl(name: str) -> Callable[[dict[str, Any]], pa.Table] | None:
    """The Polars pipeline for benchmark case ``name``, or ``None`` if there is none.

    The returned callable takes the suite's ``{table name -> eager pl.DataFrame}``
    handle map (what ``PolarsEngine.handle`` produces) and returns the query result as
    an Arrow table. Each call re-derives the ``LazyFrame``s and collects, so the timed
    region covers planning and execution the way the SQL engines' does.
    """
    if not importlib.util.find_spec("polars"):
        return None

    # Importing the query modules is what populates `IMPLS` (they self-register).
    from . import queries_a, queries_b  # noqa: F401
    from .base import IMPLS

    query = IMPLS.get(name)
    if query is None:
        return None

    def run(handles: dict[str, Any]) -> pa.Table:
        return query({t: h.lazy() for t, h in handles.items()}).collect().to_arrow()

    return run
