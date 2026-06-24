"""The public, fluent, lazy, expression-first API surface.

`api` is the conductor: it builds `LogicalPlan`s and orchestrates the three layers
(Kyber → Carbonite → Core) to execute them. It is the only package allowed to
import all three layers. This module is a re-export façade — the expression
functions come from `api.functions` (re-exported wholesale, governed by its
``__all__``), plus the Dataset, IO, and a curated set of session constructors.
"""

from __future__ import annotations

from batcher.api import functions as _functions
from batcher.api.dataset import Dataset, GroupBy
from batcher.api.functions import *  # noqa: F403  (governed by functions.__all__)
from batcher.api.io_namespace import read
from batcher.api.session import (
    catalog,
    compact,
    date_range,
    engine_version,
    from_arrow,
    from_batches,
    from_dask,
    from_huggingface,
    from_items,
    from_numpy,
    from_pandas,
    from_polars,
    from_pydict,
    from_pylist,
    from_ray_dataset,
    from_spark,
    from_tf,
    from_torch,
    range,
    read_memory,
    register_function,
    sql,
    streams,
)
from batcher.api.sql_session import Session
from batcher.plan.streaming import OutputMode, Trigger

# Session names listed as literals so ruff recognizes the explicit imports above as
# re-exports; the expression functions come in via `*_functions.__all__`.
__all__ = [
    "Dataset",
    "GroupBy",
    "OutputMode",
    "Session",
    "Trigger",
    "catalog",
    "compact",
    "date_range",
    "engine_version",
    "from_arrow",
    "from_batches",
    "from_dask",
    "from_huggingface",
    "from_items",
    "from_numpy",
    "from_pandas",
    "from_polars",
    "from_pydict",
    "from_pylist",
    "from_ray_dataset",
    "from_spark",
    "from_tf",
    "from_torch",
    "range",
    "read",
    "read_memory",
    "register_function",
    "sql",
    "streams",
    *_functions.__all__,
]
