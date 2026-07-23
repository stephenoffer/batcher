"""`dbapi` — the DB-API 2.0 (PEP 249) source, split by responsibility.

`source` owns connections, cursors, pushdown, and partitioning. `_arrow` owns the
row-to-Arrow conversion, which is where every type-fidelity question on this path lives
— a driver hands back Python objects and, for most drivers, no usable type information,
so getting to a typed batch without inventing a type or losing a value is the whole
correctness surface. Keeping that apart from connection handling is the seam.

Importing this package registers the ``dbapi`` source exactly as the single module it
replaced did, and the public import path is unchanged.
"""

from __future__ import annotations

from batcher.io.formats.sql.dbapi._arrow import arrow_type, reconcile, rows_to_batch
from batcher.io.formats.sql.dbapi.source import (
    DEFAULT_BATCH_SIZE,
    DBAPISource,
    _as_dbapi_connection,
)

__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DBAPISource",
    "_as_dbapi_connection",
    "arrow_type",
    "reconcile",
    "rows_to_batch",
]
