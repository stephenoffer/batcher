"""`dbapi` — the DB-API 2.0 (PEP 249) source and sink, split by responsibility.

`source` owns connections, cursors, pushdown, and partitioning. `_arrow` owns the
row-to-Arrow conversion, which is where every type-fidelity question on the read path lives
— a driver hands back Python objects and, for most drivers, no usable type information, so
getting to a typed batch without inventing a type or losing a value is the whole
correctness surface. `sink` is the mirror: the row-level write path (``INSERT`` /
``UPSERT`` / ``UPDATE`` / ``DELETE``), with `_statements` owning the dialect-specific SQL,
`_ddl` the Arrow-to-column-type mapping, `_bind` the Arrow-to-parameter conversion, and
`_dsn` the connection URI a PEP 249 driver cannot take whole. Keeping each apart from
connection handling is the seam.

Importing this package registers the ``dbapi`` source and sink; the public import paths are
unchanged.
"""

from __future__ import annotations

from batcher.io.formats.sql.dbapi._arrow import arrow_type, reconcile, rows_to_batch
from batcher.io.formats.sql.dbapi.sink import WRITE_MODES, DBAPISink
from batcher.io.formats.sql.dbapi.source import (
    DEFAULT_BATCH_SIZE,
    DBAPISource,
    _as_dbapi_connection,
)

__all__ = [
    "DEFAULT_BATCH_SIZE",
    "WRITE_MODES",
    "DBAPISink",
    "DBAPISource",
    "_as_dbapi_connection",
    "arrow_type",
    "reconcile",
    "rows_to_batch",
]
