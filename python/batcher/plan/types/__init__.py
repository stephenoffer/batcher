"""The neutral type vocabulary and inference for the plan layer.

Arrow types are Batcher's types; this package owns the *name* vocabulary the cast
surface and JSON IR use (`registry`), and (added incrementally) the lossless
promotion lattice and per-expression type inference that let the plan know a
column's output Arrow type before the engine runs.

Neutral layer: imports only `pyarrow` and `plan`; never `kyber`/`carbonite`/
`core`/`api`.
"""

from __future__ import annotations

from batcher.plan.types.compact import one_batch
from batcher.plan.types.footprint import (
    logical_bytes,
    retained_bytes,
    total_logical_bytes,
    total_retained_bytes,
)
from batcher.plan.types.infer import infer_type
from batcher.plan.types.lattice import promote, widen
from batcher.plan.types.registry import (
    CAST_DTYPES,
    DTYPE_REGISTRY,
    canonical_dtype_name,
    dtype_name,
    resolve_dtype,
)
from batcher.plan.types.widths import (
    DEFAULT_VARLEN_BYTES,
    column_bytes,
    projected_row_bytes,
    schema_row_bytes,
)

__all__ = [
    "CAST_DTYPES",
    "DEFAULT_VARLEN_BYTES",
    "DTYPE_REGISTRY",
    "canonical_dtype_name",
    "column_bytes",
    "dtype_name",
    "infer_type",
    "logical_bytes",
    "one_batch",
    "projected_row_bytes",
    "promote",
    "resolve_dtype",
    "retained_bytes",
    "schema_row_bytes",
    "total_logical_bytes",
    "total_retained_bytes",
    "widen",
]
