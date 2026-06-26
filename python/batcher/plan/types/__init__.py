"""The neutral type vocabulary and inference for the plan layer.

Arrow types are Batcher's types; this package owns the *name* vocabulary the cast
surface and JSON IR use (`registry`), and (added incrementally) the lossless
promotion lattice and per-expression type inference that let the plan know a
column's output Arrow type before the engine runs.

Neutral layer: imports only `pyarrow` and `plan`; never `kyber`/`carbonite`/
`core`/`api`.
"""

from __future__ import annotations

from batcher.plan.types.infer import infer_type
from batcher.plan.types.lattice import promote, widen
from batcher.plan.types.registry import CAST_DTYPES, DTYPE_REGISTRY

__all__ = ["CAST_DTYPES", "DTYPE_REGISTRY", "infer_type", "promote", "widen"]
