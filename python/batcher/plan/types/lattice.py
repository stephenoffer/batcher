"""The lossless numeric type lattice and the FFI narrow-widening mirror.

Two pure functions over `pyarrow.DataType`, shared by every part of the control
plane that has to reason about a column's type without running the engine:

- `promote(a, b)` — the conservative, never-lossy common supertype (the basis of
  `Union`/`Coalesce`/`Case` output types and the `io` multi-file schema
  reconciliation). Returns `None` when there is no non-lossy common type, so
  callers can degrade gracefully (inference) or raise a typed error (`io`).
- `widen(dt)` — the Python mirror of the engine's FFI narrow-type widening
  (`bc_py::widen_to`): the boundary normalizes narrow numerics once on input, so
  type *inference* must predict the same widened types the engine actually
  produces, or `Dataset.schema` would lie.

Neutral layer: imports only `pyarrow`.
"""

from __future__ import annotations

import pyarrow as pa

__all__ = ["promote", "widen"]


def _is_int(t: pa.DataType) -> bool:
    return pa.types.is_integer(t)


def _is_float(t: pa.DataType) -> bool:
    return pa.types.is_floating(t)


def promote(a: pa.DataType, b: pa.DataType) -> pa.DataType | None:
    """The common supertype of `a` and `b` under a conservative, never-lossy lattice.

    ``null`` adopts the other side; integers widen to ``int64``; floats widen to
    ``float64``; an int/float mix promotes to ``float64``. Returns ``None`` when the
    types are not already equal and have no non-lossy common type (the caller
    decides whether that is a fallback or an error).
    """
    if a.equals(b):
        return a
    if pa.types.is_null(a):
        return b
    if pa.types.is_null(b):
        return a
    if _is_int(a) and _is_int(b):
        return pa.int64()
    if (_is_float(a) or _is_int(a)) and (_is_float(b) or _is_int(b)):
        return pa.float64()
    return None


def widen(dt: pa.DataType) -> pa.DataType:
    """Widen a narrow numeric type the way the FFI boundary does, else pass through.

    Mirrors ``bc_py::widen_to``: Int8/16/32 and every unsigned int normalize to
    ``int64``; Float16/32 normalize to ``float64``. Wider numerics, booleans,
    strings, and nested types are unchanged. Idempotent.
    """
    if pa.types.is_integer(dt):
        # All narrow + unsigned ints normalize to int64 at the boundary; int64
        # itself is already wide and unchanged.
        return pa.int64()
    if pa.types.is_floating(dt):
        # Float16/32 → Float64; Float64 unchanged.
        return pa.float64()
    return dt
