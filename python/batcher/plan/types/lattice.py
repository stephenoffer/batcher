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

    Mirrors ``bc_py::normalize_to``: Int8/16/32 and every unsigned int normalize to
    ``int64``; Float16/32 normalize to ``float64``. The widening **recurses into nested
    types** — a ``struct<a: int32>`` becomes ``struct<a: int64>`` and a ``list<float32>``
    becomes ``list<float64>`` — because the boundary widens a narrow numeric at every
    nesting depth, so a narrow field buried in a struct/list must be predicted widened too
    or ``Dataset.schema`` would lie (and later arithmetic on the widened engine value would
    disagree with the inferred narrow type). Booleans, strings, and dictionaries are
    unchanged. Idempotent.
    """
    if pa.types.is_integer(dt):
        # All narrow + unsigned ints normalize to int64 at the boundary; int64
        # itself is already wide and unchanged.
        return pa.int64()
    if pa.types.is_floating(dt):
        # Float16/32 → Float64; Float64 unchanged.
        return pa.float64()
    if pa.types.is_struct(dt):
        return pa.struct([f.with_type(widen(f.type)) for f in dt])
    if pa.types.is_list(dt) or pa.types.is_large_list(dt):
        vf = dt.value_field
        make = pa.large_list if pa.types.is_large_list(dt) else pa.list_
        return make(vf.with_type(widen(vf.type)))
    if pa.types.is_fixed_size_list(dt):
        vf = dt.value_field
        return pa.list_(vf.with_type(widen(vf.type)), dt.list_size)
    if pa.types.is_map(dt):
        return pa.map_(
            dt.key_field.with_type(widen(dt.key_type)),
            dt.item_field.with_type(widen(dt.item_type)),
            dt.keys_sorted,
        )
    return dt
