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


def _is(t: pa.DataType, *predicates: str) -> bool:
    """Whether `t` matches any of `predicates`, on a `pyarrow` that defines them.

    The view and run-end predicates arrived in `pyarrow` well after the types they test
    for, so calling one unguarded turns an old runtime into an `AttributeError` at import
    of the type lattice.
    """
    return any(
        check(t) for name in predicates if (check := getattr(pa.types, name, None)) is not None
    )


def _is_int(t: pa.DataType) -> bool:
    return pa.types.is_integer(t)


def _is_float(t: pa.DataType) -> bool:
    return pa.types.is_floating(t)


def _is_string(t: pa.DataType) -> bool:
    return pa.types.is_string(t) or pa.types.is_large_string(t)


def _is_binary(t: pa.DataType) -> bool:
    return pa.types.is_binary(t) or pa.types.is_large_binary(t)


_TIME_UNITS = ("s", "ms", "us", "ns")

# The widest precision a `decimal128` carries, and the integer digits an `int64` needs
# when it is widened into one (`i64::MAX` has 19).
_DECIMAL128_MAX_PRECISION = 38
_INT64_DECIMAL_DIGITS = 19


def _finer(a: str, b: str) -> str:
    """The finer of two Arrow time units — converting up to it is exact."""
    return a if _TIME_UNITS.index(a) >= _TIME_UNITS.index(b) else b


def _unify_decimal(p1: int, s1: int, p2: int, s2: int) -> pa.DataType | None:
    """Keep the finer scale and the wider integer part, or ``None`` past 38 digits."""
    scale = max(s1, s2)
    int_digits = max(p1 - s1, p2 - s2)
    precision = int_digits + scale
    if not 0 < precision <= _DECIMAL128_MAX_PRECISION:
        return None
    return pa.decimal128(precision, scale)


def _promote_decimal(a: pa.DataType, b: pa.DataType) -> pa.DataType | None:
    """Promote a pair where at least one side is a ``decimal128``.

    A float dominates a decimal (DuckDB casts DECIMAL up to DOUBLE); an integer is
    widened *into* the decimal so a money column keeps its cents; two decimals unify
    field-wise. A boolean is deliberately not widened into a decimal: arrow has no
    bool-to-decimal cast, so naming one would advertise a type the engine cannot produce.
    """
    a_dec, b_dec = pa.types.is_decimal128(a), pa.types.is_decimal128(b)
    if _is_float(a) or _is_float(b):
        return pa.float64()
    if a_dec and b_dec:
        return _unify_decimal(a.precision, a.scale, b.precision, b.scale)
    if a_dec and _is_int(b):
        return _unify_decimal(a.precision, a.scale, _INT64_DECIMAL_DIGITS, 0)
    if b_dec and _is_int(a):
        return _unify_decimal(_INT64_DECIMAL_DIGITS, 0, b.precision, b.scale)
    return None


def _promote_temporal(a: pa.DataType, b: pa.DataType) -> pa.DataType | None:
    """Promote a pair of date/time/timestamp/duration types, or ``None``.

    Two instants in the same zone meet at the finer resolution; a date widens into a
    timestamp (a date is midnight, so nothing is lost, and DuckDB returns TIMESTAMP for
    ``DATE UNION TIMESTAMP``). A *differing* timezone is a genuine disagreement about
    which instant a stored value denotes, so it is declined.
    """
    a_date = pa.types.is_date(a)
    b_date = pa.types.is_date(b)
    if pa.types.is_timestamp(a) and pa.types.is_timestamp(b):
        return pa.timestamp(_finer(a.unit, b.unit), a.tz) if a.tz == b.tz else None
    if a_date and pa.types.is_timestamp(b):
        return b
    if pa.types.is_timestamp(a) and b_date:
        return a
    if a_date and b_date:
        return pa.date64()  # date32 (days) vs date64 (millis): the wider of the two
    if pa.types.is_time(a) and pa.types.is_time(b):
        # time32 carries s/ms and time64 us/ns, so any cross-family pairing lands in
        # time64 and a within-family one keeps its family at the finer unit.
        unit = _finer(a.unit, b.unit)
        return pa.time64(unit) if unit in ("us", "ns") else pa.time32(unit)
    if pa.types.is_duration(a) and pa.types.is_duration(b):
        return pa.duration(_finer(a.unit, b.unit))
    return None


def promote(a: pa.DataType, b: pa.DataType) -> pa.DataType | None:
    """The common supertype of `a` and `b` under a conservative, never-lossy lattice.

    ``null`` adopts the other side; integers widen to ``int64``; floats widen to
    ``float64``; an int/float mix promotes to ``float64``. Decimals keep the finer scale
    and the wider integer part; temporal types widen to the finer resolution and a date
    widens into a timestamp; a `string`/`large_string` pair meets at the large variant.
    Returns ``None`` when the types are not already equal and have no non-lossy common
    type (the caller decides whether that is a fallback or an error).

    This mirrors the engine's `bc_expr::common_supertype`, and the mirroring is
    load-bearing rather than incidental: this function is what `Dataset.schema` predicts
    for a union, a coalesce, and a case expression, while that one is what the engine
    actually produces. When the two disagreed, the control plane advertised ``int64`` for
    a ``null``-typed branch unioned with an ``int64`` one and the query then failed in the
    engine — a schema that lies is worse than a type error. Change one, change the other.
    """
    if a.equals(b):
        return a
    if pa.types.is_null(a):
        return b
    if pa.types.is_null(b):
        return a
    # A dictionary is an *encoding* of its value type, not a distinct logical type — a
    # dict-encoded and a plain column both read as the value type at the FFI boundary.
    if pa.types.is_dictionary(a):
        return promote(a.value_type, b)
    if pa.types.is_dictionary(b):
        return promote(a, b.value_type)
    if pa.types.is_decimal128(a) or pa.types.is_decimal128(b):
        return _promote_decimal(a, b)
    if _is_int(a) and _is_int(b):
        return pa.int64()
    if (_is_float(a) or _is_int(a)) and (_is_float(b) or _is_int(b)):
        return pa.float64()
    # Boolean widens into a number, as DuckDB does (`SELECT true UNION SELECT 1` is
    # INTEGER, with `true` reading as 1). It never widens into anything else.
    if pa.types.is_boolean(a) and _is_int(b):
        return pa.int64()
    if _is_int(a) and pa.types.is_boolean(b):
        return pa.int64()
    temporal = _promote_temporal(a, b)
    if temporal is not None:
        return temporal
    # Same logical type, wider offsets — lossless, exactly as int32/int64 widen. The
    # types are already known to differ, so a string-family pair here is precisely
    # `string` against `large_string`.
    if _is_string(a) and _is_string(b):
        return pa.large_string()
    if _is_binary(a) and _is_binary(b):
        return pa.large_binary()
    return None


def widen(dt: pa.DataType) -> pa.DataType:
    """Widen a narrow numeric type the way the FFI boundary does, else pass through.

    Mirrors ``bc_py::normalize_to``: Int8/16/32 and every unsigned int normalize to
    ``int64``; Float16/32 normalize to ``float64``; ``LargeUtf8`` normalizes to ``string``;
    a ``dictionary`` column is **decoded to its value type** (then widened). The *view*
    layouts normalize to the layout they respell — ``string_view`` to ``string``,
    ``binary_view`` to ``binary``, ``list_view``/``large_list_view`` to ``list`` — and a
    ``run_end_encoded`` column decodes to its value type, exactly as a dictionary does. The
    widening
    **recurses into nested types** — a ``struct<a: int32>`` becomes ``struct<a: int64>`` and
    a ``list<float32>`` becomes ``list<float64>`` — because the boundary widens a narrow
    numeric at every nesting depth, so a narrow field buried in a struct/list must be
    predicted widened too or ``Dataset.schema`` would lie (and later arithmetic on the
    widened engine value would disagree with the inferred narrow type). Booleans and
    strings are unchanged. Idempotent.

    The dictionary and ``LargeUtf8`` arms are what make this an actual mirror. Leaving them
    out did make ``Dataset.schema`` lie in exactly the way the paragraph above warns
    against: a dictionary-encoded column — what Parquet emits natively for a
    low-cardinality string — was reported as ``dictionary<values=string, indices=int32>``
    while ``collect()`` returned plain ``string``, which is what the boundary produces and
    what ``test_dictionary_decodes_to_value_type`` already pinned. Worse than a cosmetic
    lie: joining such a column against a plain-string one was rejected at build time with
    "join key type mismatch: left is dictionary<...> but right is string" for a join the
    engine would have run correctly, because both sides reach it decoded.
    """
    if _is(dt, "is_run_end_encoded"):
        # Decoded by `bc_py::decode_run_ends` at the *column* level, so this arm is not in
        # `_widen_nested` below: a run-end column buried in a struct or list stays encoded,
        # because decoding one means rebuilding the containing array rather than casting it.
        return widen(dt.value_type)
    return _widen_nested(dt)


def _widen_nested(dt: pa.DataType) -> pa.DataType:
    """The part of `widen` that recurses -- everything `bc_py::normalize_to` can `cast`."""
    if pa.types.is_dictionary(dt):
        return _widen_nested(dt.value_type)
    if pa.types.is_large_string(dt) or _is(dt, "is_string_view"):
        return pa.string()
    if _is(dt, "is_binary_view"):
        return pa.binary()
    if pa.types.is_integer(dt):
        # All narrow + unsigned ints normalize to int64 at the boundary; int64
        # itself is already wide and unchanged.
        return pa.int64()
    if pa.types.is_floating(dt):
        # Float16/32 → Float64; Float64 unchanged.
        return pa.float64()
    if pa.types.is_struct(dt):
        return pa.struct([f.with_type(_widen_nested(f.type)) for f in dt])
    if pa.types.is_list(dt) or pa.types.is_large_list(dt):
        vf = dt.value_field
        make = pa.large_list if pa.types.is_large_list(dt) else pa.list_
        return make(vf.with_type(_widen_nested(vf.type)))
    if _is(dt, "is_list_view", "is_large_list_view"):
        vf = dt.value_field
        return pa.list_(vf.with_type(_widen_nested(vf.type)))
    if pa.types.is_fixed_size_list(dt):
        vf = dt.value_field
        return pa.list_(vf.with_type(_widen_nested(vf.type)), dt.list_size)
    if pa.types.is_map(dt):
        return pa.map_(
            dt.key_field.with_type(_widen_nested(dt.key_type)),
            dt.item_field.with_type(_widen_nested(dt.item_type)),
            dt.keys_sorted,
        )
    return dt
