"""The dtype-name ↔ Arrow-type vocabulary — the canonical cast-name grammar.

Arrow types ARE Batcher's types (see `plan/schema.py`); this module owns the *name*
vocabulary the public `cast` surface and the JSON IR use to refer to them.
`bc_expr::Expr::Cast` carries the dtype as a raw string on the wire, so this grammar is
part of the IR contract with the Rust engine. The canonical implementation lives in Rust
(`bc_arrow::dtype_name`); this module mirrors it so a bad dtype fails at plan-build time
with a clear message instead of surfacing as an opaque FFI error mid-execution, and a
parity test pins both halves to the live engine (`tests/unit/test_dtype_registry_parity`).

The vocabulary has two halves, and the split matters:

- `DTYPE_REGISTRY` holds the **fixed** names — `int64`, `string`, `date32` and their SQL
  aliases. It is a table, and `CAST_DTYPES` is its key set.
- `resolve_dtype` additionally parses the **parametrized** names, which carry their
  parameters in parentheses: ``decimal(12,4)``, ``timestamp(us, UTC)``, ``time64(ns)``,
  ``duration(s)``. Those cannot be enumerated — there are 38 x 39 legal decimals alone —
  so anything validating a user-supplied dtype must call `resolve_dtype`, never test
  membership in `CAST_DTYPES`.
"""

from __future__ import annotations

import pyarrow as pa

__all__ = [
    "CAST_DTYPES",
    "DTYPE_REGISTRY",
    "canonical_dtype_name",
    "dtype_name",
    "resolve_dtype",
]

# Cast dtype name → Arrow type, for the names that take no parameters. Mirrors
# `bc_arrow::dtype_name::fixed_dtype` exactly, including the aliases (`long`/`int64`,
# `double`/`float64`, and the SQL spellings a user porting a query reaches for first).
DTYPE_REGISTRY: dict[str, pa.DataType] = {
    "int64": pa.int64(),
    "long": pa.int64(),
    "bigint": pa.int64(),
    "int32": pa.int32(),
    "int": pa.int32(),
    "integer": pa.int32(),
    "int16": pa.int16(),
    "smallint": pa.int16(),
    "int8": pa.int8(),
    "tinyint": pa.int8(),
    "uint64": pa.uint64(),
    "ubigint": pa.uint64(),
    "uint32": pa.uint32(),
    "uinteger": pa.uint32(),
    "uint16": pa.uint16(),
    "usmallint": pa.uint16(),
    "uint8": pa.uint8(),
    "utinyint": pa.uint8(),
    "float64": pa.float64(),
    "double": pa.float64(),
    "float32": pa.float32(),
    "float": pa.float32(),
    "real": pa.float32(),
    "float16": pa.float16(),
    "half": pa.float16(),
    "bool": pa.bool_(),
    "boolean": pa.bool_(),
    "string": pa.string(),
    "utf8": pa.string(),
    "varchar": pa.string(),
    "text": pa.string(),
    "large_string": pa.large_string(),
    "large_utf8": pa.large_string(),
    "binary": pa.binary(),
    "blob": pa.binary(),
    "bytea": pa.binary(),
    "large_binary": pa.large_binary(),
    "date": pa.date32(),
    "date32": pa.date32(),
    "date64": pa.date64(),
    # The bare spellings keep the resolutions they have always had, so a plan already on
    # disk lowers to exactly the type it used to.
    "timestamp": pa.timestamp("us"),
    "datetime": pa.timestamp("us"),
    "time": pa.time64("us"),
    "time64": pa.time64("us"),
    "time32": pa.time32("ms"),
    "duration": pa.duration("us"),
    "interval": pa.duration("us"),
    "null": pa.null(),
}

# The set of accepted *fixed* dtype names. Parametrized names are a grammar rather than a
# set, so validation goes through `resolve_dtype`; this set is what an error message lists.
CAST_DTYPES: frozenset[str] = frozenset(DTYPE_REGISTRY)

# Time-unit names, in the Arrow spellings plus the words SQL users write.
_TIME_UNITS: dict[str, str] = {
    "s": "s",
    "sec": "s",
    "second": "s",
    "seconds": "s",
    "ms": "ms",
    "milli": "ms",
    "millisecond": "ms",
    "milliseconds": "ms",
    "us": "us",
    "micro": "us",
    "microsecond": "us",
    "microseconds": "us",
    "ns": "ns",
    "nano": "ns",
    "nanosecond": "ns",
    "nanoseconds": "ns",
}

# The resolutions each time-of-day width can physically carry. Arrow splits time-of-day
# across two widths and neither can carry the other's units.
_TIME64_UNITS = frozenset({"us", "ns"})

_DECIMAL_MAX_PRECISION = {False: 38, True: 76}


def _split_parametrized(name: str) -> tuple[str, list[str]] | None:
    """Split ``name(a, b)`` into its head and arguments, or ``None`` if not that shape.

    Arguments are trimmed but keep their case: a timezone identifier is case-sensitive
    where a type name is not.
    """
    open_at = name.find("(")
    if open_at <= 0 or not name.endswith(")"):
        return None
    inner = name[open_at + 1 : -1].strip()
    args = [a.strip() for a in inner.split(",")] if inner else []
    return name[:open_at].strip(), args


def _decimal(args: list[str], wide: bool) -> pa.DataType | None:
    """``decimal(p)`` (scale 0) or ``decimal(p, s)``, bounded by what the width carries.

    An out-of-range precision returns ``None`` rather than being clamped: silently
    building a ``decimal(38, 4)`` where the caller asked for ``decimal(50, 4)`` would
    overflow on the very values the extra digits were requested for.
    """
    if not args:
        return None
    try:
        precision = int(args[0])
        scale = int(args[1]) if len(args) > 1 else 0
    except ValueError:
        return None
    if not 0 < precision <= _DECIMAL_MAX_PRECISION[wide] or scale > precision:
        return None
    return pa.decimal256(precision, scale) if wide else pa.decimal128(precision, scale)


def _timestamp(args: list[str]) -> pa.DataType | None:
    """``timestamp(unit)`` or ``timestamp(unit, tz)``."""
    if not args or args[0] not in _TIME_UNITS:
        return None
    tz = args[1] if len(args) > 1 and args[1] else None
    return pa.timestamp(_TIME_UNITS[args[0]], tz)


def _time_of_day(args: list[str], wide: bool | None) -> pa.DataType | None:
    """``time(unit)`` / ``time32(unit)`` / ``time64(unit)``.

    The unqualified ``time(unit)`` picks the width the unit requires, which is the only
    spelling a user should have to know. An explicit ``time32(us)`` names an impossible
    type and returns ``None`` rather than silently promoting to ``time64`` — the caller
    asked for a specific physical width.
    """
    if not args or args[0] not in _TIME_UNITS:
        return None
    unit = _TIME_UNITS[args[0]]
    needs_64 = unit in _TIME64_UNITS
    if wide is None:
        return pa.time64(unit) if needs_64 else pa.time32(unit)
    if wide != needs_64:
        return None
    return pa.time64(unit) if wide else pa.time32(unit)


def canonical_dtype_name(dtype: str) -> str:
    """Lowercase a user-written dtype name, leaving a timezone identifier's case alone.

    Type names are matched case-insensitively because pandas spells them ``"Int64"`` and
    SQL ``"BIGINT"``, and a case mismatch is a typo the user cannot see. A **timezone** is
    the one part that must not be folded: Arrow stores it as a string and compares it
    byte-wise, so ``timestamp(us, utc)`` and ``timestamp(us, UTC)`` are different types
    and a blanket ``.lower()`` would silently produce the one the user did not ask for.

    Args:
        dtype: The dtype name as the user wrote it.

    Returns:
        The name with everything but a timezone argument lowercased.

    Examples:
        .. doctest::

            >>> from batcher.plan.types import canonical_dtype_name
            >>> canonical_dtype_name("DECIMAL(12,4)")
            'decimal(12, 4)'

            >>> canonical_dtype_name("TIMESTAMP(US, America/New_York)")
            'timestamp(us, America/New_York)'
    """
    parsed = _split_parametrized(dtype)
    if parsed is None:
        return dtype.lower()
    head, args = parsed
    head = head.lower()
    if head in ("timestamp", "datetime") and len(args) > 1:
        args = [args[0].lower(), *args[1:]]
    else:
        args = [a.lower() for a in args]
    return f"{head}({', '.join(args)})" if args else f"{head}()"


def resolve_dtype(name: str) -> pa.DataType | None:
    """Resolve a cast dtype name to its Arrow type, or ``None`` when nothing parses it.

    Accepts a fixed name (``int64``, ``string``, ...) or a parametrized one
    (``decimal(p,s)``, ``timestamp(unit[, tz])``, ``time32(unit)``, ``time64(unit)``,
    ``duration(unit)``). This is the function every caller should use: testing membership
    in `CAST_DTYPES` sees only the fixed half, which is how a parametrized cast that the
    engine accepts can be rejected at plan-build time.

    Args:
        name: The dtype name as it appears on the JSON IR wire, already lowercased
            except for a timezone identifier, which is case-sensitive.

    Returns:
        The Arrow type, or ``None`` if `name` is not a dtype this vocabulary knows.

    Examples:
        .. doctest::

            >>> from batcher.plan.types import resolve_dtype
            >>> resolve_dtype("decimal(12,4)")
            Decimal128Type(decimal128(12, 4))

            >>> resolve_dtype("not_a_type") is None
            True
    """
    fixed = DTYPE_REGISTRY.get(name)
    if fixed is not None:
        return fixed
    parsed = _split_parametrized(name)
    if parsed is None:
        return None
    head, args = parsed
    if head in ("decimal", "decimal128", "numeric"):
        return _decimal(args, wide=False)
    if head == "decimal256":
        return _decimal(args, wide=True)
    if head in ("timestamp", "datetime"):
        return _timestamp(args)
    if head == "time32":
        return _time_of_day(args, wide=False)
    if head == "time64":
        return _time_of_day(args, wide=True)
    if head == "time":
        return _time_of_day(args, wide=None)
    if head in ("duration", "interval"):
        return pa.duration(_TIME_UNITS[args[0]]) if args and args[0] in _TIME_UNITS else None
    return None


# The one name this vocabulary emits for each unparametrized Arrow type — the inverse of
# `DTYPE_REGISTRY`, which is many-to-one because of the aliases. Built explicitly rather
# than by inverting the table, so which spelling wins is a decision rather than an
# accident of dict ordering.
_CANONICAL_FIXED_NAME: list[tuple[pa.DataType, str]] = [
    (pa.null(), "null"),
    (pa.bool_(), "bool"),
    (pa.int8(), "int8"),
    (pa.int16(), "int16"),
    (pa.int32(), "int32"),
    (pa.int64(), "int64"),
    (pa.uint8(), "uint8"),
    (pa.uint16(), "uint16"),
    (pa.uint32(), "uint32"),
    (pa.uint64(), "uint64"),
    (pa.float16(), "float16"),
    (pa.float32(), "float32"),
    (pa.float64(), "float64"),
    (pa.string(), "string"),
    (pa.large_string(), "large_string"),
    (pa.binary(), "binary"),
    (pa.large_binary(), "large_binary"),
    (pa.date32(), "date32"),
    (pa.date64(), "date64"),
]


def dtype_name(dtype: pa.DataType) -> str | None:
    """The cast-target name for `dtype`, or ``None`` for a type the grammar cannot spell.

    The inverse of `resolve_dtype`, and round-trip exact: ``resolve_dtype(dtype_name(t))``
    is ``t`` for every type this returns a name for. That is the property callers depend
    on — anything building a `Cast` node from a *type* it computed (a join aligning two
    key columns, a plan rewrite reconciling a union) has to name that exact type, and a
    name that resolved to something merely similar would silently cast to the wrong thing.

    Returns ``None`` for nested and extension types, which the cast vocabulary does not
    name at all. Callers treat that as "cannot express this cast" and decline.

    Args:
        dtype: The Arrow type to name.

    Returns:
        The canonical name, or ``None`` when the grammar cannot spell `dtype`.

    Examples:
        .. doctest::

            >>> import pyarrow as pa
            >>> from batcher.plan.types import dtype_name
            >>> dtype_name(pa.decimal128(12, 4))
            'decimal(12, 4)'

            >>> dtype_name(pa.timestamp("us", "UTC"))
            'timestamp(us, UTC)'

            >>> dtype_name(pa.list_(pa.int64())) is None
            True
    """
    for candidate, name in _CANONICAL_FIXED_NAME:
        if dtype.equals(candidate):
            return name
    if pa.types.is_decimal128(dtype):
        return f"decimal({dtype.precision}, {dtype.scale})"
    if pa.types.is_decimal256(dtype):
        return f"decimal256({dtype.precision}, {dtype.scale})"
    if pa.types.is_timestamp(dtype):
        return f"timestamp({dtype.unit}, {dtype.tz})" if dtype.tz else f"timestamp({dtype.unit})"
    if pa.types.is_time32(dtype):
        return f"time32({dtype.unit})"
    if pa.types.is_time64(dtype):
        return f"time64({dtype.unit})"
    if pa.types.is_duration(dtype):
        return f"duration({dtype.unit})"
    return None
