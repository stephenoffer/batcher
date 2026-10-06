"""The per-call bindings of a SQL statement: the tables it names and the values it binds.

`bt.sql`, `Session.sql` and `ds.sql` take the same two kinds of binding, and this module is
the one place either is resolved, so the three entry points cannot drift apart.

**Tables** arrive as a ``{name: table}`` mapping, as keywords, or both. Each is coerced to
something the session can scan: a `Dataset` and a pyarrow table pass through, and anything a
``bt.from_*`` constructor understands is converted here.

**Values** (``params=``) fill the query's parameter placeholders: ``?`` and ``$1`` take a
sequence, ``$name`` and ``:name`` take a mapping. A value is substituted into the *parsed*
tree as a typed literal node, never spliced into the query text, so a string holding a
quote or a ``'; DROP TABLE`` is only ever a string. Each Python type maps to the SQL type
DuckDB's own parameter binding gives it: a `Decimal` keeps its precision and scale, a
`datetime` is a ``TIMESTAMP`` (``TIMESTAMPTZ`` when it carries a time zone), a float is a
``DOUBLE`` rather than the ``DECIMAL`` the same digits would be written as, and `bytes` are
a ``BLOB``.

This is the `api` layer; it imports no subsystem.
"""

from __future__ import annotations

import datetime as dt
import math
import numbers
from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Any

from sqlglot import expressions as exp

from batcher._internal.errors import PlanError
from batcher._internal.sql_errors import PLACEHOLDER_OFFSET

__all__ = ["bind_params", "bind_tables", "params_key"]

#: The types `params=` accepts, named in the refusal of any other.
_SUPPORTED = (
    "None, bool, int, float, str, bytes, Decimal, datetime.date, datetime.datetime, datetime.time"
)

#: Placeholder styles, which a query may not mix.
_POSITIONAL, _NUMBERED, _NAMED = "?", "$n", "$name"


def bind_tables(tables: Mapping[str, Any] | None, keywords: Mapping[str, Any]) -> dict[str, Any]:
    """Merge the mapping and keyword table bindings, coercing each to a scannable table.

    Args:
        tables: The ``{name: table}`` mapping, or None.
        keywords: The keyword bindings; a name in both takes the keyword's table.

    Returns:
        Name to `Dataset`, pyarrow table or record batch.

    Raises:
        PlanError: If `tables` is not a mapping.
    """
    import pyarrow as pa

    from batcher.api.dataset import Dataset
    from batcher.api.session.frameworks import from_any

    if tables is not None and not isinstance(tables, Mapping):
        raise PlanError(
            "sql(): the second positional argument must be a {name: table} mapping, "
            f"got {type(tables).__name__}"
        )
    out: dict[str, Any] = {}
    for name, value in {**(tables or {}), **keywords}.items():
        passes = isinstance(value, (Dataset, pa.Table, pa.RecordBatch))
        out[name] = value if passes else from_any(value)
    return out


def params_key(params: Sequence[Any] | Mapping[str, Any] | None) -> tuple:
    """A hashable key for `params`, for the prepared-statement cache.

    Each value is tagged with its type, so ``1``, ``1.0`` and ``True``, which compare equal
    in Python and bind to three different SQL types, never share a cached plan.

    Args:
        params: The values passed as ``params=``.

    Returns:
        A tuple that is equal for two calls exactly when they bind the same values.
    """
    if params is None:
        return ()
    items = sorted(params.items()) if isinstance(params, Mapping) else enumerate(params)
    return tuple((k, type(v).__qualname__, repr(v)) for k, v in items)


def bind_params(ast: Any, params: Sequence[Any] | Mapping[str, Any] | None) -> Any:
    """Replace every parameter placeholder in `ast` with the typed literal for its value.

    Args:
        ast: The parsed statement; edited in place.
        params: A sequence for ``?``/``$1`` placeholders, a mapping for ``$name``/``:name``.

    Returns:
        `ast`, with no placeholder left in it.

    Raises:
        PlanError: For placeholders without `params`, `params` without placeholders, mixed
            styles, a missing or unused value, or a value of an unsupported type.
    """
    slots = [n for n in ast.walk() if isinstance(n, (exp.Placeholder, exp.Parameter))]
    if not slots:
        if params:
            raise PlanError(
                "params= was given, but the query has no parameter placeholders",
                hint="Write ? or $name in the query where each value goes.",
            )
        return ast
    if params is None:
        raise PlanError(
            f"the query has {len(slots)} parameter placeholder(s) but no params= were given",
            hint="Pass params=[...] for ? or $1, or params={...} for $name.",
        )
    keyed = [(slot, *_slot_key(slot)) for slot in slots]
    styles = {style for _, style, _ in keyed}
    if len(styles) > 1:
        raise PlanError(
            f"a query may not mix parameter styles; it uses {' and '.join(sorted(styles))}"
        )
    style = styles.pop()
    if style == _NAMED:
        values = _by_name(keyed, params)
    else:
        values = _by_position(keyed, params, numbered=style == _NUMBERED)
    for slot, value in values:
        slot.replace(_literal(value))
    return ast


def _slot_key(slot: Any) -> tuple[str, Any]:
    """A placeholder's style and its key: an offset for ``?``, an index or a name otherwise."""
    if isinstance(slot, exp.Parameter):
        inner = slot.this
        text = inner.name if isinstance(inner, exp.Expression) else str(inner)
    else:
        text = slot.this
    if text is None or text == "":
        return _POSITIONAL, slot.meta.get(PLACEHOLDER_OFFSET, 0)
    text = str(text)
    if text.isdigit():
        return _NUMBERED, int(text)
    return _NAMED, text


def _by_position(keyed: list, params: Any, *, numbered: bool) -> list[tuple[Any, Any]]:
    """Pair ``?`` placeholders with a sequence in written order, or ``$n`` with its index."""
    if isinstance(params, (str, bytes, Mapping)) or not isinstance(params, Sequence):
        raise PlanError(
            f"? and $1 placeholders take a sequence of values, got {type(params).__name__}",
            hint="Use $name placeholders to bind a mapping.",
        )
    if not numbered:
        ordered = sorted(keyed, key=lambda entry: entry[2])
        if len(ordered) != len(params):
            raise PlanError(
                f"the query has {len(ordered)} ? placeholder(s) but params= has "
                f"{len(params)} value(s)"
            )
        return [(slot, value) for (slot, _, _), value in zip(ordered, params, strict=True)]
    indices = {index for _, _, index in keyed}
    if indices != set(range(1, len(params) + 1)):
        raise PlanError(
            f"$n placeholders must use each of $1..${len(params)} for {len(params)} value(s); "
            f"the query uses {', '.join(f'${i}' for i in sorted(indices))}"
        )
    return [(slot, params[index - 1]) for slot, _, index in keyed]


def _by_name(keyed: list, params: Any) -> list[tuple[Any, Any]]:
    """Pair ``$name`` placeholders with a mapping's values."""
    if not isinstance(params, Mapping):
        raise PlanError(
            f"$name placeholders take a mapping of values, got {type(params).__name__}",
            hint="Use ? placeholders to bind a sequence.",
        )
    names = {name for _, _, name in keyed}
    missing, unused = sorted(names - set(params)), sorted(set(params) - names)
    if missing or unused:
        parts = [f"missing value(s) for {missing}"] if missing else []
        parts += [f"unused value(s) {unused}"] if unused else []
        raise PlanError(f"params= does not match the query's placeholders: {'; '.join(parts)}")
    return [(slot, params[name]) for slot, _, name in keyed]


def _cast(text: str, sql_type: str) -> Any:
    return exp.Cast(this=exp.Literal.string(text), to=exp.DataType.build(sql_type))


def _literal(value: Any) -> Any:
    """The sqlglot node a Python value binds to.

    Raises:
        PlanError: For a type with no SQL literal, or bytes the engine cannot carry.
    """
    if value is None:
        return exp.Null()
    if isinstance(value, bool):
        return exp.Boolean(this=value)
    if isinstance(value, numbers.Integral):
        return exp.convert(int(value))
    if isinstance(value, Decimal):
        return _decimal(value)
    if isinstance(value, numbers.Real):
        number = float(value)
        if math.isfinite(number):
            return exp.Cast(this=exp.convert(number), to=exp.DataType.build("DOUBLE"))
        return _cast(repr(number), "DOUBLE")  # 'nan', 'inf', '-inf'
    if isinstance(value, str):
        return exp.Literal.string(value)
    if isinstance(value, dt.datetime):
        aware = value.tzinfo is not None and value.utcoffset() is not None
        return _cast(value.isoformat(sep=" "), "TIMESTAMPTZ" if aware else "TIMESTAMP")
    if isinstance(value, dt.date):
        return _cast(value.isoformat(), "DATE")
    if isinstance(value, dt.time):
        if value.tzinfo is not None:
            raise PlanError(f"params= cannot bind a time with a time zone: {value!r}")
        return _cast(value.isoformat(), "TIME")
    if isinstance(value, (bytes, bytearray, memoryview)):
        return _binary(bytes(value))
    raise PlanError(
        f"params= cannot bind a value of type {type(value).__name__}",
        hint=f"Supported types: {_SUPPORTED}.",
    )


def _decimal(value: Decimal) -> Any:
    """``CAST('<digits>' AS DECIMAL(p, s))``, with the precision and scale `value` has."""
    if not value.is_finite():
        raise PlanError(f"params= cannot bind the non-finite Decimal {value!r}")
    _sign, digits, exponent = value.as_tuple()
    assert isinstance(exponent, int)  # finite, checked above
    scale = max(-exponent, 0)
    precision = max(len(digits) + max(exponent, 0), scale, 1)
    if precision > 38:
        raise PlanError(f"params= cannot bind {value!r}: it needs more than 38 digits")
    return _cast(format(value, "f"), f"DECIMAL({precision}, {scale})")


def _binary(value: bytes) -> Any:
    """``CAST('<text>' AS BLOB)``, the one binary literal the engine reads exactly."""
    try:
        text = value.decode("utf-8")
    except UnicodeDecodeError:
        raise PlanError(
            "params= can bind bytes only when they are valid UTF-8: the engine has no "
            "literal for arbitrary binary values",
            hint="Pass the bytes in a table column instead, e.g. tables={'b': ...}.",
        ) from None
    return _cast(text, "BLOB")
