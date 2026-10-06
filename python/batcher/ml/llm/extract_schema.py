"""The declared schema of an `extract` call: resolution, JSON Schema, and per-value coercion.

`extract` types a model's JSON answer against a schema the caller declares, so every batch
carries the same Arrow types whatever the model emitted. This module owns that declaration:
it resolves the caller's spelling into Arrow types once, at plan time, renders the matching
JSON Schema for guided decoding, and coerces each parsed value into its declared type at
run time while recording *why* a value did not fit.

A field is declared as one of:

* a Batcher dtype name (``"string"``, ``"int64"``, ``"date"``, ...), the flat case;
* a ``dict`` of fields, a nested **struct** whose fields are declared the same way;
* a one-element ``list`` (``["string"]``, ``[{"sku": "string"}]``), a **list** of that type;
* a ``set`` of strings (``{"low", "high"}``), an **enum**: a string column whose values are
  only ever one of the declared spellings;
* a `pyarrow.DataType` built from the above (``pa.list_(pa.string())``, ``pa.struct(...)``).

Every declared field is **required**: the model must emit the key. ``null`` is an allowed
value, because the instruction tells the model to use it for anything it cannot determine.
A missing key, a value that will not coerce, or an off-menu enum value becomes null in the
typed column and one message in the row's diagnostics, so a failure stays countable and
explainable instead of looking exactly like a model that answered null.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pyarrow as pa

from batcher._internal.errors import PlanError
from batcher.plan.types import CAST_DTYPES, DTYPE_REGISTRY

__all__ = ["FieldSpec", "coerce_row", "json_schema", "resolve_schema"]

# The JSON Schema type each Batcher dtype maps to, for guided decoding.
_JSON_TYPES: dict[str, str] = {
    "int64": "integer",
    "int32": "integer",
    "float64": "number",
    "float32": "number",
    "bool": "boolean",
    "string": "string",
}

#: The diagnostic recorded when a response does not parse to a JSON object at all.
NOT_AN_OBJECT = "response is not a JSON object"


@dataclass(frozen=True)
class FieldSpec:
    """One resolved field: its Arrow type plus what coercion needs beyond the type.

    Args:
        arrow_type: The Arrow type the column (or nested value) is built with.
        children: The struct's fields in declared order, for a struct.
        item: The element spec, for a list.
        choices: The permitted spellings, for an enum.
    """

    arrow_type: pa.DataType
    children: tuple[tuple[str, FieldSpec], ...] = ()
    item: FieldSpec | None = None
    choices: tuple[str, ...] = ()


def resolve_schema(schema: dict[str, Any]) -> dict[str, FieldSpec]:
    """Validate the declared schema and resolve every field to a `FieldSpec`."""
    if not isinstance(schema, dict) or not schema:
        raise PlanError("extract(): schema must declare at least one field")
    return {str(name): _resolve(spec, str(name)) for name, spec in schema.items()}


def _resolve(spec: object, path: str) -> FieldSpec:
    """One field's declaration as a `FieldSpec`, raising a `PlanError` naming `path`."""
    if isinstance(spec, str):
        if spec not in CAST_DTYPES:
            raise PlanError(
                f"extract(): unknown dtype {spec!r} for field {path!r}; "
                f"use one of {sorted(CAST_DTYPES)}"
            )
        return _from_arrow(DTYPE_REGISTRY[spec], path)
    if isinstance(spec, pa.DataType):
        return _from_arrow(spec, path)
    if isinstance(spec, dict):
        if not spec:
            raise PlanError(f"extract(): nested field {path!r} declares no fields")
        children = tuple((str(k), _resolve(v, f"{path}.{k}")) for k, v in spec.items())
        return FieldSpec(pa.struct([(k, c.arrow_type) for k, c in children]), children=children)
    if isinstance(spec, list):
        if len(spec) != 1:
            raise PlanError(
                f"extract(): list field {path!r} must declare exactly one element type, "
                f"such as ['string'] or [{{'sku': 'string'}}]; got {len(spec)}"
            )
        item = _resolve(spec[0], f"{path}[]")
        return FieldSpec(pa.list_(item.arrow_type), item=item)
    if isinstance(spec, (set, frozenset)):
        return _enum(spec, path)
    raise PlanError(
        f"extract(): field {path!r} is declared as {spec!r}. Declare a dtype name, a dict "
        "(struct), a one-element list (list), a set of strings (enum), or a pyarrow type."
    )


def _enum(spec: set | frozenset, path: str) -> FieldSpec:
    """An enum field: a string column restricted to `spec`'s spellings."""
    choices = sorted(spec) if all(isinstance(c, str) for c in spec) else None
    if not choices:
        raise PlanError(f"extract(): enum field {path!r} must be a non-empty set of strings")
    if len({c.strip().lower() for c in choices}) != len(choices):
        raise PlanError(f"extract(): enum field {path!r} values must differ ignoring case")
    return FieldSpec(pa.string(), choices=tuple(choices))


def _from_arrow(arrow_type: pa.DataType, path: str) -> FieldSpec:
    """A `FieldSpec` for an Arrow type, recursing through structs and lists."""
    if pa.types.is_struct(arrow_type):
        children = tuple((f.name, _from_arrow(f.type, f"{path}.{f.name}")) for f in arrow_type)
        return FieldSpec(arrow_type, children=children)
    if pa.types.is_list(arrow_type) or pa.types.is_large_list(arrow_type):
        return FieldSpec(arrow_type, item=_from_arrow(arrow_type.value_type, f"{path}[]"))
    # A dtype `coerce_row` cannot produce is rejected here rather than yielding an all-null
    # column, which is the worst possible outcome: the generation is already paid for, the
    # schema looks right, and nothing anywhere says the field was never extracted.
    if not _is_extractable(arrow_type):
        raise PlanError(
            f"extract(): dtype {arrow_type} (field {path!r}) cannot be extracted from a "
            "model's JSON output. Declare a string/number/bool/date/time/timestamp/binary "
            "field, a struct or a list of them instead, and cast it afterwards if you need "
            "another type."
        )
    return FieldSpec(arrow_type)


def _is_extractable(arrow_type: pa.DataType) -> bool:
    """Whether `_coerce` can produce a scalar of `arrow_type` from parsed JSON."""
    return bool(
        pa.types.is_boolean(arrow_type)
        or pa.types.is_integer(arrow_type)
        or pa.types.is_floating(arrow_type)
        or pa.types.is_string(arrow_type)
        or pa.types.is_large_string(arrow_type)
        or pa.types.is_binary(arrow_type)
        or pa.types.is_large_binary(arrow_type)
        or pa.types.is_date(arrow_type)
        or pa.types.is_time(arrow_type)
        or pa.types.is_timestamp(arrow_type)
    )


def json_schema(schema: dict[str, Any]) -> dict:
    """A JSON Schema for `schema`, to hand to ``vllm_engine(guided_json=...)``.

    Guided decoding constrains the model to emit exactly this shape, so every row parses.
    `extract` still types the result independently; this only removes the parse failures.
    A nested ``dict`` becomes a nested ``object``, a one-element ``list`` an ``array``, and
    a ``set`` of strings an ``enum``. Every declared field is listed as ``required`` at
    every level, which is what `extract` checks.

    Args:
        schema: Column name to field declaration, the same mapping `extract` takes: a
            Batcher dtype (``"string"``, ``"int64"``, ``"float64"``, ``"bool"``, ...), a
            nested ``dict``, a one-element ``list``, or a ``set`` of strings.

    Returns:
        A JSON Schema ``object`` with one property per field.

    Raises:
        PlanError: If a dtype is unknown or has no JSON Schema analogue.

    Examples:
        .. doctest::

            >>> from batcher.ml import json_schema
            >>> json_schema({"sentiment": "string", "score": "float64"})["properties"]
            {'sentiment': {'type': 'string'}, 'score': {'type': 'number'}}
            >>> nested = json_schema({"tags": ["string"], "level": {"high", "low"}})
            >>> nested["properties"]["tags"]
            {'type': 'array', 'items': {'type': 'string'}}
            >>> nested["properties"]["level"]
            {'type': 'string', 'enum': ['high', 'low']}
    """
    fields = resolve_schema(schema)
    return _object_schema(fields.items())


def _object_schema(children: Any) -> dict:
    """The JSON Schema ``object`` for a struct's `children`."""
    properties = {name: _field_schema(spec, name) for name, spec in children}
    return {"type": "object", "properties": properties, "required": list(properties)}


def _field_schema(spec: FieldSpec, name: str) -> dict:
    """The JSON Schema for one resolved field."""
    if spec.children:
        return _object_schema(spec.children)
    if spec.item is not None:
        return {"type": "array", "items": _field_schema(spec.item, name)}
    if spec.choices:
        return {"type": "string", "enum": list(spec.choices)}
    json_type = next(
        (j for d, j in _JSON_TYPES.items() if DTYPE_REGISTRY[d] == spec.arrow_type), None
    )
    if json_type is None:
        raise PlanError(
            f"json_schema(): dtype {spec.arrow_type} (field {name!r}) has no JSON Schema equivalent"
        )
    return {"type": json_type}


def coerce_row(parsed: object, fields: dict[str, FieldSpec]) -> tuple[dict[str, object], list[str]]:
    """One parsed response as declared-type values, plus the row's diagnostics.

    `parsed` is the leniently parsed model output. Anything that is not a JSON object nulls
    every field with the single diagnostic `NOT_AN_OBJECT`, rather than one "missing" per
    field, which would bury the actual cause.
    """
    problems: list[str] = []
    if not isinstance(parsed, dict):
        return dict.fromkeys(fields), [NOT_AN_OBJECT]
    values = {
        name: _coerce_field(parsed, name, spec, name, problems) for name, spec in fields.items()
    }
    return values, problems


def _coerce_field(
    obj: dict, name: str, spec: FieldSpec, path: str, problems: list[str]
) -> object | None:
    """`obj[name]` coerced to `spec`, recording a missing required key."""
    if name not in obj:
        problems.append(f"{path}: missing required field")
        return None
    return _coerce_value(obj[name], spec, path, problems)


def _coerce_value(value: object, spec: FieldSpec, path: str, problems: list[str]) -> object | None:
    """One value coerced to `spec`, or null with a diagnostic naming `path`."""
    if value is None:
        return None  # an explicit null is an answer the instruction asks for, not a failure
    if spec.children:
        if not isinstance(value, dict):
            problems.append(f"{path}: expected an object, got {_show(value)}")
            return None
        return {
            name: _coerce_field(value, name, child, f"{path}.{name}", problems)
            for name, child in spec.children
        }
    if spec.item is not None:
        if not isinstance(value, list):
            problems.append(f"{path}: expected an array, got {_show(value)}")
            return None
        item = spec.item
        return [_coerce_value(v, item, f"{path}[{i}]", problems) for i, v in enumerate(value)]
    if spec.choices:
        hit = _match_choice(value, spec.choices)
        if hit is None:
            problems.append(
                f"{path}: {_show(value)} is not one of {json.dumps(list(spec.choices))}"
            )
        return hit
    out = _coerce(value, spec.arrow_type)
    if out is None:
        problems.append(f"{path}: expected {spec.arrow_type}, got {_show(value)}")
    return out


def _match_choice(value: object, choices: tuple[str, ...]) -> str | None:
    """The declared spelling `value` names, ignoring case and surrounding whitespace."""
    if not isinstance(value, str):
        return None
    wanted = value.strip().lower()
    return next((c for c in choices if c.strip().lower() == wanted), None)


def _show(value: object, limit: int = 60) -> str:
    """A short JSON rendering of `value` for a diagnostic."""
    text = json.dumps(value, default=str)
    return text if len(text) <= limit else f"{text[: limit - 3]}..."


def _coerce(value: object, arrow_type: pa.DataType) -> object | None:
    """Coerce one parsed JSON scalar to `arrow_type`, or null if it cannot be.

    A model that returns ``"42"`` for an integer field, or ``"yes"`` for a boolean, is
    doing what models do; a per-value coercion recovers the row instead of losing it.
    """
    if value is None:
        return None
    try:
        if pa.types.is_boolean(arrow_type):
            if isinstance(value, bool):
                return value
            return {"true": True, "yes": True, "false": False, "no": False}.get(
                str(value).strip().lower()
            )
        if pa.types.is_integer(arrow_type):
            return _coerce_integer(value)
        if pa.types.is_floating(arrow_type):
            if isinstance(value, bool):
                return None
            # `pa.array([1.5], type=pa.float16())` rejects a Python float outright with
            # "Expected np.float16 instance", so a half-precision field did not degrade to
            # null like every other mismatch here — it raised and failed the whole batch.
            if pa.types.is_float16(arrow_type):
                import numpy as np

                return np.float16(value)
            return float(value)
        if pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
            return value if isinstance(value, str) else json.dumps(value)
        if pa.types.is_binary(arrow_type) or pa.types.is_large_binary(arrow_type):
            return value if isinstance(value, bytes) else str(value).encode()
        if pa.types.is_temporal(arrow_type):
            return _coerce_temporal(value, arrow_type)
    except (TypeError, ValueError):
        return None
    return None


def _coerce_temporal(value: object, arrow_type: pa.DataType) -> object | None:
    """Coerce one parsed JSON value to a date / time / timestamp, or null.

    A model asked for a date answers with an ISO string, because that is what dates look
    like in the text it was trained on — never with a `datetime` object, which JSON cannot
    carry anyway. Without this every temporal field of an extraction came back null for
    every row, after the generation had already been paid for.
    """
    import datetime as dt

    if isinstance(value, (dt.date, dt.time)):  # dt.datetime is a dt.date
        parsed: object = value
    elif isinstance(value, str):
        parsed = _parse_iso(value.strip(), arrow_type)
    else:
        return None
    if parsed is None:
        return None
    # A `date` where a timestamp is declared (and the reverse) is the ordinary mismatch: the
    # model answered "2024-01-05" for a field the schema types as a timestamp. Normalize
    # rather than null, since the value the model gave is unambiguous.
    if pa.types.is_date(arrow_type) and isinstance(parsed, dt.datetime):
        return parsed.date()
    if pa.types.is_timestamp(arrow_type) and not isinstance(parsed, dt.datetime):
        if isinstance(parsed, dt.date):
            return dt.datetime(parsed.year, parsed.month, parsed.day)
        return None
    if pa.types.is_time(arrow_type) and not isinstance(parsed, dt.time):
        return parsed.time() if isinstance(parsed, dt.datetime) else None
    return parsed


def _parse_iso(text: str, arrow_type: pa.DataType) -> object | None:
    """An ISO-8601 date, time, or datetime out of `text`, or `None` when it is neither.

    ``fromisoformat`` accepts a trailing ``Z`` from Python 3.11 on, which is the spelling a
    model emits most often, so no pre-processing is needed beyond the strip the caller did.
    """
    import datetime as dt

    if pa.types.is_time(arrow_type):
        parsers = (dt.time.fromisoformat, dt.datetime.fromisoformat)
    else:
        parsers = (dt.datetime.fromisoformat, dt.date.fromisoformat)
    for parse in parsers:
        try:
            return parse(text)
        except ValueError:
            continue
    return None


def _coerce_integer(value: object) -> int | None:
    """Coerce `value` to an int, or null when the narrowing would be **lossy**.

    ``int(float("3.9"))`` is 3, which is indistinguishable from a model that genuinely
    answered 3 — a wrong number in a column that looks healthy. Only an exactly integral
    value converts; anything else degrades to null like any other uncoercible value, so
    the failures stay countable. An `int` is taken directly rather than through `float`,
    which would silently round past 2**53.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        text = value.strip()
        try:
            return int(text)  # exact, and keeps precision beyond 2**53
        except ValueError:
            pass  # not an exact integer -> fall through to the float parse below
        value = float(text)
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    return None
