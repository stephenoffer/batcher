"""Every spelling the migration-guidance tables suggest must be one that actually works.

When a pandas or Polars user types a method Batcher does not carry, `__getattr__` raises an
error naming the Batcher spelling instead of a bare `AttributeError`. That message is the
first thing a migrant reads, and nothing executed it: the tables are plain strings, so a
suggestion could name a method that does not exist, spell a method as an attribute, or pass
an argument the engine rejects, and every test would stay green while the migrant hit a
*second*, more confusing error than the one they started with.

All three had happened. `at_time` and `between_time` suggested `bt.col('t').dt.hour` without
the call (and then `.is_between` on the resulting method object); four entries — `resample`,
`asfreq`, `upsample`, `group_by_dynamic` — suggested `.dt.truncate('1h')`, but `date_trunc`
takes a calendar unit and rejects a duration, so the recommended fix raised at execution.

These checks are deliberately structural rather than "exec the string": the strings are
prose with placeholder column names, so running them wholesale reports mostly noise. What
*is* mechanical is that every name mentioned exists and is spelled the way it is used, and
that every literal handed to a unit-taking function is in that function's vocabulary.
"""

from __future__ import annotations

import re

import pytest

import batcher as bt
from batcher.api.dataset.compat.guidance._dataset_table import DATASET_UNSUPPORTED
from batcher.api.dataset.compat.guidance._groupby_table import GROUPBY_UNSUPPORTED
from batcher.plan.expr_ir.compat.guidance import (
    DT_UNSUPPORTED,
    EXPR_UNSUPPORTED,
    LIST_UNSUPPORTED,
    STR_UNSUPPORTED,
)

pytestmark = pytest.mark.unit

TABLES = {
    "Dataset": DATASET_UNSUPPORTED,
    "GroupBy": GROUPBY_UNSUPPORTED,
    "Expr": EXPR_UNSUPPORTED,
    ".str": STR_UNSUPPORTED,
    ".list": LIST_UNSUPPORTED,
    ".dt": DT_UNSUPPORTED,
}

#: Names reached as attributes rather than called — the accessor namespaces and the
#: `Dataset` sub-APIs. Mentioning one without parentheses is correct.
_ATTRIBUTE_NAMES = frozenset(
    {"str", "dt", "list", "struct", "json", "map", "image", "audio", "video",
     "write", "read", "ml", "dq", "meta", "scd", "columns", "schema", "dtypes",
     "shape", "size", "width", "height", "empty", "is_streaming", "sql", "col", "lit",
     "udf", "security", "sem"}
)  # fmt: skip

_ACCESSOR_REF = re.compile(r"\.(dt|str|list|struct|json|map)\.(\w+)(\(?)")
_MODULE_REF = re.compile(r"\b(ds|bt)\.(\w+)(\(?)")

#: What follows a name when it is being *used as a value* rather than merely named in prose:
#: an operator, a closing bracket, a comma, or another attribute access. "bt.window takes a
#: duration" is prose; "bt.col('t').dt.hour == 9" is code, and only the second is a defect.
_VALUE_POSITION = re.compile(r"\s*(?:[=<>!+\-*/%&|)\],]|\.\w)")


def _used_as_a_value(text: str, end: int) -> bool:
    return _VALUE_POSITION.match(text, end) is not None


def _accessors():
    e = bt.col("x")
    return {"dt": e.dt, "str": e.str, "list": e.list, "struct": e.struct,
            "json": e.json, "map": e.map}  # fmt: skip


def _is_method(obj, name: str) -> bool:
    return callable(getattr(type(obj), name, None)) or callable(getattr(obj, name, None))


@pytest.mark.parametrize("label", sorted(TABLES))
def test_every_name_a_suggestion_mentions_exists(label):
    """A suggestion naming a method that is not there sends the migrant nowhere."""
    accessors = _accessors()
    ds = bt.from_pydict({"x": [1]})
    missing: list[str] = []
    for key, text in TABLES[label].items():
        for acc, name, _ in _ACCESSOR_REF.findall(text):
            if not hasattr(accessors[acc], name):
                missing.append(f"{label}[{key!r}] -> .{acc}.{name}")
        for mod, name, _ in _MODULE_REF.findall(text):
            target = ds if mod == "ds" else bt
            if not hasattr(target, name) and name not in _ATTRIBUTE_NAMES:
                missing.append(f"{label}[{key!r}] -> {mod}.{name}")
    assert not missing, "guidance names something that does not exist:\n  " + "\n  ".join(missing)


@pytest.mark.parametrize("label", sorted(TABLES))
def test_a_method_is_suggested_as_a_call_not_an_attribute(label):
    """`bt.col('t').dt.hour` is a bound method, not a value; the parentheses matter.

    A migrant who pastes the attribute form gets `filter() requires an expression ... got
    bool` or `'function' object has no attribute ...` — a worse error than the one the
    message was written to replace.
    """
    accessors = _accessors()
    ds = bt.from_pydict({"x": [1]})
    wrong: list[str] = []
    for key, text in TABLES[label].items():
        for m in _ACCESSOR_REF.finditer(text):
            acc, name, paren = m.groups()
            obj = accessors[acc]
            if paren or not hasattr(obj, name) or not _is_method(obj, name):
                continue
            if _used_as_a_value(text, m.end()):
                wrong.append(f"{label}[{key!r}] -> .{acc}.{name} (a method, written bare)")
        for m in _MODULE_REF.finditer(text):
            mod, name, paren = m.groups()
            target = ds if mod == "ds" else bt
            if paren or name in _ATTRIBUTE_NAMES or not hasattr(target, name):
                continue
            if _is_method(target, name) and _used_as_a_value(text, m.end()):
                wrong.append(f"{label}[{key!r}] -> {mod}.{name} (a method, written bare)")
    assert not wrong, "guidance spells a method as an attribute:\n  " + "\n  ".join(wrong)


#: Functions whose first string literal is a *unit*, and the vocabulary each accepts.
#: `date_trunc` takes a calendar unit and rejects a duration; `bt.window` is the opposite.
#: Suggesting the wrong one produces an error at execution, not at plan build.
_CALENDAR_UNITS = frozenset(
    {"millennium", "millenium", "century", "decade", "year", "quarter", "month", "week",
     "day", "hour", "minute", "second", "millisecond", "microsecond"}
)  # fmt: skip
_UNIT_CALL = re.compile(r"\.dt\.(?:truncate|floor|ceil|round)\(\s*'([^']+)'")
_DURATION_CALL = re.compile(r"\bbt\.window\(\s*[^,]+,\s*'([^']+)'")


@pytest.mark.parametrize("label", sorted(TABLES))
def test_a_unit_argument_is_one_the_function_accepts(label):
    """`.dt.truncate('1h')` raises: `date_trunc` names a unit, it does not parse a duration."""
    from batcher.plan.functions.temporal import _duration_micros

    bad: list[str] = []
    for key, text in TABLES[label].items():
        for unit in _UNIT_CALL.findall(text):
            if unit.lower() not in _CALENDAR_UNITS:
                bad.append(f"{label}[{key!r}] -> .dt.truncate/floor/ceil/round({unit!r})")
        for duration in _DURATION_CALL.findall(text):
            try:
                _duration_micros(duration, arg="window duration")
            except Exception:  # any rejection is exactly the failure being reported
                bad.append(f"{label}[{key!r}] -> bt.window(..., {duration!r})")
    assert not bad, "guidance passes a unit the function rejects:\n  " + "\n  ".join(bad)


def test_the_checks_would_notice_a_regression():
    """The guard on the guard: a table with each defect must be caught, not waved through."""
    accessors = _accessors()
    assert not hasattr(accessors["dt"], "no_such_method_here")
    assert _is_method(accessors["dt"], "hour"), ".dt.hour is a method, so the bare form is wrong"
    assert _used_as_a_value("bt.col('t').dt.hour == 9", len("bt.col('t').dt.hour"))
    assert not _used_as_a_value("bt.window takes a duration", len("bt.window"))
    assert "1h" not in _CALENDAR_UNITS, "a duration must not pass as a calendar unit"


def _owners():
    e = bt.col("x")
    ds = bt.from_pydict({"x": [1]})
    return {
        "Expr": e, ".str": e.str, ".list": e.list, ".dt": e.dt,
        "Dataset": ds, "GroupBy": ds.group_by(),
    }  # fmt: skip


@pytest.mark.parametrize("label", sorted(TABLES))
def test_no_entry_describes_a_name_that_now_exists(label):
    """An entry keyed on a name the surface has grown is unreachable, and misleading.

    The redirect fires from `__getattr__`, which only runs when normal lookup *fails*. So
    the moment `Dataset.upsample` (say) becomes real, its entry stops being read — and it
    keeps sitting there telling a future maintainer the capability is missing. Deleting it
    is part of adding the method, and this is what remembers to ask.
    """
    owner = _owners()[label]
    live = [
        key
        for key in TABLES[label]
        if hasattr(owner, key) and not isinstance(getattr(type(owner), key, None), property)
    ]
    assert not live, (
        f"{label} guidance still redirects names that exist: {live}. "
        "Delete the entry — `__getattr__` can no longer reach it."
    )
