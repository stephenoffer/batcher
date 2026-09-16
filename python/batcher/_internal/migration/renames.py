"""The rename decisions for Batcher's own second spellings, as typed rules the codemod can apply.

`data/renames.toml` records, per receiver, each spelling being removed and what replaces it.
Most replacements are another attribute name, but not all of them are: a pandas `to_csv` is
reached through another accessor (`write.csv`), a Polars `height` property is a method call
(`count()`), an operator-method `add` is an operator, and a handful change argument shape
(`with_column(name, expr)`). `data/kwarg_renames.toml` records the second keyword spellings on
methods that stay (`sort(ascending=...)`). This module turns both files into `Rename` and
`KwargRename` rules and rejects a decision the codemod could not apply.

A value in `renames.toml` is either a string (the kept attribute name) or an inline table:

* `{ to = "write.csv" }` - the kept spelling is a dotted path from the same receiver;
* `{ to = "count", call = true }` - a property whose replacement is a zero-argument call;
* `{ operator = "add" }` - `a.add(b)` becomes `a + b` (`invert`/`neg` are unary);
* `{ transform = "<name>", to = "<kept>" }` - an argument reshaping implemented by name in
  `batcher.migrate.transforms`, with `to` naming the method the result calls.

A value in `kwarg_renames.toml`, under a `["<receiver>.<method>"]` table, is one of:
`"<new_name>"` (rename the keyword), `"*"` (the value becomes positional arguments), or
`"!<new_name>"` (rename and logically negate a boolean), or `"nulls_first"` for the pandas
`na_position="first"|"last"` string.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from batcher._internal.migration.schema import RegistryError

__all__ = [
    "OPERATORS",
    "TRANSFORMS",
    "KwargRename",
    "Rename",
    "load_kwarg_renames",
    "load_renames",
]

_DATA = Path(__file__).resolve().parent / "data"

# Operator methods and the operator each becomes. `invert` and `neg` are unary.
OPERATORS = {
    "add": "+",
    "sub": "-",
    "mul": "*",
    "truediv": "/",
    "floordiv": "//",
    "mod": "%",
    "pow": "**",
    "and": "&",
    "or": "|",
    "xor": "^",
    "eq": "==",
    "ne": "!=",
    "lt": "<",
    "le": "<=",
    "gt": ">",
    "ge": ">=",
    "invert": "~",
    "neg": "-",
}

# Argument reshapings the codemod implements by name.
TRANSFORMS = frozenset({"with_column", "slice_to_limit"})


@dataclass(frozen=True)
class Rename:
    """What one removed spelling on one receiver becomes.

    Attributes:
        receiver: The receiver the removed spelling is typed on.
        removed: The spelling being removed.
        kind: `name`, `path`, `call`, `operator`, or `transform`.
        to: The kept attribute name, or dotted path, the rewrite calls.
        operator: For `kind == "operator"`, the key into `OPERATORS`.
        transform: For `kind == "transform"`, the key into `TRANSFORMS`.
    """

    receiver: str
    removed: str
    kind: str
    to: str = ""
    operator: str = ""
    transform: str = ""


@dataclass(frozen=True)
class KwargRename:
    """One second keyword spelling on a method that stays.

    Attributes:
        method: `"<receiver>.<method>"`.
        keyword: The keyword being removed.
        action: `rename`, `positional`, `negate`, or `nulls_first`.
        to: The kept keyword, for `rename` and `negate`.
    """

    method: str
    keyword: str
    action: str
    to: str = ""


def _rule(receiver: str, removed: str, raw: object) -> Rename:
    where = f"renames.toml: {receiver}.{removed}"
    if isinstance(raw, str):
        return Rename(receiver, removed, "name", to=raw)
    if not isinstance(raw, dict):
        raise RegistryError(f"{where} must be a string or an inline table")
    unknown = set(raw) - {"to", "call", "operator", "transform"}
    if unknown:
        raise RegistryError(f"{where}: unknown field(s) {sorted(unknown)}")
    if "operator" in raw:
        op = str(raw["operator"])
        if op not in OPERATORS:
            raise RegistryError(f"{where}: operator {op!r} is not one of {sorted(OPERATORS)}")
        return Rename(receiver, removed, "operator", operator=op)
    to = str(raw.get("to", ""))
    if not to:
        raise RegistryError(f"{where} needs `to`")
    if "transform" in raw:
        name = str(raw["transform"])
        if name not in TRANSFORMS:
            raise RegistryError(f"{where}: transform {name!r} is not one of {sorted(TRANSFORMS)}")
        return Rename(receiver, removed, "transform", to=to, transform=name)
    if raw.get("call"):
        return Rename(receiver, removed, "call", to=to)
    return Rename(receiver, removed, "path" if "." in to else "name", to=to)


@lru_cache(maxsize=1)
def load_renames() -> dict[str, dict[str, Rename]]:
    """Batcher's own second spellings, as `{receiver: {removed: Rename}}`.

    Returns:
        The rules in `data/renames.toml`.

    Raises:
        RegistryError: On a malformed rule, or a kept name that is itself removed on the
            same receiver, which would make a chain the codemod could only apply in order.
    """
    doc = tomllib.loads((_DATA / "renames.toml").read_text())
    out: dict[str, dict[str, Rename]] = {}
    for receiver, table in doc.items():
        rules = {removed: _rule(receiver, removed, raw) for removed, raw in table.items()}
        for rule in rules.values():
            if rule.to and rule.to.split(".")[0] in rules:
                raise RegistryError(
                    f"renames.toml: {receiver}.{rule.removed} -> {rule.to}, "
                    f"but {rule.to.split('.')[0]} is removed too"
                )
        out[receiver] = rules
    return out


@lru_cache(maxsize=1)
def load_kwarg_renames() -> dict[str, dict[str, KwargRename]]:
    """Second keyword spellings on kept methods, as `{"<receiver>.<method>": {kw: rule}}`.

    Returns:
        The rules in `data/kwarg_renames.toml`.

    Raises:
        RegistryError: On a value that is not one of the documented actions.
    """
    doc = tomllib.loads((_DATA / "kwarg_renames.toml").read_text())
    out: dict[str, dict[str, KwargRename]] = {}
    for method, table in doc.items():
        rules: dict[str, KwargRename] = {}
        for keyword, raw in table.items():
            if not isinstance(raw, str) or not raw:
                raise RegistryError(f"kwarg_renames.toml: {method}({keyword}=) must be a string")
            if raw == "*":
                rules[keyword] = KwargRename(method, keyword, "positional")
            elif raw == "nulls_first":
                rules[keyword] = KwargRename(method, keyword, "nulls_first", to="nulls_first")
            elif raw.startswith("!"):
                rules[keyword] = KwargRename(method, keyword, "negate", to=raw[1:])
            else:
                rules[keyword] = KwargRename(method, keyword, "rename", to=raw)
        out[method] = rules
    return out
