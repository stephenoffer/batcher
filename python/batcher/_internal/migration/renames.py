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
  `batcher.migrate.canonical`, with `to` naming the method the result calls.

Any rule may add `fill = ["<param>@<position>=<literal>"]` when the kept spelling's default
differs from the removed one's (a call that omits the argument gets it passed explicitly), and
`keys = ["<param>", ...]` when the kept spelling takes the arguments at other positions (the
call's positional arguments become those keywords), and `args = <n>` when the spelling stays on
the surface with another meaning and only a call with exactly `n` arguments is the old one
(`arg_max(by)` is `max_by(by)`; `arg_max()` is the position and is left alone).

A value in `kwarg_renames.toml`, under a `["<receiver>.<method>"]` table, is one of:
`"<new_name>"` (rename the keyword), `"*"` (a literal list becomes positional arguments),
`"@"` (the value becomes the first positional argument), or
`"!<new_name>"` (rename and logically negate a boolean), or `"nulls_first"` for the pandas
`na_position="first"|"last"` string.
"""

from __future__ import annotations

import ast
import dataclasses
import re
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
TRANSFORMS = frozenset({"with_column", "slice_to_limit", "identity"})


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
        fill: `(param, position, literal)` triples: when a call omits `param` both as a
            keyword and at `position`, the rewrite passes `param=literal`, because the kept
            spelling's default differs from the removed one's.
        keys: Keywords the call's positional arguments become, in order, because the kept
            spelling takes them at other positions (`clip_max(2)` is `clip(upper=2)`).
        args: For a spelling that stays with a new meaning, the exact argument count of a
            call in the old meaning; any other call is not rewritten. `None` for a spelling
            that is gone.
    """

    receiver: str
    removed: str
    kind: str
    to: str = ""
    operator: str = ""
    transform: str = ""
    fill: tuple[tuple[str, int, str], ...] = ()
    keys: tuple[str, ...] = ()
    args: int | None = None


@dataclass(frozen=True)
class KwargRename:
    """One second keyword spelling on a method that stays.

    Attributes:
        method: `"<receiver>.<method>"`.
        keyword: The keyword being removed.
        action: `rename`, `positional` (a literal list splatted into positional arguments),
            `first_positional` (the value passed whole as the first positional argument),
            `negate`, or `nulls_first`.
        to: The kept keyword, for `rename` and `negate`.
    """

    method: str
    keyword: str
    action: str
    to: str = ""


_FILL = re.compile(r"^(?P<param>[A-Za-z_]\w*)@(?P<pos>\d+)=(?P<literal>.+)$")


def _fills(where: str, raw: object) -> tuple[tuple[str, int, str], ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise RegistryError(f"{where}: fill must be a list of 'param@position=literal'")
    out = []
    for item in raw:
        match = _FILL.match(str(item))
        if match is None:
            raise RegistryError(f"{where}: fill entry {item!r} is not 'param@position=literal'")
        try:
            ast.literal_eval(match["literal"])
        except (ValueError, SyntaxError) as exc:
            raise RegistryError(
                f"{where}: fill literal {match['literal']!r} is not a literal"
            ) from exc
        out.append((match["param"], int(match["pos"]), match["literal"]))
    return tuple(out)


def _rule(receiver: str, removed: str, raw: object) -> Rename:
    rule = _base_rule(receiver, removed, raw)
    fill = _fills(
        f"renames.toml: {receiver}.{removed}", raw.get("fill") if isinstance(raw, dict) else None
    )
    keys = raw.get("keys", []) if isinstance(raw, dict) else []
    if not isinstance(keys, list) or not all(isinstance(k, str) and k.isidentifier() for k in keys):
        raise RegistryError(f"renames.toml: {receiver}.{removed}: keys must be a list of names")
    args = raw.get("args") if isinstance(raw, dict) else None
    if args is not None and (not isinstance(args, int) or isinstance(args, bool) or args < 0):
        raise RegistryError(f"renames.toml: {receiver}.{removed}: args must be a count")
    return dataclasses.replace(rule, fill=fill, keys=tuple(keys), args=args)


def _base_rule(receiver: str, removed: str, raw: object) -> Rename:
    where = f"renames.toml: {receiver}.{removed}"
    if isinstance(raw, str):
        return Rename(receiver, removed, "name", to=raw)
    if not isinstance(raw, dict):
        raise RegistryError(f"{where} must be a string or an inline table")
    unknown = set(raw) - {"to", "call", "operator", "transform", "fill", "keys", "args"}
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
            elif raw == "@":
                rules[keyword] = KwargRename(method, keyword, "first_positional")
            elif raw == "nulls_first":
                rules[keyword] = KwargRename(method, keyword, "nulls_first", to="nulls_first")
            elif raw.startswith("!"):
                rules[keyword] = KwargRename(method, keyword, "negate", to=raw[1:])
            else:
                rules[keyword] = KwargRename(method, keyword, "rename", to=raw)
        out[method] = rules
    return out
