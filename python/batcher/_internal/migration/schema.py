"""The shape of one migration-registry row: a competitor's name and what it becomes here.

Batcher keeps one spelling per capability and never adds a competitor's name as a second
one. A migrating user therefore needs somewhere that says, for every name they already
type, what the Batcher spelling is, whether it means the same thing, and if not, what is
different. That somewhere is the registry, and this module is its row type.

The registry is data rather than code for two reasons. It is large (every public name in
PySpark, Polars, Daft and Ray Data, several thousand rows), and it is read by consumers in
very different layers: the migration-error guidance on `Expr` (layer 1) and `Dataset`
(layer 5), the `batcher.migrate` codemod, the census harnesses and the documentation
generator. Holding it here, at layer 0 with no imports above `_internal`, is what lets all of
them read the same rows.

Each row carries a `Status`, and the status decides which other fields must be present.
`validate` enforces that, so a row missing the one field that makes it actionable (a
mismatch with no note saying what differs, a gap with no wave) fails at load time rather
than surfacing as a blank cell in the generated docs.

A row may also carry a `template`: how the codemod rewrites a call whose arguments do not carry
over unchanged, or whose difference a rewrite can restore. It is a small DSL rather than code,
so every template is validated here at load time and none can import or execute anything:

.. code-block:: text

    <name>(<parameters>) -> <expression>     one way: foreign -> Batcher only
    <name>(<parameters>) <-> <expression>    reversible: Batcher -> foreign as well

The left side is the foreign call as a Python parameter list, which binds the call's arguments
exactly as Python would (positional, keyword, defaults, `*args`, `**kwargs`). The right side is
one Python expression over four kinds of name:

* the parameters, which substitute the call's argument, or the parameter's default when the
  call omits it, so the rewrite states the foreign default explicitly;
* `self`, the rewritten receiver the method was called on;
* `bt`, the Batcher module;
* `sem.<transform>(...)`, a named transform in `batcher.migrate.semantics` for what the DSL
  cannot say (Java date patterns, join `how` spellings, Spark window specs). A transform may
  decline, and a declined template leaves the call alone with a marker comment.

`*args` and `**kwargs` splice where they appear starred, and `**{"name": value}` with an
identifier key renders as `name=value`, so `withColumn(colName, col) ->
self.with_columns(**{colName: col})` turns `df.withColumn("total", e)` into
`df.with_columns(total=e)`. A reversible template's right side must be one call on `self` or
`bt` whose arguments are parameters (bare or starred) or literals, so its inverse is a pure
re-binding of the Batcher call's arguments onto the left side.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from enum import StrEnum

__all__ = [
    "ENGINES",
    "WAVES",
    "Mapping",
    "RegistryError",
    "Status",
    "Template",
    "parse_template",
    "validate",
]

ENGINES = ("pyspark", "polars", "daft", "ray_data")

# The delivery waves a not-yet-parity row is scheduled into. `WF` is the foundations wave
# that has to land before the codemod can translate the rest.
WAVES = ("WF", *(f"W{i}" for i in range(15)))

_TARGET = re.compile(r"^(op:[a-z_]+|[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*)$")


class RegistryError(ValueError):
    """A registry file holds a row that cannot be used as written."""


class Status(StrEnum):
    """How a competitor's name relates to Batcher's surface.

    `CANONICAL` means the capability exists with the same semantics under the Batcher
    spelling in `batcher`, whatever the competitor calls it. `PARAM` means the capability
    exists but lacks an option the competitor offers (`need`). `GAP` means there is no
    capability yet. `MISMATCH` means both engines have it and answer differently (`note`).
    `OUT_OF_SCOPE` is a deliberate decline (`reason`).
    """

    CANONICAL = "canonical"
    PARAM = "param"
    GAP = "gap"
    MISMATCH = "mismatch"
    OUT_OF_SCOPE = "out_of_scope"


# Which optional fields each status requires. `batcher` is the target spelling.
_REQUIRED: dict[Status, tuple[str, ...]] = {
    Status.CANONICAL: ("batcher",),
    Status.PARAM: ("batcher", "need", "wave"),
    Status.GAP: ("need", "wave"),
    Status.MISMATCH: ("batcher", "note", "wave"),
    Status.OUT_OF_SCOPE: ("reason",),
}

_FIELDS = frozenset({"status", "batcher", "template", "need", "note", "reason", "wave"})


@dataclass(frozen=True)
class Mapping:
    """One competitor name on one of its surfaces, and its relation to Batcher.

    Attributes:
        engine: The competitor, one of `ENGINES`.
        surface: The receiver the name is typed on, such as `DataFrame` or `Expr.str`.
        name: The competitor's spelling.
        status: How the name relates to Batcher's surface.
        batcher: The Batcher spellings the capability is reached through, as dotted paths
            rooted at a receiver (`Dataset.with_columns`, `Expr.str.starts_with`,
            `bt.coalesce`) or an operator (`op:add`). Empty for a pure gap.
        template: How a call is rewritten when the arguments do not carry over unchanged,
            in the codemod's argument DSL.
        need: What is missing, for a parameter gap or a gap.
        note: What differs, for a mismatch.
        reason: Why the name is declined, for an out-of-scope row.
        wave: The delivery wave that closes a parameter gap, gap, or mismatch.
    """

    engine: str
    surface: str
    name: str
    status: Status
    batcher: tuple[str, ...] = ()
    template: str | None = None
    need: str | None = None
    note: str | None = None
    reason: str | None = None
    wave: str | None = None


def validate(engine: str, surface: str, name: str, raw: dict[str, object]) -> Mapping:
    """Build a `Mapping` from one TOML entry, rejecting any row that cannot be acted on.

    Args:
        engine: The competitor the file belongs to.
        surface: The TOML table the entry sits under.
        name: The entry's key.
        raw: The entry's inline table.

    Returns:
        The validated row.

    Raises:
        RegistryError: On an unknown field or status, a missing required field, a
            malformed Batcher target, or an unknown wave.
    """
    where = f"{engine}:{surface}.{name}"
    unknown = set(raw) - _FIELDS
    if unknown:
        raise RegistryError(f"{where}: unknown field(s) {sorted(unknown)}")
    try:
        status = Status(str(raw.get("status")))
    except ValueError as exc:
        raise RegistryError(f"{where}: status {raw.get('status')!r} is not a Status") from exc
    for field in _REQUIRED[status]:
        if not raw.get(field):
            raise RegistryError(f"{where}: status {status.value} requires {field!r}")
    target = raw.get("batcher", ())
    targets = (target,) if isinstance(target, str) else tuple(target)  # type: ignore[arg-type]
    for t in targets:
        if not isinstance(t, str) or not _TARGET.match(t):
            raise RegistryError(f"{where}: batcher target {t!r} is not a dotted path")
    wave = raw.get("wave")
    if wave is not None and wave not in WAVES:
        raise RegistryError(f"{where}: wave {wave!r} is not one of {WAVES}")
    if raw.get("template") is not None:
        try:
            parse_template(str(raw["template"]), name)
        except RegistryError as exc:
            raise RegistryError(f"{where}: {exc}") from exc
    return Mapping(
        engine=engine,
        surface=surface,
        name=name,
        status=status,
        batcher=targets,
        template=_opt_str(raw, "template"),
        need=_opt_str(raw, "need"),
        note=_opt_str(raw, "note"),
        reason=_opt_str(raw, "reason"),
        wave=_opt_str(raw, "wave"),
    )


def _opt_str(raw: dict[str, object], key: str) -> str | None:
    value = raw.get(key)
    return None if value is None else str(value)


_RESERVED = frozenset({"self", "bt", "sem", "True", "False", "None"})


@dataclass(frozen=True)
class Template:
    """One parsed registry template.

    Attributes:
        name: The foreign spelling the left side calls.
        params: The left side's parameter list, as parsed by `ast`.
        target: The right side's expression source.
        reversible: Whether the template was written with `<->`.
    """

    name: str
    params: ast.arguments
    target: str
    reversible: bool


def parse_template(text: str, name: str) -> Template:
    """Parse and validate one template against the DSL in the module docstring.

    Args:
        text: The template, `<name>(<params>) -> <expr>` or with `<->`.
        name: The registry row's name, which the left side must call.

    Returns:
        The parsed template.

    Raises:
        RegistryError: On a side that does not parse, a left side calling another name, a
            right side using a name that is neither a parameter nor `self`/`bt`/`sem`, or a
            reversible template whose right side is not a plain re-binding call.
    """
    reversible = "<->" in text
    left, sep, right = text.partition("<->" if reversible else "->")
    if not sep:
        raise RegistryError(f"template {text!r} has no `->` or `<->`")
    try:
        head = ast.parse(f"def {left.strip()}: pass").body[0]
        body = ast.parse(right.strip(), mode="eval").body
    except SyntaxError as exc:
        raise RegistryError(f"template {text!r} does not parse: {exc.msg}") from exc
    if not isinstance(head, ast.FunctionDef) or head.name != name:
        raise RegistryError(f"template {text!r} does not rewrite {name!r}")
    args = head.args
    params = {a.arg for a in [*args.posonlyargs, *args.args, *args.kwonlyargs]}
    params |= {a.arg for a in (args.vararg, args.kwarg) if a is not None}
    comprehended = {
        t.id
        for node in ast.walk(body)
        if isinstance(node, ast.comprehension)
        for t in ast.walk(node.target)
        if isinstance(t, ast.Name)
    }
    for node in ast.walk(body):
        if isinstance(node, ast.Name) and node.id not in params | comprehended | _RESERVED:
            raise RegistryError(f"template {text!r} uses unknown name {node.id!r}")
    if reversible:
        _check_reversible(text, body, params)
    return Template(head.name, args, right.strip(), reversible)


def _check_reversible(text: str, body: ast.expr, params: set[str]) -> None:
    root = body.func if isinstance(body, ast.Call) else None
    while isinstance(root, ast.Attribute):
        root = root.value
    if not (isinstance(body, ast.Call) and isinstance(root, ast.Name)):
        raise RegistryError(f"reversible template {text!r} must be one call on self or bt")
    if root.id not in ("self", "bt"):
        raise RegistryError(f"reversible template {text!r} must be one call on self or bt")
    values = [a.value if isinstance(a, ast.Starred) else a for a in body.args]
    values += [k.value for k in body.keywords]
    for value in values:
        if not (isinstance(value, ast.Name) and value.id in params) and not isinstance(
            value, ast.Constant
        ):
            raise RegistryError(f"reversible template {text!r} may only re-bind parameters")
