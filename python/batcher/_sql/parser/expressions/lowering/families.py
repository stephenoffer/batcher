"""SQL → the public function library, by name.

Batcher's function library is a `Dataset` API *and* a SQL front end over one engine, and the
two saw different halves of it. `bc-geo`'s 113 geospatial functions, `bc-spatial`'s 42
rigid-body ones, and the ~200 text-quality, evaluation and statistics functions were all
public — `bt.st_area(...)`, `bt.quat_angle(...)`, `bt.pii_rate(...)` all answered — and
**none of them was reachable from SQL**: ``SELECT ST_Area(g)`` raised "unknown function
'ST_Area'". For the geospatial family that is not even Batcher's name for the operation, it
is *the* name, the one PostGIS, DuckDB spatial, Snowflake, BigQuery and the OGC standard all
use, and spatial work is written in SQL far more often than in a DataFrame.

The dispatch is **derived**: `plan.functions`' own public exports are the vocabulary, and the
SQL name is the Python name with the underscores removed — which is also what makes
`ST_AsText` and `st_as_text` agree. A table would be a copy of that surface and could drift
from it, which is exactly how the window vocabulary lost six functions; here a function is
reachable from SQL the moment it is exported, and the tests walk the exports to keep that so.

**It runs last**, after every typed node and every named family, so a name SQL already means
keeps its SQL meaning and this can never shadow one. That ordering is also what makes the
derivation safe to widen: a handler added later automatically wins over it.

Three shapes are excluded rather than guessed at, by `is_scalar_callable`: a grouped
parameter, a grouped return, and a Python-literal parameter. See its docstring.

**Aggregates are excluded too**, and at call time rather than by signature, because many are
expressions *over* aggregates and only building one reveals it. An aggregate needs the
translator to know that a projection reduces, so it can build the `GROUP BY` around it;
reached through the scalar path it answers correctly with no grouping and then fails under
`GROUP BY` with an internal error about a window column. Half a feature with a confusing
error is worse than a clear "not served here", so they raise and say where to go instead.

Layer note: this belongs beside `maps.py`, `strings.py` and `temporal.py` in `expressions/`,
which is at its twelve-file limit. It sits here rather than growing that directory past the
structural gate.
"""

from __future__ import annotations

import inspect
from typing import Any

from sqlglot import expressions as exp

from batcher._sql.parser.expressions.lowering.signatures import build_arguments, parameter_kinds
from batcher.plan.expr_ir import Expr
from batcher.plan.expr_ir.walk import contains_aggregate

__all__ = ["family_function", "is_scalar_callable", "library_aggregate", "positional_arity"]

#: PostGIS / DuckDB-spatial spellings that normalize to a *different* word than Batcher's, so
#: the underscore rule cannot reach them: `ST_NPoints` is `st_num_points`, not `st_npoints`.
#: Each pair is verified to return the identical answer, column for column, against DuckDB's
#: spatial extension -- a name that is only *nearly* the same function would answer a
#: different question, which is the failure an alias table invites.
_ALIASES: dict[str, str] = {
    "stnpoints": "stnumpoints",
    "stngeometries": "stnumgeometries",
    "stninteriorrings": "stnuminteriorrings",
    "staswkb": "stasbinary",
    "staswkt": "stastext",
    "stmakepoint": "stpoint",
}

#: Trailing arguments the SQL standard makes optional where Batcher's Python signature does
#: not. Only where a *published* default exists, and only that one: PostGIS documents
#: `ST_Buffer(geom, radius, num_seg_quarter_circle = 8)`, and DuckDB spatial's two-argument
#: form is verified to equal its three-argument form with 8. Guessing a default for a
#: function that has none published would answer a different question than the caller asked,
#: so every other arity mismatch raises and names the arity instead.
_STANDARD_DEFAULTS: dict[str, Any] = {"stbuffer": 8}

#: A value per literal parameter kind for the aggregate probe below. Deliberately the
#: emptiest of each type: the probe only has to *build*, and a value with structure (a real
#: regex, a real format) would make the classification depend on the value.
_PROBE_CONSTANTS: dict[type, object] = {str: "", int: 0, float: 0.0, bool: False}

_REGISTRY: dict[str, Any] | None = None


def _registry() -> dict[str, Any]:
    """Normalized SQL name → builder, over the whole public function library.

    Built once, lazily, and only reached after every other handler has declined — so a name
    SQL already means keeps its SQL meaning, and this cannot shadow one. That ordering is
    also what makes the derivation safe to widen: a handler added later automatically wins.
    """
    global _REGISTRY
    if _REGISTRY is None:
        import importlib
        import pkgutil

        import batcher.plan.functions as library

        built: dict[str, Any] = {}
        modules = [library]
        for found in pkgutil.walk_packages(library.__path__, library.__name__ + "."):
            try:
                modules.append(importlib.import_module(found.name))
            except Exception:  # pragma: no cover - an optional dependency is absent
                continue
        for module in modules:
            for member in getattr(module, "__all__", ()) or ():
                fn = getattr(module, member, None)
                if callable(fn) and not inspect.isclass(fn):
                    # The *whole* vocabulary, callable or not. A name that is in the library
                    # but has no scalar SQL spelling gets an explanation naming the shape;
                    # filtering it out here would send it to "unknown function", which is
                    # false — the function plainly exists — and is the less useful of the
                    # two errors.
                    built.setdefault(member.replace("_", ""), fn)
        _REGISTRY = built
    return _REGISTRY


def positional_arity(fn) -> tuple[int, int | None]:
    """(required, maximum) positional parameters — maximum None when the function is variadic.

    `greatest(*columns)` takes any number, and counting only the *named* positional
    parameters said it took none, so every call was refused as "takes 0 argument(s)".
    """
    signature = inspect.signature(fn)
    params = [
        p
        for p in signature.parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    required = sum(1 for p in params if p.default is p.empty)
    variadic = any(p.kind is p.VAR_POSITIONAL for p in signature.parameters.values())
    return required, None if variadic else len(params)


#: Parameter annotations that stand for a *group* of columns rather than one.
_GROUPED_TYPES = frozenset({"point", "quaternion", "pose"})


def _grouped_parameters(fn) -> list[str]:
    """Parameters annotated as a multi-column value (`Point`, `Quaternion`, `Pose`)."""
    return [
        f"{p.name}: {p.annotation}".replace("'", "")
        for p in inspect.signature(fn).parameters.values()
        if str(p.annotation).strip("'\"").lower() in _GROUPED_TYPES
    ]


def is_scalar_callable(fn) -> bool:
    """Whether `fn` is a one-column-in, one-column-out function SQL can call.

    Two shapes are not. A **grouped parameter** (`distance_3d(a: Point, b: Point)`) wants a
    tuple of columns where SQL has only scalars, and a **grouped return**
    (`quat_multiply -> dict[str, Expr]`) answers a whole rotation where a SQL expression is
    one value. Both have per-component spellings that *are* callable.

    A **Python-literal parameter** (a bare `str`/`int`) used to be a third, on the reasoning
    that reading the constant off the literal node is a per-function decision. It is not:
    `signatures.parameter_kinds` makes it mechanically, the accessor dispatch has been
    making it since it existed, and an argument that is not a literal is refused by name.
    Treating it as a veto cost **49 public functions their SQL spelling** -- `hmac_sha256`,
    `aes_encrypt`, `great_circle_distance`, `render_template` and the text-quality metrics
    were all `bt.`-callable and unknown to `SELECT`.

    Deciding this here rather than at each call site is what stops the dispatcher and the
    test that walks these families from disagreeing about which names are in scope — and
    the arity check alone cannot decide it: `quat_multiply(a, b)` takes exactly two
    arguments, so two scalars pass the count and then fail inside the function with a bare
    `TypeError: a batcher expression is not iterable`.

    Args:
        fn: A member of one of the families.

    Returns:
        True when the function takes single columns or plan-time constants and returns one
        column.
    """
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):  # pragma: no cover - a builtin without a signature
        return False
    if _grouped_parameters(fn) or parameter_kinds(fn) is None:
        return False
    returns = str(signature.return_annotation)
    return "dict" not in returns and "tuple" not in returns


def _call_arguments(node) -> tuple[str, list] | None:
    """The normalized name and argument nodes of `node`, or None if it is not a call."""
    kind = type(node).__name__.lower()
    if isinstance(node, exp.Anonymous):
        return str(node.name).lower().replace("_", ""), list(node.expressions)
    if kind.startswith(("st", "quat", "se3")):
        # sqlglot promotes a handful (`ST_Point`, `ST_Distance`) to *typed* nodes, whose
        # class name carries the function and whose operands sit in `this`/`expression`.
        # Without this branch those few failed with "unsupported SQL expression: StPoint"
        # beside every other spelling of the same family working.
        args = [a for a in (node.this, node.args.get("expression")) if a is not None]
        args += [a for a in node.args.get("expressions") or [] if a is not None]
        return kind, args
    return None


def family_function(tr, node) -> Expr | None:
    """Build the geospatial / spatial expression for `node`, or None if it is not one.

    Args:
        tr: The translator, used to lower each argument.
        node: The sqlglot node to dispatch.

    Returns:
        The Batcher expression, or None when `node` is not a call this serves.

    Raises:
        NotImplementedError: If the name is one of these functions called with the wrong
            number of arguments, or one whose result is several columns rather than one.
            A silent miss would fall through to "unknown function", which names the wrong
            problem entirely.
    """
    call = _call_arguments(node)
    if call is None:
        return None
    key, args = call
    fn = _registry().get(_ALIASES.get(key, key))
    if fn is None:
        return None
    if not is_scalar_callable(fn):
        raise NotImplementedError(
            f"{node.name or key}() takes or returns a group of columns, or an argument that "
            f"is a Python value rather than a column, so it has no scalar SQL spelling. Use "
            f"the per-component functions where there are any "
            f"({node.name or key}_x, _y, _z, _w), or the DataFrame API"
        )
    required, total = positional_arity(fn)
    built = build_arguments(tr, fn, args, node.name or key)
    if len(built) == required - 1 and key in _STANDARD_DEFAULTS:
        built.append(_STANDARD_DEFAULTS[key])
    if len(built) < required or (total is not None and len(built) > total):
        if total is None:
            expected = f"at least {required}"
        else:
            expected = str(required) if required == total else f"{required} to {total}"
        raise NotImplementedError(
            f"{node.name or key}() takes {expected} argument(s), got {len(built)}"
        )
    result = fn(*built)
    if contains_aggregate(result):
        # An aggregate needs the *aggregate* dispatch — the translator has to know a
        # projection reduces, so it can build the GROUP BY around it. Reached through the
        # scalar path it answers correctly with no grouping and then fails under `GROUP BY`
        # with "window function '__bt_win_0' references unknown column", which is a worse
        # outcome than not serving it: a half-working function with an internal error.
        raise NotImplementedError(
            f"{node.name or key}() is an aggregate, and only the scalar function library is "
            f"reachable from SQL by name. Use the DataFrame API "
            f"(`ds.group_by(...).agg(v=bt.{fn.__name__}(...))`) for it"
        )
    return result


#: Normalized name → whether the builder produces an aggregate. Cached because deciding it
#: means *building* the expression: 175 of these are annotated `-> Expr` and are aggregates
#: anyway, being expressions *over* aggregate leaves the way `regr_r2` and `sem` are, so the
#: signature cannot answer it and only `contains_aggregate` on a built tree can.
_IS_AGGREGATE: dict[str, bool] = {}


def library_aggregate(name: str):
    """The builder for `name` when it is a library *aggregate*, else None.

    The SQL translator has to know a projection reduces *before* it builds the query, so it
    can put the `GROUP BY` around it — which is why this is a name lookup rather than
    something decided from the built expression. Classification is done once per name, by
    building the function over placeholder columns and asking whether the result contains an
    aggregate; a builder that raises on placeholders is not one SQL can call anyway.

    Args:
        name: The SQL function name as written, in any case.

    Returns:
        The builder, or None when the name is not a library aggregate.
    """
    key = name.lower().replace("_", "")
    fn = _registry().get(_ALIASES.get(key, key))
    if fn is None or not is_scalar_callable(fn):
        return None
    known = _IS_AGGREGATE.get(key)
    if known is None:
        known = _IS_AGGREGATE[key] = _builds_an_aggregate(fn)
    return fn if known else None


def _builds_an_aggregate(fn) -> bool:
    """Whether `fn` over placeholder columns yields something containing an aggregate."""
    import warnings

    from batcher.plan.expr_ir import col

    required = [
        p
        for p in inspect.signature(fn).parameters.values()
        if p.default is p.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    # One probe value per parameter *kind*: a column where the function wants a column, and
    # a constant where it wants one. Probing a constant parameter with a column made every
    # such function raise, which read as "not an aggregate" -- a wrong answer arrived at by
    # a broken probe, and the direction that hides an aggregate rather than inventing one.
    kinds = parameter_kinds(fn) or []
    probes = [
        col("__bc_probe")
        if index >= len(kinds) or kinds[index] is Expr
        else _PROBE_CONSTANTS.get(kinds[index], 0)
        for index in range(len(required))
    ]
    try:
        with warnings.catch_warnings():
            # A couple of builders warn when handed a literal key (`aes_encrypt`); this is
            # a classification probe, not a query, so the warning has no audience.
            warnings.simplefilter("ignore")
            built = fn(*probes)
    except Exception:
        return False
    return contains_aggregate(built)
