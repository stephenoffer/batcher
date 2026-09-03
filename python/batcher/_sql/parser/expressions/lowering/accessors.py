"""SQL → the typed accessor namespaces, by name.

The `Expr` accessors (`.str`, `.dt`, `.list`, `.struct`, `.json`, `.map`, `.image`,
`.audio`, `.video`, `.seq`) are 470 operations of the public expression surface, and SQL
reached 162 of them. The rest were not missing from the engine -- ``col("s").str.slugify()``,
``col("i").image.phash()``, ``col("a").audio.mfcc(...)`` all answer -- they were missing a
*name* on the SQL side, because that side is a curated table per family and a table only
grows when someone remembers to grow it.

This is the same derivation `families.py` performs for the free-function library, applied
to the namespaces: the accessor surface *is* the vocabulary, and the SQL name is the
namespace and the method with the underscores removed (``str_slugify``, ``image_phash``,
``dt_quarter_end``, ``list_softmax``). A method is reachable from SQL the moment it exists,
and `tests/unit/test_sql_accessor_vocabulary.py` walks the namespaces to keep that so.

**Only the namespace-qualified spelling is registered**, deliberately. A bare ``slugify``
would read better, but three of these method names are defined by more than one namespace,
and -- more to the point -- a bare name that another SQL engine already means something
else by would start *answering* where it used to raise. Qualified names cannot do that to
anyone.

**It runs last**, after every typed node, every curated family table, and the free-function
library, so a name SQL already means keeps its SQL meaning and this can never shadow one.

**What a parameter means is decided by `signatures.py`**, which both derived dispatches
share. A parameter annotated `int`/`str`/`float`/`bool` is a plan-time constant read off
the literal node -- not a guess, since it cannot take a column in any front-end -- and an
argument that is not a literal is refused by name rather than mistranslated. Declining
those instead would leave the whole of `.image`, `.audio` and `.video` unreachable, since
every one of their methods is parameterized that way.
"""

from __future__ import annotations

import inspect
from typing import Any

from sqlglot import expressions as exp

from batcher._sql.parser.expressions.lowering.signatures import (
    STRINGS,
    arity,
    build_arguments,
    hints,
    parameter_kinds,
    union_members,
    written_name,
)
from batcher.plan.expr_ir import Expr

__all__ = ["accessor_function", "accessor_namespaces", "accessor_vocabulary"]

#: Marks a property on `Expr` as an accessor namespace: its getter is annotated with the
#: namespace class, and every one of those is named `_<Name>Namespace`.
_NAMESPACE_SUFFIX = "Namespace"


def accessor_namespaces() -> tuple[str, ...]:
    """Every accessor namespace on `Expr`, in declaration order.

    Read off `Expr` rather than listed. A hardcoded tuple is the same failure this module
    exists to fix, one level up: it was written with the nine namespaces `CLAUDE.md` names
    and silently omitted `.seq`, whose 22 bioinformatics methods stayed unreachable from
    SQL while the vocabulary reported itself complete.

    Returns:
        The attribute name of each namespace, such as ``"str"`` or ``"seq"``.
    """
    from batcher.plan.expr_ir.core import Expr

    return tuple(
        name
        for name, value in vars(Expr).items()
        if isinstance(value, property)
        and str(getattr(value.fget, "__annotations__", {}).get("return", "")).endswith(
            _NAMESPACE_SUFFIX
        )
    )


#: Accessors whose SQL name a handler outside `anonymous.py` curates, which
#: `known_names` therefore cannot see. Excluded for the reason it excludes the names it
#: can: a curated handler that serves some arities and not others would leave this one
#: serving the rest, and one name would mean two things depending on how it was written.
#: Each is reachable, and reaches the *same* accessor -- pinned by
#: `tests/unit/test_sql_accessor_vocabulary.py::test_the_curated_names_reach_the_accessor`:
#:
#: * the five `.json` readers take a JSON *path* (``json_keys(j, '$.a')``), which the
#:   curated handler normalizes from a bare key;
#: * ``list_filter``/``list_transform`` take a lambda (``list_filter(l, x -> x > 0)``);
#: * ``list_unique`` is DuckDB's *count* of distinct elements, and the list itself is
#:   ``list_distinct``;
#: * ``.str.join`` is an aggregate, spelled ``string_agg``.
_CURATED_ELSEWHERE = frozenset(
    {
        "jsonarraylength",
        "jsonexists",
        "jsonextractstring",
        "jsonkeys",
        "jsonvalue",
        "listfilter",
        "listtransform",
        "listunique",
        "strjoin",
    }
)

_VOCABULARY: dict[str, tuple[str, str, Any]] | None = None


def accessor_vocabulary() -> dict[str, tuple[str, str, Any]]:
    """Normalized SQL name → (namespace, method, function), over every accessor namespace.

    Built once, lazily. A method whose parameters or result have no SQL shape is left out
    (see `_parameters`), so what this returns is exactly what `accessor_function` serves --
    which is what stops the dispatcher and the test that walks these namespaces from
    disagreeing about the vocabulary.

    Returns:
        The SQL name, lowercased with underscores removed, mapped to what it calls.
    """
    global _VOCABULARY
    if _VOCABULARY is None:
        from batcher._sql.parser.expressions.anonymous import known_names
        from batcher.plan.expr_ir import col

        claimed = known_names()
        probe = col("__bc_probe")
        built: dict[str, tuple[str, str, Any]] = {}
        for namespace in accessor_namespaces():
            cls = type(getattr(probe, namespace))
            for method in sorted(n for n in dir(cls) if not n.startswith("_")):
                fn = getattr(cls, method, None)
                if not callable(fn) or inspect.isclass(fn):
                    continue
                returns = hints(fn).get("return")
                if returns is None or not all(
                    isinstance(m, type) and issubclass(m, Expr) for m in union_members(returns)
                ):
                    continue
                if parameter_kinds(fn, skip_first=True) is None:
                    continue
                key = f"{namespace}{method}".replace("_", "")
                if key in claimed or key in _CURATED_ELSEWHERE:
                    # A name another handler curates. Declining it here rather than only
                    # at runtime is what keeps a name from meaning two things by arity:
                    # `anonymous.py` serves `list_slice` at three arguments with DuckDB's
                    # inclusive 1-based bounds, and this would otherwise have served the
                    # two-argument form with `.list.slice`'s 0-based offset.
                    continue
                built.setdefault(key, (namespace, method, fn))
        _VOCABULARY = built
    return _VOCABULARY


def _call(node) -> tuple[str, list] | None:
    """The vocabulary key and argument nodes of `node`, or None if it names no call.

    sqlglot promotes a handful of these names to *typed* nodes rather than leaving them
    `Anonymous` -- ``str_to_date(s, fmt)`` parses as `StrToDate`, whose operands sit in
    `this`/`expression`. Reading only `Anonymous` left those few raising "unsupported SQL
    expression: StrToDate" beside every other accessor working, which is the same trap
    `families.py` had to step around for `ST_Point`. Reaching them here is safe for the
    same reason it is safe there: this runs after every typed handler has declined, so it
    can only turn an error into an answer.
    """
    if isinstance(node, exp.Anonymous):
        return str(node.name).lower().replace("_", ""), list(node.expressions)
    if isinstance(node, exp.Func):
        # In the class's own `arg_types` order, which is the order the function is written
        # in. Reading `this`/`expression`/`expressions` instead silently *drops* an operand
        # sqlglot files under its own name: `StrToDate` keeps the format under `format`, so
        # `str_to_date(s, '%Y')` reached `.str.to_date()` with no format at all and parsed
        # against the default -- an answer, and the wrong one.
        args = []
        for slot in type(node).arg_types:
            value = node.args.get(slot)
            if isinstance(value, list):
                args += [a for a in value if a is not None]
            elif value is not None:
                args.append(value)
        return type(node).__name__.lower(), args
    return None


def accessor_function(tr, node) -> Expr | None:
    """Build the accessor-namespace expression for `node`, or None if it is not one.

    Args:
        tr: The translator, used to lower the subject and any column argument.
        node: The sqlglot node to dispatch.

    Returns:
        The Batcher expression, or None when `node` names no accessor method.

    Raises:
        NotImplementedError: If the name is an accessor method called with the wrong number
            of arguments, or with a column where the method takes a plan-time constant. A
            silent miss would fall through to "unknown function", which names the wrong
            problem entirely.
    """
    call = _call(node)
    if call is None:
        return None
    key, args = call
    entry = accessor_vocabulary().get(key)
    if entry is None:
        return None
    namespace, method, fn = entry
    kinds = parameter_kinds(fn, skip_first=True)
    assert kinds is not None  # a method with unclassifiable parameters is not in the vocabulary
    required, total = arity(fn, skip_first=True)
    if kinds and kinds[-1] is STRINGS:
        total = max(total, len(args) - 1)  # the tail takes as many strings as are written
    if not (required + 1 <= len(args) <= total + 1):
        expected = required + 1 if required == total else f"{required + 1} to {total + 1}"
        raise NotImplementedError(
            f"{written_name(node)}() takes {expected} argument(s) — the {namespace} value "
            f"and {required if required == total else f'{required} to {total}'} more — "
            f"got {len(args)}"
        )
    subject = tr._scalar(args[0])
    built = build_arguments(tr, fn, list(args[1:]), written_name(node), skip_first=True)
    return getattr(getattr(subject, namespace), method)(*built)
