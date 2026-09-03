"""What a Python signature means to the SQL translator.

The two *derived* dispatches -- `families` over the public function library and
`accessors` over the typed `Expr` namespaces -- both have to answer the same question
about a Python callable: which of its parameters is a **column** (so a SQL argument is
lowered to an `Expr`), which is a **plan-time constant** the engine reads while the plan is
built (so the SQL argument has to be a literal, read off the node), and which is neither
(so the function has no SQL spelling at all and is left out rather than mistranslated).

They answered it separately, and differently: `accessors` read the constant off the literal
node, `families` declined every function that had one. That cost 49 public functions their
SQL spelling for no reason either dispatch could state -- `hmac_sha256`, `aes_encrypt`,
`great_circle_distance`, `render_template` and the text-quality metrics were all callable
as `bt.hmac_sha256(...)` and unknown to `SELECT`. One classifier, used by both, is what
keeps that from happening again in either direction.

Resolving an annotation is more work than it looks, and the module docstrings of the two
callers used to carry half the reasons each. See `hints` and `union_members`.
"""

from __future__ import annotations

import contextlib
import inspect
import types
import typing
from collections.abc import Iterable
from typing import Any, ForwardRef

from batcher._sql.parser.expressions.lowering.dynamic import (
    const_bool,
    const_float,
    const_int,
    const_str,
)
from batcher.plan.expr_ir import Expr

__all__ = [
    "LITERAL_READERS",
    "STRINGS",
    "arity",
    "build_arguments",
    "hints",
    "node_classes",
    "parameter_kinds",
    "read_argument",
    "union_members",
    "written_name",
]

#: Reader per literal parameter type. `const_int` and `const_str` are the parser's existing
#: "the Python value this node denotes" helpers; the other two are their siblings.
LITERAL_READERS: dict[type, Any] = {
    str: const_str,
    bool: const_bool,
    int: const_int,
    float: const_float,
}

#: The kind of a parameter that takes *several* strings (`contains_any(patterns)`). SQL
#: spells a set of patterns as trailing arguments, so it consumes the rest of the call.
STRINGS = "strings"


def union_members(annotation: Any) -> list[Any]:
    """The non-None members of `annotation`, which is one type or a union of them.

    A member left as a `ForwardRef` is resolved against the package's node classes.

    `IntoExpr` is spelled ``Union["Expr", int, float, bool, str]``, and where the alias
    reaches this through a generated accessor's `__signature__` the string is still a
    forward reference -- so `.list.set_union` and eleven of its neighbours read as
    "literal-only" and were left out, while `.list.transform` beside them resolved and was
    served. Two spellings of one annotation must not decide whether SQL can call a method.
    """
    origin = typing.get_origin(annotation)
    members = (
        [a for a in typing.get_args(annotation) if a is not type(None)]
        if origin is typing.Union or origin is types.UnionType
        else [annotation]
    )
    return [
        node_classes().get(m.__forward_arg__, m) if isinstance(m, ForwardRef) else m
        for m in members
    ]


def hints(fn: Any) -> dict[str, Any]:
    """`fn`'s annotations as types, from `typing` first and the signature second.

    `typing.get_type_hints` is the right answer and answers most of these, but it cannot
    answer all of them and it fails in two different ways. Most of `.list` and half of
    `.dt` are **generated** accessors (`namespaces/_bind.py`) whose closure carries an
    empty `__annotations__` and a `__signature__` holding the resolved classes -- so
    `get_type_hints` returns ``{}`` rather than raising. A handful more are written out but
    name a node class their module does not import, so resolving the *string* raises
    `NameError` while the signature again holds the class.

    Reading the signature only when `get_type_hints` *raised* therefore missed methods
    silently, which is the whole failure this vocabulary exists to stop. Both sources are
    consulted for every name, `typing` winning where it answers. Measured over the accessor
    namespaces: `typing` alone types 378 of the 470 returns and the signature alone types
    92 more, so each is necessary and neither is sufficient.

    Args:
        fn: The accessor method to type.

    Returns:
        Parameter names (and ``"return"``) mapped to the type each is annotated with;
        a name neither source can resolve is absent rather than guessed at.
    """
    resolved: dict[str, Any] = {}
    with contextlib.suppress(Exception):
        resolved.update(typing.get_type_hints(fn))
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):  # pragma: no cover - a builtin without a signature
        return resolved
    annotations = [(p.name, p.annotation) for p in signature.parameters.values()]
    annotations.append(("return", signature.return_annotation))
    for name, annotation in annotations:
        if name in resolved or annotation is inspect.Parameter.empty:
            continue
        if isinstance(annotation, str):
            scope = {**node_classes(), **getattr(fn, "__globals__", {})}
            try:
                annotation = eval(annotation, scope, {})
            except Exception:
                continue
        resolved[name] = annotation
    return resolved


_NODE_CLASSES: dict[str, type] | None = None


def node_classes() -> dict[str, type]:
    """Every `Expr` node class in `plan.expr_ir`, by name, for resolving an annotation.

    An accessor written as an alias annotates its result with the node class it builds
    (``-> "StrFunc"``, ``-> "ListGet"``) in a module that does not import that class at
    module scope -- the annotation is never evaluated at runtime, so nothing made it wrong.
    It is evaluated *here*, so the names have to come from somewhere, and the package's own
    submodules are the one place that is guaranteed to have all of them.
    """
    global _NODE_CLASSES
    if _NODE_CLASSES is None:
        import importlib
        import pkgutil

        import batcher.plan.expr_ir as ir

        found: dict[str, type] = {}
        modules = [ir]
        for module in pkgutil.walk_packages(ir.__path__, ir.__name__ + "."):
            try:
                modules.append(importlib.import_module(module.name))
            except Exception:  # pragma: no cover - an optional dependency is absent
                continue
        for module in modules:
            for name, value in vars(module).items():
                if isinstance(value, type) and issubclass(value, Expr):
                    found.setdefault(name, value)
        _NODE_CLASSES = found
    return _NODE_CLASSES


def parameter_kinds(fn: Any, *, skip_first: bool = False) -> list[Any] | None:
    """The kind of each positional parameter, or None if any one of them is unclear.

    A parameter is either a **column** (annotated with an `Expr` type, so the SQL argument
    is lowered normally), a **plan-time constant** (`str`/`int`/`float`/`bool`, read off
    the literal node), or a trailing set of strings. Anything else -- a callable, a bare
    `Any`, a multi-column `Point` -- has no SQL argument shape, so the whole function is
    left out of the vocabulary rather than mistranslated.

    Args:
        fn: The callable to classify.
        skip_first: Drop the first parameter, for an unbound accessor method whose `self`
            is the SQL call's subject rather than an argument.

    Returns:
        One kind per positional parameter, or None when the function has no SQL shape.
    """
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):  # pragma: no cover - a builtin without a signature
        return None
    annotations = hints(fn)
    kinds: list[Any] = []
    positional = (
        list(signature.parameters.values())[1:]
        if skip_first
        else list(signature.parameters.values())
    )
    last = positional[-1] if positional else None
    for parameter in positional:
        if parameter.kind is parameter.KEYWORD_ONLY and parameter.default is not parameter.empty:
            # An optional keyword-only tuning parameter (`resize(..., *, format="png")`).
            # SQL cannot name it, so the call takes its default -- which is what leaving the
            # method out entirely would deny, and it is 22 of `.image` alone.
            continue
        if parameter.kind is parameter.VAR_POSITIONAL:
            # `max_horizontal(*columns)` takes any number of columns, and every one of them
            # is a column. Stopping here (rather than declining the function) leaves the
            # tail unclassified, which `build_arguments` reads as `Expr` — the right answer,
            # and the one that keeps the eleven `*_horizontal` functions callable.
            break
        if parameter.kind is parameter.VAR_KEYWORD:
            continue  # `**options`: SQL cannot name one, so it takes none
        if parameter.kind is not parameter.POSITIONAL_ONLY and (
            parameter.kind is not parameter.POSITIONAL_OR_KEYWORD
        ):
            return None  # a keyword-only parameter with no default
        annotation = annotations.get(parameter.name)
        if annotation is None:
            return None
        members = union_members(annotation)
        if any(isinstance(m, type) and issubclass(m, Expr) for m in members):
            kinds.append(Expr)
        elif members and all(m in LITERAL_READERS for m in members):
            # `int | float` reads as the widest member that accepts the narrower one.
            kinds.append(float if float in members else members[0])
        elif _is_string_iterable(annotation) and parameter is last:
            kinds.append(STRINGS)
        else:
            return None
    return kinds


def _is_string_iterable(annotation: Any) -> bool:
    """Whether `annotation` is an iterable of strings (`Iterable[str]`, `Sequence[str]`).

    `contains_any(patterns)` takes a set of patterns, which SQL writes as trailing
    arguments -- ``str_contains_any(s, 'a', 'b')``. Only as the *last* parameter, where
    "the rest of the call" is unambiguous.
    """
    origin = typing.get_origin(annotation)
    if origin is None or not isinstance(origin, type) or not issubclass(origin, Iterable):
        return False
    return typing.get_args(annotation) == (str,)


def arity(fn: Any, *, skip_first: bool = False) -> tuple[int, int]:
    """(required, total) positional parameters, optionally past a bound `self`.

    Args:
        fn: The callable to measure.
        skip_first: Drop the first parameter, as `parameter_kinds` does.

    Returns:
        How many positional parameters the call requires, and how many it accepts.
        Keyword-only parameters are excluded; they take their defaults, which is what
        `parameter_kinds` decided when it left them out of the kinds.
    """
    values = list(inspect.signature(fn).parameters.values())
    parameters = [
        p
        for p in (values[1:] if skip_first else values)
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    return sum(1 for p in parameters if p.default is p.empty), len(parameters)


def read_argument(kind: Any, node, written: str) -> Any:
    """The Python value a SQL argument node contributes for a parameter of `kind`.

    Args:
        kind: `Expr` for a column, a literal type for a plan-time constant.
        node: The sqlglot argument node. A column argument arrives already lowered.
        written: How the call was written, for the error message.

    Returns:
        The value to pass to the Python function.

    Raises:
        NotImplementedError: If a constant parameter was given something that is not a
            literal of its type. Naming the parameter is the whole value here: the
            alternative is a `TypeError` from inside a function the caller never named.
    """
    value = LITERAL_READERS[kind](node)
    if value is None:
        raise NotImplementedError(
            f"{written}() takes a constant {kind.__name__} where {node.sql()} was given; "
            f"the engine reads that parameter when the plan is built, so it cannot be a "
            f"column"
        )
    return value


def written_name(node) -> str:
    """How a call was written, for an error message; a typed node has no `name`."""
    return str(node.name) or type(node).__name__


def build_arguments(tr, fn, args: list, written: str, *, skip_first: bool = False) -> list[Any]:
    """Lower a SQL call's argument nodes into what `fn` takes, parameter by parameter.

    The three dispatches that call a Python function by name -- the function library, its
    aggregates, and the accessor namespaces -- each used to spell this as
    ``[tr._scalar(a) for a in args]``, which is right only while every parameter is a
    column. It is not: `hmac_sha256(text, key)` wants the key as a Python string, and
    handing it a lowered `Lit` fails inside the function with an error naming an internal
    node (``str.chunk(): size must be an integer, got Lit lit(3)``) rather than the
    argument the caller wrote.

    Args:
        tr: The translator, for lowering a column argument.
        fn: The function being called.
        args: The call's argument nodes, in order.
        written: How the call was written, for the error messages.
        skip_first: The function's first parameter is a bound `self`, so `args` lines up
            with the parameters after it.

    Returns:
        One value per argument: an `Expr` for a column parameter, a Python constant for a
        plan-time one, and a list of strings for a trailing set-of-strings parameter.

    Raises:
        NotImplementedError: If a constant parameter was given something that is not a
            literal of its type.
    """
    kinds = parameter_kinds(fn, skip_first=skip_first) or []
    built: list[Any] = []
    for index, node in enumerate(args):
        kind = kinds[index] if index < len(kinds) else Expr
        if kind is STRINGS:
            collected = [const_str(a) for a in args[index:]]
            if any(text is None for text in collected):
                raise NotImplementedError(f"{written}() takes constant strings there")
            built.append(collected)
            break
        built.append(tr._scalar(node) if kind is Expr else read_argument(kind, node, written))
    return built
