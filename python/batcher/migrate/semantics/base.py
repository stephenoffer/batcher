"""The transform registry, the context a transform reads, and the node helpers they share.

A transform is a function `fn(ctx, *values)` registered with `@transform`. Each value is a
`templates.Bound` (the rewritten node plus the original the inference can type), a list of them
for a template's `*args`, or a dict for its `**kwargs`. A transform returns a replacement node,
a list of `cst.Arg` to splice into the enclosing call, or `None` to decline; raising
`templates.Declined` declines too.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, Protocol

from batcher._internal.optional import require
from batcher.migrate.templates import Bound, Declined, literal, simple_call

cst = require("libcst", feature="batcher.migrate", provides="libcst", extra="migrate")

__all__ = ["TRANSFORMS", "Context", "lookup", "transform"]

Transform = Callable[..., Any]
TRANSFORMS: dict[str, Transform] = {}


class Context(Protocol):
    """What a transform may ask of the rewrite in progress."""

    engine: str
    bt: str

    def receiver(self, original: Any) -> str | None:
        """The foreign receiver an original node is."""

    def rewritten(self, original: Any) -> Any:
        """The rewritten form of an original subexpression."""

    def consume(self, original: Any) -> None:
        """Drop the markers an inner call recorded, because this rewrite absorbed it."""

    def definition(self, name: str) -> Any | None:
        """The one value a name is assigned in the file, or `None`."""

    def note(self, text: str) -> None:
        """Keep a marker on the call although it was rewritten; `""` means the row's note."""


def transform(fn: Transform) -> Transform:
    """Register a transform under its function name, as templates call it (`sem.<name>`).

    Args:
        fn: The transform.

    Returns:
        The same function.
    """
    TRANSFORMS[fn.__name__] = fn
    return fn


def lookup(ctx: Context) -> Callable[[str], Transform | None]:
    """A name-to-transform lookup bound to one site, for `templates.render`.

    Args:
        ctx: The site's context.

    Returns:
        The lookup; a transform that raises `Declined` returns `None`.
    """

    def find(name: str) -> Transform | None:
        fn = TRANSFORMS.get(name)
        if fn is None:
            return None

        def call(*args: Any) -> Any:
            try:
                return fn(ctx, *args)
            except Declined:
                return None

        return call

    return find


def expression_surfaces(ctx: Context) -> frozenset[str]:
    """The engine's surfaces whose values are column expressions.

    Args:
        ctx: The site's context.

    Returns:
        The surface names.
    """
    from batcher.migrate.engines import SPECS

    return SPECS[ctx.engine].expressions


def string(value: str) -> Any:
    """A double-quoted string literal node."""
    return cst.SimpleString(json.dumps(value))


def python(value: Any) -> str:
    """Python source for a literal, with double-quoted strings as hand-written code has."""
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, dict):
        return "{" + ", ".join(f"{python(k)}: {python(v)}" for k, v in value.items()) + "}"
    if isinstance(value, list):
        return "[" + ", ".join(python(v) for v in value) + "]"
    return repr(value)


def call(func: str, args: list[Any]) -> Any:
    """A call to a dotted name (`bt.col`)."""
    node: Any = None
    for part in func.split("."):
        node = cst.Name(part) if node is None else cst.Attribute(value=node, attr=cst.Name(part))
    return simple_call(node, args)


def keyword(name: str, value: Any) -> Any:
    """A `name=value` argument."""
    return cst.Arg(value, keyword=cst.Name(name))


def callee_name(func: Any) -> str | None:
    """The name a call's function is spelled with (`F.desc` and `desc` both give `desc`)."""
    if isinstance(func, cst.Attribute):
        return func.attr.value
    return func.value if isinstance(func, cst.Name) else None


def chain(node: Any, root: str, ctx: Context) -> list[tuple[Any, str]] | None:
    """The calls of a builder chain, innermost first, when it starts from `root`."""
    steps: list[tuple[Any, str]] = []
    while True:
        if not isinstance(node, cst.Call) and ctx.receiver(node) == root:
            return steps[::-1]
        if not (isinstance(node, cst.Call) and isinstance(node.func, cst.Attribute)):
            return None
        steps.append((node, node.func.attr.value))
        node = node.func.value


def flatten(values: list[Bound]) -> list[Bound]:
    """Values with literal lists and tuples spread into their elements."""
    out: list[Bound] = []
    for value in values:
        if not isinstance(value.node, (cst.List, cst.Tuple)):
            out.append(value)
            continue
        original = value.original
        items = original.elements if isinstance(original, (cst.List, cst.Tuple)) else None
        for i, element in enumerate(value.node.elements):
            out.append(Bound(element.value, items[i].value if items else None))
    return out


def bools(value: Bound, n: int) -> list[bool] | None:
    """A literal bool (repeated `n` times) or a literal list of `n` bools."""
    try:
        found = literal(value.node)
    except Declined:
        return None
    if isinstance(found, bool):
        return [found] * n
    if isinstance(found, list) and len(found) == n and all(isinstance(f, bool) for f in found):
        return found
    return None


def is_none(node: Any) -> bool:
    """Whether a node is the literal `None`."""
    return isinstance(node, cst.Name) and node.value == "None"


def bool_node(flag: bool) -> Any:
    """`True` or `False`."""
    return cst.Name("True" if flag else "False")


def bool_list(flags: list[bool]) -> Any:
    """A literal list of bools."""
    return cst.List([cst.Element(bool_node(f)) for f in flags])


def tuple_node(items: list[Any]) -> Any:
    """A literal tuple."""
    return cst.Tuple([cst.Element(i) for i in items])


def list_node(items: list[Any]) -> Any:
    """A literal list."""
    return cst.List([cst.Element(i) for i in items])
