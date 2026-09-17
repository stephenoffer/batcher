"""Apply one registry template, or check one call against a signature, on libcst nodes.

The template DSL is specified (and validated at load time) in
`batcher._internal.migration.schema`. This module is its evaluator: it binds a call's
arguments to the template's left side exactly as Python would, then renders the right side
with each parameter replaced by the argument node, `self` by the rewritten receiver and `bt`
by the Batcher module, calling `sem.<transform>` for the parts the DSL cannot say. Anything that
does not bind or that a transform declines makes the whole template decline, and the caller
leaves the call alone with a marker. Nothing is ever half-applied.

It also holds the signature check every non-template rewrite goes through (`Signature.accepts`):
a rewrite that would pass an argument the target does not take is refused rather than guessed.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import count
from typing import Any

from batcher._internal.migration import Template
from batcher._internal.optional import require

cst = require("libcst", feature="batcher.migrate", provides="libcst", extra="migrate")

__all__ = ["Bound", "Declined", "Signature", "literal", "parens", "render", "simple_call"]

_ATOMS = ("Name", "Attribute", "Call", "Subscript", "SimpleString", "Integer", "Float", "List")


class Declined(Exception):
    """A template or transform cannot rewrite this call exactly."""


@dataclass(frozen=True)
class Bound:
    """One value a template works with: the rewritten node and the original it came from.

    Attributes:
        node: The node as rewritten so far (inner calls already translated).
        original: The node in the unmodified script, which the inference can type; `None`
            for a node the template itself produced (a default, a literal).
        passed: Whether the call supplied it, rather than a parameter default.
    """

    node: Any
    original: Any = None
    passed: bool = True


def parens(node: Any) -> Any:
    """Parenthesize a node unless it is an atom, so it can be an attribute's base or operand.

    Args:
        node: A libcst expression.

    Returns:
        The node, parenthesized when needed.
    """
    if type(node).__name__ in _ATOMS or getattr(node, "lpar", None):
        return node
    return node.with_changes(lpar=[cst.LeftParen()], rpar=[cst.RightParen()])


def literal(node: Any) -> Any:
    """The Python value of a literal node (`"a"`, `3`, `True`, `["a", "b"]`).

    Args:
        node: A libcst expression, or `None`.

    Returns:
        The value.

    Raises:
        Declined: When the node is not a literal.
    """
    if node is None:
        raise Declined("no value")
    try:
        return ast.literal_eval(cst.Module([]).code_for_node(node))
    except (ValueError, SyntaxError) as exc:
        raise Declined("not a literal") from exc


def simple_call(func: Any, args: list[Any]) -> Any:
    """A call with `name=value` keywords spelled tightly, the way hand-written code is.

    Args:
        func: The callee node.
        args: `cst.Arg` nodes.

    Returns:
        The call node.
    """
    tight = cst.AssignEqual(
        whitespace_before=cst.SimpleWhitespace(""), whitespace_after=cst.SimpleWhitespace("")
    )
    comma = cst.Comma(whitespace_after=cst.SimpleWhitespace(" "))
    fixed = []
    for i, arg in enumerate(args):
        arg = arg.with_changes(comma=comma if i < len(args) - 1 else cst.MaybeSentinel.DEFAULT)
        if arg.keyword is not None:
            arg = arg.with_changes(equal=tight)
        fixed.append(arg)
    return cst.Call(func=func, args=fixed)


@dataclass(frozen=True)
class Signature:
    """A parameter list parsed from the generated tokens (`name`, `name=`, `*a`, `**k`, ...)."""

    positional: tuple[tuple[str, bool], ...] = ()
    posonly: int = 0
    keyword: tuple[tuple[str, bool], ...] = ()
    vararg: str | None = None
    kwarg: str | None = None
    columns: frozenset[str] = frozenset()
    is_property: bool = False

    @classmethod
    def parse(cls, tokens: list[str]) -> Signature:
        """Parse generated parameter tokens.

        Args:
            tokens: As written by `tools/parity/gen_codemod_tables.py`.

        Returns:
            The signature.
        """
        if tokens == ["@property"]:
            return cls(is_property=True)
        positional: list[tuple[str, bool]] = []
        keyword: list[tuple[str, bool]] = []
        posonly, vararg, kwarg, columns, kw_only = 0, None, None, set(), False
        for token in tokens:
            if token == "/":
                posonly = len(positional)
                continue
            if token == "*":
                kw_only = True
                continue
            name = token.lstrip("*").rstrip("=").rstrip("~")
            if token.rstrip("=").endswith("~"):
                columns.add(name)
            if token.startswith("**"):
                kwarg = name
            elif token.startswith("*"):
                vararg, kw_only = name, True
            else:
                (keyword if kw_only else positional).append((name, token.endswith("=")))
        return cls(tuple(positional), posonly, tuple(keyword), vararg, kwarg, frozenset(columns))

    def names(self) -> set[str]:
        """Every parameter a caller can pass by keyword.

        Returns:
            The names.
        """
        return {n for n, _ in self.positional[self.posonly :]} | {n for n, _ in self.keyword}

    def accepts(self, positional: int, keywords: list[str], source: Signature | None) -> bool:
        """Whether a call with this many positionals and these keywords binds here.

        A keyword that the *source* signature names as its own parameter only passes through
        this signature's `**kwargs` when it is not a named option there: a Polars
        `group_by(maintain_order=True)` must not become a Batcher named group key.

        Args:
            positional: The number of positional arguments.
            keywords: The keyword names passed.
            source: The signature the call was written against, when known.

        Returns:
            True when the call binds without guessing.
        """
        if self.is_property:
            return False
        if positional > len(self.positional) and self.vararg is None:
            return False
        named = self.names()
        source_named = source.names() if source is not None else set()
        for keyword in keywords:
            if keyword in named:
                continue
            if self.kwarg is None or keyword in source_named:
                return False
        supplied = set(keywords) | {n for n, _ in self.positional[:positional]}
        required = [n for n, default in (*self.positional, *self.keyword) if not default]
        return all(n in supplied for n in required)


@dataclass
class _Render:
    template: Template
    bindings: dict[str, Bound | list[Bound] | dict[str, Bound]]
    base: Bound | None
    transforms: Callable[[str], Callable[..., Any] | None]
    bt: str
    splices: dict[str, list[Any]] = field(default_factory=dict)
    _ids: Any = field(default_factory=count)

    def splice(self, args: list[Any]) -> Any:
        key = f"__batcher_migrate_splice_{next(self._ids)}__"
        self.splices[key] = args
        return cst.Name(key)

    def value(self, node: Any) -> Any:
        """Evaluate a right-side subtree to a `Bound`, list or dict for a transform's argument."""
        if isinstance(node, cst.Name):
            if node.value == "self":
                if self.base is None:
                    raise Declined("template needs a receiver")
                return self.base
            if node.value in self.bindings:
                return self.bindings[node.value]
        return Bound(self.expression(node), None)

    def expression(self, node: Any) -> Any:
        return node.visit(_Substitute(self))


class _Substitute(cst.CSTTransformer):  # type: ignore[misc]
    def __init__(self, render: _Render) -> None:
        super().__init__()
        self.render = render
        self.results: dict[int, Any] = {}
        self.labels: set[int] = set()  # keyword and attribute names, which never substitute
        self.wrapped: set[int] = set()  # parentheses added around a substituted argument
        self.absent: set[int] = set()  # `None` defaults the call did not pass
        self.keep: list[Any] = []  # holds nodes whose ids the two sets above record

    def visit_Arg(self, node: Any) -> None:
        if node.keyword is not None:
            self.labels.add(id(node.keyword))

    def visit_Attribute(self, node: Any) -> None:
        self.labels.add(id(node.attr))

    def visit_Call(self, node: Any) -> bool:
        func = node.func
        if not (
            isinstance(func, cst.Attribute)
            and isinstance(func.value, cst.Name)
            and func.value.value == "sem"
        ):
            return True
        transform = self.render.transforms(func.attr.value)
        if transform is None:
            raise Declined(f"unknown transform sem.{func.attr.value}")
        args = [self.render.value(a.value) for a in node.args]
        result = transform(*args)
        if result is None:
            raise Declined(f"sem.{func.attr.value} declined")
        self.results[id(node)] = self.render.splice(result) if isinstance(result, list) else result
        return False

    def leave_Call(self, original: Any, updated: Any) -> Any:
        if id(original) in self.results:
            return self.results[id(original)]
        args: list[Any] = []
        for arg in updated.args:
            args.extend(self._expand(arg))
        return simple_call(updated.func, args)

    def _expand(self, arg: Any) -> list[Any]:
        value = arg.value
        if id(value) in self.wrapped:  # an argument needs no parentheses of its own
            arg = arg.with_changes(value=value.with_changes(lpar=[], rpar=[]))
            value = arg.value
        if isinstance(value, cst.Name) and value.value in self.render.splices:
            return list(self.render.splices[value.value])
        if arg.star == "**" and isinstance(value, cst.Dict):
            out = []
            for element in value.elements:
                key = element.key
                text = key.evaluated_value if isinstance(key, cst.SimpleString) else None
                if not (isinstance(text, str) and text.isidentifier()):
                    return [arg]
                item = element.value
                if id(item) in self.wrapped:
                    item = item.with_changes(lpar=[], rpar=[])
                out.append(cst.Arg(keyword=cst.Name(text), value=item))
            return out
        if arg.keyword is not None and id(value) in self.absent:
            return []  # a `None` default the call did not pass is simply not passed
        return [arg]

    def leave_Name(self, original: Any, updated: Any) -> Any:
        render = self.render
        if id(original) in self.labels:
            return updated
        if original.value == "self":
            if render.base is None:
                raise Declined("template needs a receiver")
            return parens(render.base.node)
        if original.value == "bt":
            return cst.Name(render.bt)
        bound = render.bindings.get(original.value)
        if isinstance(bound, Bound) and not bound.passed and _is_none(bound.node):
            node = cst.Name("None")
            self.absent.add(id(node))
            self.keep.append(node)
            return node
        if isinstance(bound, Bound):
            wrapped = parens(bound.node)
            if wrapped is not bound.node:
                self.wrapped.add(id(wrapped))
                self.keep.append(wrapped)
            return wrapped
        if isinstance(bound, list):
            return render.splice([cst.Arg(b.node) for b in bound])
        if isinstance(bound, dict):
            return render.splice([cst.Arg(b.node, keyword=cst.Name(k)) for k, b in bound.items()])
        return updated

    def leave_Arg(self, _original: Any, updated: Any) -> Any:
        # `*cols` / `**kw` where the parameter is a vararg/kwarg: the Name already became a
        # splice, so drop the star and let `_expand` splice the arguments in.
        value = updated.value
        if updated.star and isinstance(value, cst.Name) and value.value in self.render.splices:
            return updated.with_changes(star="")
        return updated


def _is_none(node: Any) -> bool:
    return isinstance(node, cst.Name) and node.value == "None"


def bind(
    params: ast.arguments, args: list[Any], originals: list[Any]
) -> dict[str, Bound | list[Bound] | dict[str, Bound]]:
    """Bind call arguments to a template's left side, as Python binds a call.

    Args:
        params: The template's parameter list.
        args: The call's rewritten `cst.Arg` nodes.
        originals: The same arguments in the unmodified script.

    Returns:
        Each parameter's bound value; a vararg is a list and a kwarg a dict.

    Raises:
        Declined: On a starred argument, a surplus or duplicate argument, or a missing one.
    """
    positional = [*params.posonlyargs, *params.args]
    kwonly = params.kwonlyargs
    out: dict[str, Bound | list[Bound] | dict[str, Bound]] = {}
    extra: list[Bound] = []
    extra_kw: dict[str, Bound] = {}
    index = 0
    for arg, orig in zip(args, originals, strict=True):
        bound = Bound(arg.value, orig.value)
        if arg.star:
            raise Declined("starred argument")
        if arg.keyword is None:
            if index < len(positional):
                out[positional[index].arg] = bound
                index += 1
            elif params.vararg is not None:
                extra.append(bound)
            else:
                raise Declined("too many positional arguments")
            continue
        name = arg.keyword.value
        names = {a.arg for a in [*params.args, *kwonly]}
        if name in out:
            raise Declined(f"{name} passed twice")
        if name in names:
            out[name] = bound
        elif params.kwarg is not None:
            extra_kw[name] = bound
        else:
            raise Declined(f"unexpected keyword {name}")
    defaults = dict(zip([a.arg for a in positional][::-1], params.defaults[::-1], strict=False))
    defaults |= {a.arg: d for a, d in zip(kwonly, params.kw_defaults, strict=True) if d is not None}
    for param in [*positional, *kwonly]:
        if param.arg in out:
            continue
        if param.arg not in defaults:
            raise Declined(f"missing {param.arg}")
        out[param.arg] = Bound(_default(defaults[param.arg]), None, False)
    if params.vararg is not None:
        out[params.vararg.arg] = extra
    if params.kwarg is not None:
        out[params.kwarg.arg] = extra_kw
    return out


def render(
    template: Template,
    call_args: list[Any],
    originals: list[Any],
    base: Bound | None,
    transforms: Callable[[str], Callable[..., Any] | None],
    bt: str = "bt",
) -> Any | None:
    """Render a template for one call, or `None` when it declines.

    Args:
        template: The parsed template.
        call_args: The call's rewritten arguments.
        originals: The call's arguments in the unmodified script.
        base: The rewritten receiver (and its original), for `self`.
        transforms: Looks a `sem.<name>` transform up; each takes `Bound`/list/dict values.
        bt: The name the Batcher module is bound to.

    Returns:
        The replacement expression, or `None`.
    """
    try:
        bindings = bind(template.params, call_args, originals)
        state = _Render(template, bindings, base, transforms, bt)
        return state.expression(cst.parse_expression(template.target))
    except Declined:
        return None


def _default(expr: ast.expr) -> Any:
    """A template default as a node, with a string spelled in double quotes."""
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        return cst.SimpleString(json.dumps(expr.value))
    return cst.parse_expression(ast.unparse(expr))
