"""Rewrite Batcher's own removed second spellings to the one spelling that stays.

This is the rule set the alias removal runs over the repository before the removed names are
deleted, and that a user runs over their code after upgrading. The decisions come from
`renames.toml` and `kwarg_renames.toml` (typed in `_internal.migration.renames`); which
expressions are Batcher objects comes from `receivers`. Five kinds of rule apply:

* a rename, `ds.groupby("k")` to `ds.group_by("k")`;
* a path, `ds.to_csv(p)` to `ds.write.csv(p)`, and `bt.read_csv(p)` to `bt.read.csv(p)`;
* a call, the property `ds.height` to `ds.count()`;
* an operator, `a.add(b)` to `(a + b)`;
* a transform, `ds.with_column("x", e)` to `ds.with_columns(x=e)`.

Keyword spellings are rewritten on the same calls: `sort(by=["a"], ascending=False)` becomes
`sort("a", descending=True)`. A keyword value the rewrite cannot transform exactly (a
non-literal `ascending=flag`, which could be a list) is left as written and reported.

Nothing is rewritten on an expression whose receiver cannot be inferred. That is the common
case in a test comparing against pandas or Polars, where the same word on a different
library's object must not change.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

from batcher._internal.migration import OPERATORS, KwargRename, Rename
from batcher._internal.optional import require
from batcher.migrate.receivers import ClassRef, FunctionRef, Inference, Member, infer_module

cst = require("libcst", feature="batcher.migrate", provides="libcst", extra="migrate")
metadata = require("libcst.metadata", feature="batcher.migrate", provides="libcst", extra="migrate")

__all__ = ["Edit", "RenameReport", "canonicalize"]

_EXPR_RECEIVERS = ("AggExpr", "WindowExpr")
_BUILTIN_METHODS = frozenset(
    name for kind in (str, bytes, list, dict, set) for name in dir(kind) if not name.startswith("_")
)
_CALL_KINDS = frozenset({"operator", "transform"})


def _kwarg(name: str, value: object) -> object:
    """A keyword argument rendered `name=value`, the way hand-written code spells it."""
    tight = cst.AssignEqual(
        whitespace_before=cst.SimpleWhitespace(""), whitespace_after=cst.SimpleWhitespace("")
    )
    return cst.Arg(keyword=cst.Name(name), value=value, equal=tight)


@dataclass(frozen=True)
class Edit:
    """One site the rewrite touched or declined."""

    line: int
    old: str
    new: str | None
    receiver: str | None


@dataclass
class RenameReport:
    """Every site rewritten, and every one left alone with the reason recorded in `new`."""

    renamed: list[Edit] = field(default_factory=list)
    unresolved: list[Edit] = field(default_factory=list)


def _rule(renames: dict[str, dict[str, Rename]], receiver: str, name: str) -> Rename | None:
    found = renames.get(receiver, {}).get(name)
    if found is None and receiver in _EXPR_RECEIVERS:
        found = renames.get("Expr", {}).get(name)
    return found


def _parens(node: object) -> object:
    atoms = (
        cst.Name,
        cst.Attribute,
        cst.Call,
        cst.Subscript,
        cst.SimpleString,
        cst.Integer,
        cst.Float,
    )
    if isinstance(node, atoms) or getattr(node, "lpar", None):
        return node
    return node.with_changes(lpar=[cst.LeftParen()], rpar=[cst.RightParen()])  # type: ignore[attr-defined]


def _binary() -> dict[str, object]:
    return {
        "+": cst.Add,
        "-": cst.Subtract,
        "*": cst.Multiply,
        "/": cst.Divide,
        "//": cst.FloorDivide,
        "%": cst.Modulo,
        "**": cst.Power,
        "&": cst.BitAnd,
        "|": cst.BitOr,
        "^": cst.BitXor,
    }


def _compare() -> dict[str, object]:
    return {
        "==": cst.Equal,
        "!=": cst.NotEqual,
        "<": cst.LessThan,
        "<=": cst.LessThanEqual,
        ">": cst.GreaterThan,
        ">=": cst.GreaterThanEqual,
    }


def _operator_call(rule: Rename, base: object, call: object) -> object | None:
    symbol = OPERATORS[rule.operator]
    args = call.args  # type: ignore[attr-defined]
    if any(a.keyword is not None or a.star for a in args):
        return None
    left = _parens(base)
    if rule.operator in ("invert", "neg"):
        if args:
            return None
        op = cst.BitInvert() if rule.operator == "invert" else cst.Minus()
        node = cst.UnaryOperation(operator=op, expression=left)
    elif len(args) != 1:
        return None
    elif symbol in _compare():
        target = cst.ComparisonTarget(
            operator=_compare()[symbol](), comparator=_parens(args[0].value)
        )
        node = cst.Comparison(left=left, comparisons=[target])
    else:
        node = cst.BinaryOperation(
            left=left, operator=_binary()[symbol](), right=_parens(args[0].value)
        )
    return node.with_changes(lpar=[cst.LeftParen()], rpar=[cst.RightParen()])


def _argument(call: object, position: int, keyword: str) -> object | None:
    args = call.args  # type: ignore[attr-defined]
    for arg in args:
        if arg.keyword is not None and arg.keyword.value == keyword:
            return arg.value
    positional = [a for a in args if a.keyword is None and not a.star]
    return positional[position].value if position < len(positional) else None


def _transform_call(rule: Rename, base: object, call: object) -> object | None:
    if rule.transform == "identity":
        # `ds.lazy()` / `ds.copy()`: a Dataset is already lazy and immutable, so the call is
        # the dataset itself.
        return base if not call.args else None  # type: ignore[attr-defined]
    func = cst.Attribute(value=base, attr=cst.Name(rule.to))
    if rule.transform == "with_column":
        name, expr = _argument(call, 0, "name"), _argument(call, 1, "expr")
        if name is None or expr is None or len(call.args) != 2:  # type: ignore[attr-defined]
            return None
        text = name.evaluated_value if isinstance(name, cst.SimpleString) else None
        if isinstance(text, str) and text.isidentifier():
            arg = _kwarg(text, expr)
        else:
            arg = cst.Arg(value=cst.Dict([cst.DictElement(name, expr)]), star="**")
        return cst.Call(func=func, args=[arg])
    if rule.transform == "slice_to_limit":
        offset, length = _argument(call, 0, "offset"), _argument(call, 1, "length")
        if offset is None or length is None:
            # `slice(offset)` with no length means "to the end", which `limit` cannot say.
            return None
        offset_arg = _kwarg("offset", offset)
        return cst.Call(func=func, args=[cst.Arg(length), offset_arg])
    return None


def _literal_elements(value: object) -> list[object] | None:
    if isinstance(value, (cst.List, cst.Tuple)):
        return [e.value for e in value.elements if not isinstance(e, cst.StarredElement)]
    if isinstance(value, cst.SimpleString):
        return [value]
    return None


def _negated(value: object) -> object | None:
    if isinstance(value, cst.Name) and value.value in ("True", "False"):
        return cst.Name("False" if value.value == "True" else "True")
    if isinstance(value, (cst.List, cst.Tuple)):
        flipped = [_negated(e.value) for e in value.elements]
        if all(f is not None for f in flipped):
            return value.with_changes(elements=[cst.Element(f) for f in flipped])
    return None


def _apply_kwargs(call: object, rules: dict[str, KwargRename], failed: list[str]) -> object:
    """Rewrite second keyword spellings on one call; record the ones it cannot rewrite."""
    if not rules:
        return call
    positional: list[object] = []
    keywords: list[object] = []
    inserted: list[object] = []
    changed = False
    present = {a.keyword.value for a in call.args if a.keyword is not None}  # type: ignore[attr-defined]
    for arg in call.args:  # type: ignore[attr-defined]
        rule = rules.get(arg.keyword.value) if arg.keyword is not None else None
        if rule is None:
            is_keyword = arg.keyword is not None or arg.star == "**"
            (keywords if is_keyword else positional).append(arg)
            continue
        value = arg.value
        changed = True
        # A call already passing the kept keyword as well (`descending=True, ascending=False`)
        # is a conflict the user has to resolve, not something to merge silently.
        collides = rule.to in present and rule.to != arg.keyword.value
        if collides:
            failed.append(arg.keyword.value)
            keywords.append(arg)
        elif rule.action == "rename":
            keywords.append(arg.with_changes(keyword=cst.Name(rule.to)))
        elif rule.action == "positional" and (elems := _literal_elements(value)) is not None:
            inserted.extend(cst.Arg(e) for e in elems)
        elif rule.action == "first_positional" and not positional:
            inserted.append(cst.Arg(value))
        elif rule.action == "negate" and (neg := _negated(value)) is not None:
            keywords.append(arg.with_changes(keyword=cst.Name(rule.to), value=neg))
        elif (
            rule.action == "nulls_first"
            and isinstance(value, cst.SimpleString)
            and value.evaluated_value in ("first", "last")
        ):
            first = cst.Name(str(value.evaluated_value == "first"))
            keywords.append(arg.with_changes(keyword=cst.Name(rule.to), value=first))
        else:
            failed.append(arg.keyword.value)
            keywords.append(arg)
    if not changed:
        return call
    args = [*positional, *inserted, *keywords]
    # Keep each argument's own comma (and the newline inside it on a multi-line call); only
    # an argument that had none and is no longer last needs one.
    last = cst.MaybeSentinel.DEFAULT
    trailing = call.args[-1].comma if call.args else last  # type: ignore[attr-defined]
    fixed = []
    for i, a in enumerate(args):
        if i == len(args) - 1:
            fixed.append(a.with_changes(comma=trailing))
        elif isinstance(a.comma, cst.Comma):
            fixed.append(a)
        else:
            fixed.append(
                a.with_changes(comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")))
            )
    return call.with_changes(args=fixed)  # type: ignore[attr-defined]


def _apply_keys(call: object, rule: Rename) -> object:
    """Turn the call's leading positional arguments into the rule's keywords, in order."""
    if not rule.keys:
        return call
    args = list(call.args)  # type: ignore[attr-defined]
    leading = []
    for arg in args:
        if arg.keyword is not None or arg.star:
            break
        leading.append(arg)
    for i, (arg, key) in enumerate(zip(leading, rule.keys, strict=False)):
        args[i] = arg.with_changes(keyword=cst.Name(key), equal=_kwarg(key, arg.value).equal)
    return call.with_changes(args=args)  # type: ignore[attr-defined]


def _apply_fill(call: object, rule: Rename) -> object:
    """Pass the removed spelling's default explicitly where the kept spelling's differs."""
    args = list(call.args)  # type: ignore[attr-defined]
    positional = [a for a in args if a.keyword is None and not a.star]
    named = {a.keyword.value for a in args if a.keyword is not None}
    has_star = any(a.star for a in args)
    for param, position, literal in rule.fill:
        if param in named or position < len(positional) or has_star:
            continue
        if args and not isinstance(args[-1].comma, cst.Comma):
            args[-1] = args[-1].with_changes(
                comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" "))
            )
        args.append(_kwarg(param, cst.parse_expression(literal)))
    return call.with_changes(args=args)  # type: ignore[attr-defined]


def _meaning_rename(rule: Rename, call: object) -> Rename | None:
    """The rule as a plain rename when the call has the old meaning's argument count, else None."""
    args = list(call.args)  # type: ignore[attr-defined]
    if any(a.star for a in args) or len(args) != rule.args:
        return None
    return dataclasses.replace(rule, args=None)


def _replacement(rule: Rename, base: object) -> object:
    node = base
    for segment in rule.to.split("."):
        node = cst.Attribute(value=node, attr=cst.Name(segment))
    return cst.Call(func=node) if rule.kind == "call" else node


class _Canonicalize(cst.CSTTransformer):  # type: ignore[misc]
    METADATA_DEPENDENCIES = (metadata.PositionProvider,)

    def __init__(
        self,
        scopes: dict[object, Inference],
        module: object,
        renames: dict[str, dict[str, Rename]],
        kwargs: dict[str, dict[str, KwargRename]],
        report: RenameReport,
        assume_accessors: bool = False,
    ) -> None:
        super().__init__()
        self.assume_accessors = assume_accessors
        self.scopes = scopes
        self.stack: list[object] = [module]
        self.renames = renames
        self.kwargs = kwargs
        self.report = report
        self.removed = {name for table in renames.values() for name in table}
        self.kwarg_methods = {m.rpartition(".")[2] for m in kwargs}
        self.attr_names: set[int] = set()
        top = scopes[module].scope.names.values()
        # An unknown receiver is only worth reporting in a file that uses Batcher at all, and
        # only for a word that is not also a method of Python's own containers.
        self.uses_batcher = any(
            v == "bt" or isinstance(v, (Member, ClassRef, FunctionRef)) for v in top
        )

    @property
    def inference(self) -> Inference:
        return self.scopes[self.stack[-1]]

    def _receiver(self, node: object) -> str | None:
        found = self.inference.receiver(node)
        # Opt-in, for code known to hold no pandas/cuDF/Polars objects (the expression and SQL
        # layers): there `x.str` on an unknown `x` is a Batcher accessor. It is wrong for
        # `core/gpu_plan`, which calls pandas' own `.str` and `.dt` methods.
        if (
            found is None
            and self.assume_accessors
            and isinstance(node, cst.Attribute)
            and node.attr.value in ("str", "dt", "list", "struct", "json", "map")
        ):
            return f"Expr.{node.attr.value}"
        return found

    def _line(self, node: object) -> int:
        return self.get_metadata(metadata.PositionProvider, node).start.line

    def _unresolved(self, node: object, name: str, why: str, *, known: bool = False) -> None:
        # A site on a proven Batcher receiver is always reported; an unknown receiver only in a
        # file that uses Batcher, and never for a word Python's own containers also spell.
        if known or (self.uses_batcher and name not in _BUILTIN_METHODS):
            self.report.unresolved.append(Edit(self._line(node), name, why, None))

    def visit_FunctionDef(self, node: object) -> None:
        self.stack.append(node)

    def leave_FunctionDef(self, _original: object, updated: object) -> object:
        self.stack.pop()
        return updated

    def leave_ClassDef(self, original: object, updated: object) -> object:
        """Rename the hooks a subclass of a Batcher class defines (`onQueryProgress`)."""
        tables = []
        for base in original.bases:  # type: ignore[attr-defined]
            found = self.inference.value(base.value)
            receiver = found.receiver if isinstance(found, ClassRef) else self._receiver(base.value)
            if receiver in self.renames:
                tables.append((receiver, self.renames[receiver]))
        if not tables:
            return updated
        body = []
        for stmt in updated.body.body:  # type: ignore[attr-defined]
            if isinstance(stmt, cst.FunctionDef):
                for receiver, table in tables:
                    rule = table.get(stmt.name.value)
                    if rule is not None and rule.kind == "name":
                        edit = Edit(self._line(original), stmt.name.value, rule.to, receiver)
                        self.report.renamed.append(edit)
                        stmt = stmt.with_changes(name=cst.Name(rule.to))
            body.append(stmt)
        return updated.with_changes(body=updated.body.with_changes(body=body))  # type: ignore[attr-defined]

    def visit_Attribute(self, node: object) -> None:
        self.attr_names.add(id(node.attr))  # type: ignore[attr-defined]

    def visit_ImportAlias(self, node: object) -> None:
        # The name in `from batcher import read_csv` is rewritten by `leave_ImportFrom`, not as
        # a use of the imported name.
        self.attr_names.add(id(node.name))  # type: ignore[attr-defined]

    def leave_Attribute(self, original: object, updated: object) -> object:
        name = original.attr.value  # type: ignore[attr-defined]
        if name not in self.removed:
            return updated
        receiver = self._receiver(original.value)  # type: ignore[attr-defined]
        rule = _rule(self.renames, receiver, name) if receiver else None
        if rule is None:
            if receiver is None:
                self._unresolved(original, name, "receiver unknown")
            return updated
        if rule.kind in _CALL_KINDS or rule.args is not None:
            return updated  # rewritten together with its arguments in `leave_Call`
        self.report.renamed.append(Edit(self._line(original), name, rule.to, receiver))
        return _replacement(rule, updated.value)  # type: ignore[attr-defined]

    def leave_Call(self, original: object, updated: object) -> object:
        func = original.func  # type: ignore[attr-defined]
        if not isinstance(func, cst.Attribute):
            return updated
        name = func.attr.value
        if name not in self.removed and name not in self.kwarg_methods:
            return updated
        receiver = self._receiver(func.value)
        if receiver is None:
            return updated
        rule = _rule(self.renames, receiver, name)
        if rule is not None and rule.args is not None:
            rule = _meaning_rename(rule, updated)
            if rule is None:
                return updated
            # `leave_Attribute` left the name alone until the argument count was known.
            self.report.renamed.append(Edit(self._line(original), name, rule.to, receiver))
            updated = updated.with_changes(func=_replacement(rule, updated.func.value))  # type: ignore[attr-defined]
        failed: list[str] = []
        result = _apply_kwargs(updated, self.kwargs.get(f"{receiver}.{name}", {}), failed)
        if rule is not None and rule.kind in _CALL_KINDS:
            base = updated.func.value  # type: ignore[attr-defined]
            build = _operator_call if rule.kind == "operator" else _transform_call
            rewritten = build(rule, base, result)
            if rewritten is None:
                self._unresolved(original, name, f"{rule.kind} needs a manual rewrite", known=True)
            else:
                edit = Edit(self._line(original), name, rule.to or rule.operator, receiver)
                self.report.renamed.append(edit)
                result = rewritten
        elif rule is not None and isinstance(result, cst.Call):
            # The attribute was renamed on the way up; the kept method's keyword rules apply too.
            kept = f"{receiver}.{rule.to.rpartition('.')[2]}"
            result = _apply_kwargs(result, self.kwargs.get(kept, {}), failed)
            result = _apply_fill(_apply_keys(result, rule), rule)
        for keyword in failed:
            self._unresolved(
                original, f"{name}({keyword}=)", "keyword value needs a manual rewrite"
            )
        return result

    def leave_ImportFrom(self, original: object, updated: object) -> object:
        module = original.module  # type: ignore[attr-defined]
        if not (isinstance(module, cst.Name) and module.value == "batcher"):
            return updated
        if isinstance(updated.names, cst.ImportStar):  # type: ignore[attr-defined]
            return updated
        names: list[object] = []
        seen: set[str] = set()
        for alias in updated.names:  # type: ignore[attr-defined]
            old = alias.name.value
            rule = self.renames.get("bt", {}).get(old)
            if rule is not None and rule.kind in ("name", "path") and alias.asname is None:
                self.report.renamed.append(Edit(self._line(original), old, rule.to, "bt"))
                alias = alias.with_changes(name=cst.Name(rule.to.split(".")[0]))
            if alias.asname is None and alias.name.value in seen:
                continue
            seen.add(alias.name.value)
            names.append(alias)
        names[-1] = names[-1].with_changes(comma=cst.MaybeSentinel.DEFAULT)  # type: ignore[attr-defined]
        return updated.with_changes(names=names)

    def leave_Name(self, original: object, updated: object) -> object:
        if id(original) in self.attr_names:
            return updated
        bound = self.inference.scope.lookup(original.value)  # type: ignore[attr-defined]
        if not (isinstance(bound, Member) and bound.receiver == "bt" and bound.imported):
            return updated
        rule = self.renames.get("bt", {}).get(bound.name)
        # Uses of an un-aliased import: `from batcher import read_csv` binds `read_csv`, and
        # the import itself is rewritten in `leave_ImportFrom`.
        if rule is None or rule.kind not in ("name", "path"):
            return updated
        if original.value != bound.name:  # type: ignore[attr-defined]
            return updated
        head, _, attr = rule.to.partition(".")
        if attr:
            return cst.Attribute(value=cst.Name(head), attr=cst.Name(attr))
        return updated.with_changes(value=head)  # type: ignore[attr-defined]


def canonicalize(
    source: str,
    renames: dict[str, dict[str, Rename]],
    returns: dict[str, dict[str, str]],
    kwargs: dict[str, dict[str, KwargRename]] | None = None,
    *,
    assume_accessors: bool = False,
    imported: dict[str, str] | None = None,
) -> tuple[str, RenameReport]:
    """Rewrite removed Batcher spellings in one source file.

    Args:
        source: Python source text.
        renames: `{receiver: {removed: Rename}}`, from `load_renames()`.
        returns: `{receiver: {member: returned_receiver}}`, from `load_returns()`.
        kwargs: `{"<receiver>.<method>": {keyword: KwargRename}}`, from
            `load_kwarg_renames()`; keywords are left alone when omitted.
        assume_accessors: Treat `x.str`/`x.dt`/`x.list` on an unknown `x` as a Batcher
            accessor. Only for code that holds no pandas or Polars expressions.
        imported: `{name: receiver}` for functions imported from the script's own project,
            from `batcher.migrate.project.imported_function_receivers`.

    Returns:
        The rewritten source, and a report of what changed and what was left alone.
    """
    wrapper = metadata.MetadataWrapper(cst.parse_module(source))
    scopes = infer_module(wrapper.module, returns, imported)
    report = RenameReport()
    transformer = _Canonicalize(
        scopes, wrapper.module, renames, kwargs or {}, report, assume_accessors
    )
    rewritten = wrapper.visit(transformer)
    return rewritten.code, report
