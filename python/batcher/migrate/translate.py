"""Rewrite a PySpark, Polars, Daft or Ray Data script onto Batcher, driven by the registry.

Every call and property access whose receiver the inference can type (`receivers`, seeded by
`engines.SPECS`) is looked up in the migration registry, and the row decides:

* a **template** renders the rewrite (`templates`, with `semantics` transforms), whatever the
  status, because a template is the semantics-preserving form a row vouches for; a template
  that does not bind leaves the call alone with a marker;
* `canonical` and `alias` rows with one Batcher spelling are rewritten 1:1 when the call's
  arguments bind to the Batcher signature (`templates.Signature.accepts`);
* `param` rows are rewritten the same way and keep a `# batcher-migrate:` marker naming the
  missing option;
* `mismatch`, `gap` and `out_of_scope` rows, a call that does not bind, and a call on a
  receiver the inference cannot type are left exactly as written, with a marker saying why.

The result reads as Batcher code with the residue flagged in place, and every site is in the
returned `Report`. `finish` then settles markers and imports.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from batcher._internal.migration import Mapping, Status, Template, parse_template
from batcher._internal.optional import require
from batcher.migrate.engines import ENGINE_LABELS, Tables, tables
from batcher.migrate.finish import Report, SiteRecorder, finish
from batcher.migrate.receivers import Inference, Member, infer_module
from batcher.migrate.semantics import columns, lookup, relational
from batcher.migrate.templates import Bound, Signature, parens, render

cst = require("libcst", feature="batcher.migrate", provides="libcst", extra="migrate")
metadata = require("libcst.metadata", feature="batcher.migrate", provides="libcst", extra="migrate")

__all__ = ["translate"]

_BUILTIN_METHODS = frozenset(
    n for kind in (str, bytes, list, dict, set) for n in dir(kind) if not n.startswith("_")
)
# Engine plumbing that is not a registry surface but whose methods carry over by name: Polars'
# `when/then/otherwise` builder is Batcher's `CaseBuilder`.
_PASSTHROUGH = {
    ("polars", "When"): frozenset({"then"}),
    ("polars", "Then"): frozenset({"when", "otherwise"}),
    ("polars", "ChainedWhen"): frozenset({"then"}),
    ("polars", "ChainedThen"): frozenset({"when", "otherwise"}),
}
_SESSION_BUILDERS = frozenset({"getOrCreate", "create"})


@lru_cache(maxsize=4096)
def _template(text: str, name: str) -> Template:
    return parse_template(text, name)


class _SiteContext:
    """The `semantics.Context` one rewrite site hands its transforms."""

    def __init__(self, owner: _Translate) -> None:
        self.owner = owner
        self.engine = owner.tables.spec.name
        self.bt = "bt"
        self.notes: list[str] = []

    def receiver(self, original: Any) -> str | None:
        return self.owner.inference.receiver(original) if original is not None else None

    def rewritten(self, original: Any) -> Any:
        return self.owner.rewritten.get(id(original), original)

    def consume(self, original: Any) -> None:
        self.owner.consumed.add(id(original))

    def definition(self, name: str) -> Any | None:
        return self.owner.definitions.get(name)

    def note(self, text: str) -> None:
        self.notes.append(text)


class _Translate(SiteRecorder):
    def __init__(self, tables: Tables, scopes: dict[Any, Inference], module: Any) -> None:
        super().__init__(scopes, module, ENGINE_LABELS[tables.spec.name])
        self.tables = tables
        self.definitions = _single_assignments(module)
        self.registry_surfaces = {s for s, _ in tables.rows}
        self.decorators: set[int] = set()
        self.surfaces = self.registry_surfaces | set(tables.returns)
        self.frame_names = {
            n
            for (s, n) in tables.rows
            if tables.batcher_receiver(s) in ("Dataset", "Expr", "GroupBy")
        }

    # -- calls and attributes ------------------------------------------------------------

    def visit_Decorator(self, node: Any) -> None:
        self.decorators.add(id(node.decorator))

    def leave_Call(self, original: Any, updated: Any) -> Any:
        if self.lambdas:
            return self.keep(original, updated)
        callee = self._callee(original)
        if callee is None:
            return self.keep(original, updated)
        surface, name, base = callee
        spelling = f"{surface}.{name}"
        if name in _PASSTHROUGH.get((self.tables.spec.name, surface), ()):
            return self.keep(original, self._passthrough(original, updated, surface, name))
        if surface == "SparkSession.Builder" and name in _SESSION_BUILDERS:
            return self.keep(original, self._session(original, updated, spelling))
        row = self.tables.row(surface, name)
        if row is None:
            if surface in self.registry_surfaces or surface in self.tables.spec.droppable:
                self.mark(original, None, spelling, "is not in the migration registry")
            return self.keep(original, updated)
        return self.keep(original, self._apply(row, original, updated, base))

    def leave_Attribute(self, original: Any, updated: Any) -> Any:
        if id(original) in self.call_funcs or self.lambdas:
            return self.keep(original, updated)
        surface = self.inference.receiver(original.value)
        if surface not in self.surfaces:
            return self.keep(original, updated)
        name = original.attr.value
        row = self.tables.row(surface, name)
        tokens = self.tables.params.get(surface, {}).get(name)
        if row is None and tokens is None and surface in self.tables.spec.seeds.attribute_columns:
            # PySpark's `df.age` is the column `age`; Batcher spells it `df["age"]`.
            element = cst.SubscriptElement(cst.Index(cst.SimpleString(f'"{name}"')))
            self.record(original, f"{surface}.<column>", None, "rewritten", f'["{name}"]', None)
            return self.keep(original, cst.Subscript(value=updated.value, slice=[element]))
        namespace = (
            row is not None and row.batcher[:1] and row.batcher[0] in self.tables.batcher_returns
        )
        # A decorator (`@daft.func`) is a use of the name even though it is not called here.
        decorator = id(original) in self.decorators and row is not None and row.template is None
        if decorator and row.status not in (Status.CANONICAL, Status.PARAM):
            return self.keep(original, self._apply(row, original, updated, original.value))
        if row is None or not (tokens == ["@property"] or namespace):
            return self.keep(original, updated)
        return self.keep(original, self._apply(row, original, updated, original.value))

    def _callee(self, original: Any) -> tuple[str, str, Any] | None:
        func = original.func
        if isinstance(func, cst.Attribute):
            surface = self.inference.receiver(func.value)
            if surface is None:
                self._unknown(original, func.attr.value)
                return None
            return (surface, func.attr.value, func.value) if surface in self.surfaces else None
        if isinstance(func, cst.Name):
            bound = self.inference.scope.lookup(func.value)
            if isinstance(bound, Member) and bound.receiver in self.surfaces:
                return bound.receiver, bound.name, None
        return None

    def _unknown(self, original: Any, name: str) -> None:
        top = self.scopes[self.stack[0]].scope.names.values()
        uses_engine = any(v in self.tables.spec.droppable for v in top)
        if uses_engine and name in self.frame_names and name not in _BUILTIN_METHODS:
            why = "is called on a receiver the codemod cannot type; left as written"
            self.mark(original, None, f".{name}", why)

    # -- rules ---------------------------------------------------------------------------

    def _apply(self, row: Mapping, original: Any, updated: Any, base: Any) -> Any:
        spelling = f"{row.surface}.{row.name}"
        if row.template is not None:
            return self._render(row, original, updated, base, spelling)
        if row.status not in (Status.CANONICAL, Status.PARAM):
            why = {
                Status.MISMATCH: f"differs in Batcher (`{' / '.join(row.batcher)}`): {row.note}",
                Status.GAP: f"has no Batcher equivalent yet: {row.need}",
                Status.OUT_OF_SCOPE: f"is not provided by Batcher: {row.reason}",
            }[row.status]
            return self._leave(original, updated, row, why)
        if len(row.batcher) != 1 or row.batcher[0].startswith("op:"):
            why = f"maps to {' / '.join(row.batcher)}; rewrite by hand"
            return self._leave(original, updated, row, why)
        new = self._generic(row, original, updated, base)
        target = row.batcher[0]
        if new is None:
            why = f"is `{target}` in Batcher, but this call does not carry over 1:1"
            return self._leave(original, updated, row, why)
        if row.status is Status.PARAM:
            marker = (
                f"{self.label} `{spelling}` was rewritten to `{target}`, which lacks: {row.need}"
            )
            self.record(original, spelling, row, "rewritten+marked", target, marker)
        else:
            self.record(original, spelling, row, "rewritten", target, None)
        return new

    def _leave(self, original: Any, updated: Any, row: Mapping, why: str) -> Any:
        """Mark a call left as written, arguments included.

        The arguments were already rewritten on the way up. A call that stays in the source
        engine keeps them in its terms too, so the marked line reads exactly as the source did
        (a Polars lambda, a `F.concat(F.col(...))`), and the sites inside it are not reported as
        rewritten.
        """
        self.mark(original, row, f"{row.surface}.{row.name}", why)
        if not isinstance(original, cst.Call):
            return updated
        inner: set[int] = set()
        for arg in original.args:
            arg.visit(_Consume(inner))
        # A rewrite inside the reverted arguments did not happen; a marker there still applies.
        self.consumed |= {
            i for i in inner if all(site.action != "marked" for site, _ in self.sites.get(i, []))
        }
        return updated.with_changes(args=original.args)

    def _render(self, row: Mapping, original: Any, updated: Any, base: Any, spelling: str) -> Any:
        template = _template(str(row.template), row.name)
        ctx = _SiteContext(self)
        is_call = isinstance(original, cst.Call)
        spelled = updated.func if is_call else updated
        head = spelled.value if base is not None else None
        new = render(
            template,
            list(updated.args) if is_call else [],
            list(original.args) if is_call else [],
            Bound(head, base) if base is not None else None,
            lookup(ctx),
        )
        if new is None:
            why = row.note or row.need or row.reason or "takes arguments the template cannot bind"
            return self._leave(original, updated, row, f"needs a manual rewrite: {why}")
        if head is not None and isinstance(spelled, cst.Attribute):
            new = _with_dot(new, head, spelled.dot)
        if ctx.notes:
            texts = dict.fromkeys(n or row.note or row.need or "" for n in ctx.notes)
            marker = f"{self.label} `{spelling}` was rewritten; check: {'; '.join(texts)}"
            self.record(original, spelling, row, "rewritten+marked", template.target, marker)
        else:
            self.record(original, spelling, row, "rewritten", template.target, None)
        return new

    def _generic(self, row: Mapping, original: Any, updated: Any, base: Any) -> Any | None:
        t = self.tables
        split = t.split_batcher(row.batcher[0])
        if split is None:
            return None
        receiver, member = split
        is_call = isinstance(original, cst.Call)
        args = list(updated.args) if is_call else []
        shape = self._shape(row, updated, base, _expr_root(f"{receiver}.{member}"), args, original)
        if shape is None:
            return None
        func, args, function_style = shape
        tokens = t.batcher_params.get(receiver, {}).get(member)
        if not is_call:
            return self._property(row, func, tokens)
        source_tokens = t.params.get(row.surface, {}).get(row.name)
        source = Signature.parse(source_tokens) if source_tokens else None
        if tokens is None or not _binds(Signature.parse(tokens), args, source):
            return None
        if function_style and source is not None:
            args = _column_args(args, source, skip=len(args) != len(original.args))
        return updated.with_changes(func=func, args=args)

    def _shape(
        self, row: Mapping, updated: Any, base: Any, full: str, args: list[Any], original: Any
    ) -> tuple[Any, list[Any], bool] | None:
        """The Batcher callee (and remaining arguments) a 1:1 rewrite spells.

        A method on the matching Batcher receiver keeps its base; a `bt.<name>` spelling
        replaces a module or session base; a function whose Batcher spelling is an `Expr`
        method moves its first argument into the receiver (`F.upper("n")` is
        `bt.col("n").str.upper()`).
        """
        source = self.tables.batcher_receiver(row.surface)
        function_style = base is None or row.surface in self.tables.spec.droppable
        spelled = updated.func if isinstance(updated, cst.Call) else updated
        if not function_style and source and full.startswith(_expr_root(source) + "."):
            head, path = spelled.value, full[len(_expr_root(source)) + 1 :]
        elif function_style and full.startswith("bt."):
            head, path = cst.Name("bt"), full[3:]
        elif function_style and full.startswith("Expr.") and args:
            first = args[0]
            if first.keyword is not None or first.star:
                return None
            ctx = _SiteContext(self)
            column = columns.column(ctx, Bound(first.value, original.args[0].value))
            if column is None:
                return None
            head, path, args = parens(column), full[len("Expr.") :], args[1:]
        else:
            return None
        *prefix, last = path.split(".")
        for segment in prefix:
            head = cst.Attribute(value=head, attr=cst.Name(segment))
        if isinstance(spelled, cst.Attribute) and not function_style:
            # Keep the source's own `.` layout, so a chain split over lines stays split.
            return spelled.with_changes(value=head, attr=cst.Name(last)), args, False
        return cst.Attribute(value=head, attr=cst.Name(last)), args, function_style

    def _property(self, row: Mapping, func: Any, tokens: list[str] | None) -> Any | None:
        if tokens == ["@property"] or row.batcher[0] in self.tables.batcher_returns:
            return func
        foreign_property = self.tables.params.get(row.surface, {}).get(row.name) == ["@property"]
        callable_bare = tokens is not None and Signature.parse(tokens).accepts(0, [], None)
        return cst.Call(func=func) if foreign_property and callable_bare else None

    def _passthrough(self, original: Any, updated: Any, surface: str, name: str) -> Any:
        tokens = self.tables.params.get(surface, {}).get(name)
        args = list(updated.args)
        if tokens:
            args = _column_args(args, Signature.parse(tokens), skip=False)
        self.record(original, f"{surface}.{name}", None, "rewritten", name, None)
        return updated.with_changes(args=args)

    def _session(self, original: Any, updated: Any, spelling: str) -> Any:
        ctx = _SiteContext(self)
        new = relational.spark_session(ctx, Bound(updated, original))
        if new is None:
            self.mark(original, None, spelling, "builds a session the codemod cannot translate")
            return updated
        marker = f"{self.label} `{spelling}`: {'; '.join(ctx.notes)}" if ctx.notes else None
        action = "rewritten+marked" if marker else "rewritten"
        self.record(original, spelling, None, action, "bt.Session()", marker)
        return new


class _Consume(cst.CSTVisitor):  # type: ignore[misc]
    """Collect every node under an argument, so the rewrites recorded on them can be dropped."""

    def __init__(self, consumed: set[int]) -> None:
        super().__init__()
        self.consumed = consumed

    def on_visit(self, node: Any) -> bool:
        self.consumed.add(id(node))
        return True


def _binds(target: Signature, args: list[Any], source: Signature | None) -> bool:
    if any(a.star == "*" for a in args) and target.vararg is None:
        return False
    if any(a.star == "**" for a in args) and target.kwarg is None:
        return False
    plain = [a for a in args if not a.star]
    keywords = [a.keyword.value for a in plain if a.keyword is not None]
    return target.accepts(sum(1 for a in plain if a.keyword is None), keywords, source)


def _with_dot(node: Any, head: Any, dot: Any) -> Any:
    """Give the rendered call hanging off `head` the source's `.` layout (a line break)."""
    if isinstance(node, cst.Call) and isinstance(node.func, cst.Attribute):
        if node.func.value is head:
            return node.with_changes(func=node.func.with_changes(dot=dot))
        inner = _with_dot(node.func.value, head, dot)
        if inner is not node.func.value:
            return node.with_changes(func=node.func.with_changes(value=inner))
    return node


def _expr_root(path: str) -> str:
    for prefix in ("AggExpr", "WindowExpr"):
        if path == prefix or path.startswith(prefix + "."):
            return "Expr" + path[len(prefix) :]
    return path


def _column_args(args: list[Any], source: Signature, *, skip: bool) -> list[Any]:
    """Wrap string literals passed where the source signature takes a column name."""
    names = [n for n, _ in source.positional]
    out = []
    index = 1 if skip else 0
    for arg in args:
        if arg.keyword is not None:
            param = arg.keyword.value
        elif index < len(names):
            param, index = names[index], index + 1
        else:
            param = source.vararg
        if param in source.columns and isinstance(arg.value, cst.SimpleString):
            col = cst.Attribute(value=cst.Name("bt"), attr=cst.Name("col"))
            arg = arg.with_changes(value=cst.Call(func=col, args=[cst.Arg(arg.value)]))
        out.append(arg)
    return out


def _single_assignments(module: Any) -> dict[str, Any]:
    """Names assigned exactly once anywhere in the file, to their value (for window specs)."""
    seen: dict[str, list[Any]] = {}

    class _Collect(cst.CSTVisitor):  # type: ignore[misc]
        def visit_Assign(self, node: Any) -> None:
            for target in node.targets:
                if isinstance(target.target, cst.Name):
                    seen.setdefault(target.target.value, []).append(node.value)

    module.visit(_Collect())
    return {name: values[0] for name, values in seen.items() if len(values) == 1}


def translate(source: str, engine: str) -> tuple[str, Report]:
    """Rewrite one script from a foreign engine onto Batcher.

    Args:
        source: Python source written against `engine`.
        engine: `pyspark`, `polars`, `daft` or `ray_data`.

    Returns:
        The rewritten source, and every site the rewrite touched or declined.

    Examples:
        .. doctest::

            >>> from batcher.migrate.translate import translate
            >>> code, report = translate('import polars as pl\\ne = pl.col("a")\\n', "polars")
            >>> print(code, end="")
            import batcher as bt
            e = bt.col("a")
            >>> report.counts()
            {'rewritten': 1}
    """
    t = tables(engine)
    wrapper = metadata.MetadataWrapper(cst.parse_module(source))
    module = wrapper.module
    scopes = infer_module(module, t.returns, seeds=t.spec.seeds, members=t.members())
    transformer = _Translate(t, scopes, module)
    rewritten = wrapper.visit(transformer)
    report, markers = transformer.report()
    return finish(rewritten, t.spec, markers, direction="inbound"), report
