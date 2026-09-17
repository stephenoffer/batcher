"""Rewrite a Batcher script onto PySpark, Polars, Daft or Ray Data: the conservative inverse.

Leaving Batcher uses the same registry read backwards, and only where reading it backwards is
exact. A Batcher call is rewritten when some registry row of the target engine is a `canonical`
or `alias` row whose single Batcher spelling is that call, or whose template is marked
reversible (`<->`), and the call's arguments bind to the foreign signature. When several rows
qualify, a `canonical` row beats an `alias`, a name equal to Batcher's beats another, and a
method beats a function; a tie that survives that is reported, not chosen.

Everything else is left as written with a `# batcher-migrate:` marker: a `param` or `mismatch`
row describes a difference in the *foreign-to-Batcher* direction, and the inverse of a one-way
template or a semantics transform is not something to guess at.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from batcher._internal.migration import Mapping, Status, parse_template
from batcher._internal.optional import require
from batcher.migrate.engines import ENGINE_LABELS, Tables, tables
from batcher.migrate.finish import Report, SiteRecorder, finish
from batcher.migrate.receivers import Inference, Member, infer_module
from batcher.migrate.templates import Signature, simple_call

cst = require("libcst", feature="batcher.migrate", provides="libcst", extra="migrate")
metadata = require("libcst.metadata", feature="batcher.migrate", provides="libcst", extra="migrate")

__all__ = ["export"]


@dataclass(frozen=True)
class _Candidate:
    """One way to spell a Batcher member in the foreign engine.

    `kind` is `method` (same base), `module` (`F.<name>`, `pl.<name>`, `spark.<name>`) or
    `function` (the Batcher receiver becomes the first argument). `pattern` is a reversible
    template's right side, parsed, or `None` for a 1:1 rename.
    """

    row: Mapping
    kind: str
    preference: int
    pattern: ast.Call | None = None

    def rank(self, member: str) -> tuple[int, int, int, int]:
        return (
            0 if self.row.status is Status.CANONICAL else 1,
            0 if self.row.name == member.rpartition(".")[2] else 1,
            ("method", "module", "function").index(self.kind),
            self.preference,
        )


def _norm(receiver: str) -> str:
    for prefix in ("AggExpr", "WindowExpr", "CaseBuilder"):
        if receiver == prefix or receiver.startswith(prefix + "."):
            return "Expr" + receiver[len(prefix) :]
    return receiver


@lru_cache(maxsize=8)
def _inverse(engine: str) -> dict[tuple[str, str], list[_Candidate]]:
    t = tables(engine)
    out: dict[tuple[str, str], list[_Candidate]] = {}
    objects = t.spec.objects
    for row in t.rows.values():
        source = t.batcher_receiver(row.surface)
        preference = objects.index(row.surface) if row.surface in objects else len(objects)
        if row.template is not None:
            # A reversible template vouches for its exact form whatever the row's status.
            template = parse_template(row.template, row.name)
            if template.reversible:
                pattern = ast.parse(template.target, mode="eval").body
                _add_pattern(out, _Candidate(row, "method", preference), source, pattern)  # type: ignore[arg-type]
            continue
        if row.status is not Status.CANONICAL or len(row.batcher) != 1:
            continue
        full = _norm(row.batcher[0])
        if source and row.surface in objects and full.startswith(_norm(source) + "."):
            key, kind = (_norm(source), full[len(_norm(source)) + 1 :]), "method"
        elif full.startswith("bt.") and _module_base(t, row.surface) is not None:
            key, kind = ("bt", full[3:]), "module"
        elif row.surface in t.spec.modules and full.startswith("Expr."):
            key, kind = ("Expr", full[5:]), "function"
        else:
            continue
        out.setdefault(key, []).append(_Candidate(row, kind, preference))
    return out


def _add_pattern(
    out: dict[tuple[str, str], list[_Candidate]],
    base: _Candidate,
    source: str | None,
    call: ast.Call,
) -> None:
    path: list[str] = []
    node: Any = call.func
    while isinstance(node, ast.Attribute):
        path.append(node.attr)
        node = node.value
    root = "bt" if node.id == "bt" else (_norm(source) if source else None)
    if root is None:
        return
    kind = "module" if node.id == "bt" else "method"
    candidate = _Candidate(base.row, kind, base.preference, call)
    out.setdefault((root, ".".join(reversed(path))), []).append(candidate)


def _module_base(t: Tables, surface: str) -> str | None:
    if surface in t.spec.modules:
        return t.spec.modules[surface]
    if t.spec.session and surface == t.spec.session[0]:
        return "spark"
    return None


class _Export(SiteRecorder):
    def __init__(self, t: Tables, scopes: dict[Any, Inference], module: Any) -> None:
        super().__init__(scopes, module, ENGINE_LABELS[t.spec.name])
        self.t = t
        self.index = _inverse(t.spec.name)

    def leave_Call(self, original: Any, updated: Any) -> Any:
        if self.lambdas:
            return self.keep(original, updated)
        func = original.func
        lookups: list[tuple[str, str, int]] = []
        if isinstance(func, cst.Attribute):
            receiver = self.inference.receiver(func.value)
            if receiver is not None:
                lookups.append((_norm(receiver), func.attr.value, 1))
            inner = func.value
            if isinstance(inner, cst.Attribute) and (base := self.inference.receiver(inner.value)):
                lookups.append((_norm(base), f"{inner.attr.value}.{func.attr.value}", 2))
        elif isinstance(func, cst.Name):
            bound = self.inference.scope.lookup(func.value)
            if isinstance(bound, Member) and bound.receiver == "bt" and bound.imported:
                lookups.append(("bt", bound.name, 0))
        if not lookups:
            return self.keep(original, updated)
        return self.keep(original, self._rewrite(original, updated, lookups))

    def _rewrite(self, original: Any, updated: Any, lookups: list[tuple[str, str, int]]) -> Any:
        spelling = f"{lookups[0][0]}.{lookups[0][1]}"
        for receiver, member, depth in lookups:
            candidates = sorted(
                self.index.get((receiver, member), []), key=lambda c: c.rank(member)
            )
            built = [(c, self._build(c, updated, depth)) for c in candidates]
            built = [(c, n) for c, n in built if n is not None]
            if not built:
                continue
            best = [b for b in built if b[0].rank(member) == built[0][0].rank(member)]
            if len(best) > 1:
                names = ", ".join(f"{c.row.surface}.{c.row.name}" for c, _ in best)
                self.mark(original, None, spelling, f"has several {self.label} spellings: {names}")
                return updated
            candidate, new = best[0]
            if depth == 2:
                self.consumed.add(id(original.func.value))
            detail = f"{candidate.row.surface}.{candidate.row.name}"
            self.record(original, spelling, candidate.row, "rewritten", detail, None)
            return new
        if lookups[0][0] in self.t.batcher_params or lookups[0][0] == "bt":
            self.mark(
                original, None, spelling, f"has no exact {self.label} spelling; left as written"
            )
        return updated

    def _build(self, c: _Candidate, updated: Any, depth: int) -> Any | None:
        name = c.row.name
        func = updated.func
        if c.kind == "method":
            base = func.value if depth <= 1 else func.value.value
            new_func = func.with_changes(value=base, attr=cst.Name(name))
            args = list(updated.args)
        elif c.kind == "module":
            base_code = _module_base(self.t, c.row.surface)
            if base_code is None:
                return None
            new_func = cst.Attribute(value=cst.parse_expression(base_code), attr=cst.Name(name))
            args = list(updated.args)
        else:
            base_code = self.t.spec.modules.get(c.row.surface)
            if base_code is None or depth == 0:
                return None
            receiver = func.value if depth == 1 else func.value.value
            new_func = cst.Attribute(value=cst.parse_expression(base_code), attr=cst.Name(name))
            args = [cst.Arg(receiver), *updated.args]
        if c.pattern is not None:
            args = _unbind(c, args)
            if args is None:
                return None
        tokens = self.t.params.get(c.row.surface, {}).get(name)
        if tokens is None or any(a.star for a in args):
            return None
        keywords = [a.keyword.value for a in args if a.keyword is not None]
        positional = sum(1 for a in args if a.keyword is None)
        if not Signature.parse(tokens).accepts(positional, keywords, None):
            return None
        return (
            simple_call(new_func, args)
            if c.pattern is not None
            else updated.with_changes(func=new_func, args=args)
        )


def _unbind(c: _Candidate, args: list[Any]) -> list[Any] | None:
    """Map a Batcher call's arguments back onto a reversible template's left side."""
    pattern = c.pattern
    assert pattern is not None
    template = parse_template(str(c.row.template), c.row.name)
    bound: dict[str, Any] = {}
    positional = [a for a in args if a.keyword is None]
    keywords = {a.keyword.value: a.value for a in args if a.keyword is not None}
    if len(positional) > len(pattern.args) or any(isinstance(p, ast.Starred) for p in pattern.args):
        return None
    for arg, slot in zip(positional, pattern.args, strict=False):
        if not isinstance(slot, ast.Name):
            return None
        bound[slot.id] = arg.value
    for keyword in pattern.keywords:
        if keyword.arg is None or not isinstance(keyword.value, ast.Name):
            return None
        if keyword.arg in keywords:
            bound[keyword.value.id] = keywords.pop(keyword.arg)
    if keywords:
        return None
    params = template.params
    out = []
    for param in [*params.posonlyargs, *params.args]:
        if param.arg in bound:
            out.append(cst.Arg(bound[param.arg]))
    for param in params.kwonlyargs:
        if param.arg in bound:
            out.append(cst.Arg(bound[param.arg], keyword=cst.Name(param.arg)))
    return out


def export(source: str, engine: str) -> tuple[str, Report]:
    """Rewrite one Batcher script onto a foreign engine, where the registry is exact.

    Args:
        source: Python source written against Batcher.
        engine: `pyspark`, `polars`, `daft` or `ray_data`.

    Returns:
        The rewritten source, and every site the rewrite touched or declined.

    Examples:
        .. doctest::

            >>> from batcher.migrate.outbound import export
            >>> code, _ = export('import batcher as bt\\ne = bt.col("a").alias("b")\\n', "polars")
            >>> print(code, end="")
            import polars as pl
            e = pl.col("a").alias("b")
    """
    t = tables(engine)
    wrapper = metadata.MetadataWrapper(cst.parse_module(source))
    module = wrapper.module
    scopes = infer_module(module, t.batcher_returns)
    transformer = _Export(t, scopes, module)
    rewritten = wrapper.visit(transformer)
    report, markers = transformer.report()
    return finish(rewritten, t.spec, markers, direction="outbound"), report
