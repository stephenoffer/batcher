"""Rewrite Batcher's own removed second spellings to the one spelling that stays.

This is the rule set the alias removal runs over the repository before the removed names are
deleted, and that a user runs over their code after upgrading: `ds.groupby("k")` becomes
`ds.group_by("k")`, `bt.col("s").str.to_lowercase()` becomes `bt.col("s").str.lower()`, and
`from batcher import from_dict` becomes `from batcher import from_pydict` together with every
use of the imported name. The decisions come from `renames.toml`; which expressions are
Batcher objects comes from `receivers`.

A removed spelling on an expression whose receiver cannot be inferred is left alone and
reported. That is the common case in a test comparing against pandas or Polars, where the
same word on a different library's object must not change.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from batcher._internal.optional import require
from batcher.migrate.receivers import ClassRef, Inference, Member, infer_module

cst = require("libcst", feature="batcher.migrate", provides="libcst", extra="migrate")
metadata = require("libcst.metadata", feature="batcher.migrate", provides="libcst", extra="migrate")

__all__ = ["Edit", "RenameReport", "canonicalize"]

_EXPR_RECEIVERS = ("AggExpr", "WindowExpr")
_BUILTIN_METHODS = frozenset(
    name for kind in (str, bytes, list, dict, set) for name in dir(kind) if not name.startswith("_")
)


@dataclass(frozen=True)
class Edit:
    """One site the rewrite touched or declined."""

    line: int
    old: str
    new: str | None
    receiver: str | None


@dataclass
class RenameReport:
    """Every site renamed, and every removed spelling left alone for want of a receiver."""

    renamed: list[Edit] = field(default_factory=list)
    unresolved: list[Edit] = field(default_factory=list)


def _kept(renames: dict[str, dict[str, str]], receiver: str, name: str) -> str | None:
    found = renames.get(receiver, {}).get(name)
    if found is None and receiver in _EXPR_RECEIVERS:
        found = renames.get("Expr", {}).get(name)
    return found


class _Canonicalize(cst.CSTTransformer):  # type: ignore[misc]
    METADATA_DEPENDENCIES = (metadata.PositionProvider,)

    def __init__(
        self,
        scopes: dict[object, Inference],
        module: object,
        renames: dict[str, dict[str, str]],
        report: RenameReport,
    ) -> None:
        super().__init__()
        self.scopes = scopes
        self.stack: list[object] = [module]
        self.renames = renames
        self.report = report
        self.removed = {name for table in renames.values() for name in table}
        self.attr_names: set[int] = set()
        # An unknown receiver is only worth reporting in a file that uses Batcher at all, and
        # only for a word that is not also a method of Python's own containers: `list.append`
        # and `str.strip` would otherwise bury the handful of sites that need a person.
        top = scopes[module].scope.names.values()
        self.uses_batcher = any(v == "bt" or isinstance(v, (Member, ClassRef)) for v in top)

    @property
    def inference(self) -> Inference:
        return self.scopes[self.stack[-1]]

    def _line(self, node: object) -> int:
        return self.get_metadata(metadata.PositionProvider, node).start.line

    def visit_FunctionDef(self, node: object) -> None:
        self.stack.append(node)

    def leave_FunctionDef(self, _original: object, updated: object) -> object:
        self.stack.pop()
        return updated

    def visit_Attribute(self, node: object) -> None:
        self.attr_names.add(id(node.attr))  # type: ignore[attr-defined]

    def leave_Attribute(self, original: object, updated: object) -> object:
        name = original.attr.value  # type: ignore[attr-defined]
        if name not in self.removed:
            return updated
        receiver = self.inference.receiver(original.value)  # type: ignore[attr-defined]
        kept = _kept(self.renames, receiver, name) if receiver else None
        line = self._line(original)
        if kept is None:
            if receiver is None and self.uses_batcher and name not in _BUILTIN_METHODS:
                self.report.unresolved.append(Edit(line, name, None, None))
            return updated
        self.report.renamed.append(Edit(line, name, kept, receiver))
        return updated.with_changes(attr=cst.Name(kept))  # type: ignore[attr-defined]

    def leave_ImportFrom(self, original: object, updated: object) -> object:
        module = original.module  # type: ignore[attr-defined]
        if not (isinstance(module, cst.Name) and module.value == "batcher"):
            return updated
        if isinstance(updated.names, cst.ImportStar):  # type: ignore[attr-defined]
            return updated
        names = []
        for alias in updated.names:  # type: ignore[attr-defined]
            old = alias.name.value
            kept = self.renames.get("bt", {}).get(old)
            if kept is not None:
                self.report.renamed.append(Edit(self._line(original), old, kept, "bt"))
                alias = alias.with_changes(name=cst.Name(kept))
            names.append(alias)
        return updated.with_changes(names=names)

    def leave_Name(self, original: object, updated: object) -> object:
        if id(original) in self.attr_names:
            return updated
        bound = self.inference.scope.lookup(original.value)  # type: ignore[attr-defined]
        if isinstance(bound, Member) and bound.receiver == "bt":
            kept = self.renames.get("bt", {}).get(bound.name)
            # Rename uses of an un-aliased import: `from batcher import from_dict` binds
            # `from_dict`, and the import itself is rewritten in `leave_ImportFrom`.
            if kept is not None and original.value == bound.name:  # type: ignore[attr-defined]
                return updated.with_changes(value=kept)  # type: ignore[attr-defined]
        return updated


def canonicalize(
    source: str,
    renames: dict[str, dict[str, str]],
    returns: dict[str, dict[str, str]],
) -> tuple[str, RenameReport]:
    """Rewrite removed Batcher spellings in one source file.

    Args:
        source: Python source text.
        renames: `{receiver: {removed: kept}}`, from `renames.toml`.
        returns: `{receiver: {member: returned_receiver}}`, from `returns.toml`.

    Returns:
        The rewritten source, and a report of what changed and what was left alone.
    """
    wrapper = metadata.MetadataWrapper(cst.parse_module(source))
    scopes = infer_module(wrapper.module, returns)
    report = RenameReport()
    transformer = _Canonicalize(scopes, wrapper.module, renames, report)
    rewritten = wrapper.visit(transformer)
    return rewritten.code, report
