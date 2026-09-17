"""What both foreign-engine directions share: the site record, markers, and finishing a module.

A rewrite in either direction (`translate` onto Batcher, `export` off it) walks a module with a
`SiteRecorder`. Each call or property it looks at becomes a `Site` in a `Report`, and a site left
alone (or rewritten with a caveat) carries a `# batcher-migrate:` marker. Sites are recorded
against the statement that holds them but attached only at the end, because a later rewrite can
still absorb an earlier site: a PySpark `Window` spec assigned on one line and inlined into
`.over(...)` on another.

`finish` then makes the module consistent with its rewritten expressions. It attaches the
markers, drops an assignment the rewrite made dead (the inlined `Window` spec, a session no call
needs any more) and settles the imports: an engine import goes once nothing uses a name it binds,
and the other side's import is added where the first import stood. An import some left-alone
call still needs stays, so a partly migrated file keeps running.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from batcher._internal.migration import Mapping
from batcher._internal.optional import require
from batcher.migrate.engines import EngineSpec

cst = require("libcst", feature="batcher.migrate", provides="libcst", extra="migrate")
metadata = require("libcst.metadata", feature="batcher.migrate", provides="libcst", extra="migrate")

__all__ = ["MARKER", "Report", "Site", "SiteRecorder", "finish", "referenced_names"]

MARKER = "# batcher-migrate:"


@dataclass(frozen=True)
class Site:
    """One call or property a rewrite looked at.

    Attributes:
        line: The 1-based line in the source.
        spelling: `<receiver>.<name>` in the source's terms.
        status: The registry status, or `None` when the name has no row.
        action: `rewritten`, `marked` (left alone with a marker), or `rewritten+marked`.
        detail: The spelling written, or why the site was left alone.
    """

    line: int
    spelling: str
    status: str | None
    action: str
    detail: str


@dataclass
class Report:
    """Every site of one file, in source order."""

    sites: list[Site] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        """Sites per action.

        Returns:
            `{"rewritten": n, "marked": n, "rewritten+marked": n}`, absent actions omitted.
        """
        out: dict[str, int] = {}
        for site in self.sites:
            out[site.action] = out.get(site.action, 0) + 1
        return out


class SiteRecorder(cst.CSTTransformer):  # type: ignore[misc]
    """A transformer that tracks scopes and statements and records sites against them."""

    METADATA_DEPENDENCIES = (metadata.PositionProvider,)

    def __init__(self, scopes: dict[Any, Any], module: Any, label: str) -> None:
        super().__init__()
        self.scopes = scopes
        self.label = label
        self.stack: list[Any] = [module]
        self.rewritten: dict[int, Any] = {}
        self.consumed: set[int] = set()
        self.call_funcs: set[int] = set()
        self.frames: list[list[int]] = []
        self.sites: dict[int, list[tuple[Site, str | None]]] = {}
        self.statements: list[tuple[Any, list[int]]] = []
        # Code inside a lambda runs on whatever the enclosing call hands it (a foreign frame in
        # a UDF), so the rules leave it exactly as written.
        self.lambdas = 0

    @property
    def inference(self) -> Any:
        """The inference for the innermost enclosing scope."""
        return self.scopes[self.stack[-1]]

    def on_visit(self, node: Any) -> bool:
        if isinstance(node, (cst.SimpleStatementLine, cst.BaseCompoundStatement)):
            self.frames.append([])
        if isinstance(node, cst.FunctionDef):
            self.stack.append(node)
        if isinstance(node, cst.Lambda):
            self.lambdas += 1
        if isinstance(node, cst.Call):
            self.call_funcs.add(id(node.func))
        return super().on_visit(node)

    def on_leave(self, original: Any, updated: Any) -> Any:
        result = super().on_leave(original, updated)
        if isinstance(original, cst.Lambda):
            self.lambdas -= 1
        if isinstance(original, cst.FunctionDef):
            self.stack.pop()
        if isinstance(original, (cst.SimpleStatementLine, cst.BaseCompoundStatement)):
            self.statements.append((result, self.frames.pop()))
        return result

    def record(
        self,
        node: Any,
        spelling: str,
        row: Mapping | None,
        action: str,
        detail: str,
        marker: str | None,
    ) -> None:
        """Record one site, and the marker it leaves when it has one.

        Args:
            node: The original node.
            spelling: How the source spells it.
            row: The registry row that decided it, if any.
            action: `rewritten`, `marked` or `rewritten+marked`.
            detail: The spelling written, or why the site was left alone.
            marker: The marker text, or `None`.
        """
        line = self.get_metadata(metadata.PositionProvider, node).start.line
        status = row.status.value if row is not None else None
        self.sites.setdefault(id(node), []).append(
            (Site(line, spelling, status, action, detail), marker)
        )
        if self.frames:
            self.frames[-1].append(id(node))

    def mark(self, node: Any, row: Mapping | None, spelling: str, why: str) -> None:
        """Record a site left as written, with a marker saying why.

        Args:
            node: The original node.
            row: The registry row, if any.
            spelling: How the source spells it.
            why: The reason, which follows the spelling in the marker.
        """
        self.record(node, spelling, row, "marked", why, f"{self.label} `{spelling}` {why}")

    def keep(self, original: Any, result: Any) -> Any:
        """Remember what an original node became, keeping its own parentheses.

        Args:
            original: The original node.
            result: Its replacement (or its updated self).

        Returns:
            The result, parenthesized like the original was.
        """
        lpar = getattr(original, "lpar", None)
        if lpar and result is not original and hasattr(result, "lpar") and not result.lpar:
            result = result.with_changes(lpar=original.lpar, rpar=original.rpar)
        self.rewritten[id(original)] = result
        return result

    def report(self) -> tuple[Report, dict[int, list[str]]]:
        """The sites not absorbed by a later rewrite, and the markers by statement.

        Returns:
            The report, and marker texts keyed by the `id` of the rewritten statement.
        """
        report = Report()
        markers: dict[int, list[str]] = {}
        for statement, ids in self.statements:
            for node_id in ids:
                if node_id in self.consumed:
                    continue
                for site, marker in self.sites.get(node_id, []):
                    if site not in report.sites:
                        report.sites.append(site)
                    if marker:
                        markers.setdefault(id(statement), []).append(marker)
        report.sites.sort(key=lambda s: s.line)
        return report, markers


def referenced_names(module: Any) -> dict[str, int]:
    """How often each bare name is read outside import statements.

    Args:
        module: A libcst module.

    Returns:
        `{name: count}`, not counting attribute names, keyword names or assignment targets.
    """
    counts: dict[str, int] = {}

    class _Count(cst.CSTVisitor):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.labels: set[int] = set()

        def visit_Import(self, _node: Any) -> bool:
            return False

        def visit_ImportFrom(self, _node: Any) -> bool:
            return False

        def visit_Attribute(self, node: Any) -> None:
            self.labels.add(id(node.attr))

        def visit_Arg(self, node: Any) -> None:
            if node.keyword is not None:
                self.labels.add(id(node.keyword))

        def visit_AssignTarget(self, node: Any) -> None:
            if isinstance(node.target, cst.Name):
                self.labels.add(id(node.target))

        def visit_Name(self, node: Any) -> None:
            if id(node) not in self.labels:
                counts[node.value] = counts.get(node.value, 0) + 1

    module.visit(_Count())
    return counts


def _code(node: Any) -> str:
    return cst.Module([]).code_for_node(node)


def _bound_names(stmt: Any) -> list[str]:
    if isinstance(stmt.names, cst.ImportStar):
        return ["*"]
    names = []
    for alias in stmt.names:
        if alias.asname is not None and isinstance(alias.asname.name, cst.Name):
            names.append(alias.asname.name.value)
        else:
            names.append(_code(alias.name).split(".")[0])
    return names


def _package(stmt: Any) -> str | None:
    if isinstance(stmt, cst.Import):
        return _code(stmt.names[0].name).split(".")[0]
    if isinstance(stmt, cst.ImportFrom) and stmt.module is not None:
        return _code(stmt.module).split(".")[0]
    return None


class _Finish(cst.CSTTransformer):  # type: ignore[misc]
    def __init__(self, package: str, markers: dict[int, list[str]], counts: dict[str, int]):
        super().__init__()
        self.package = package
        self.markers = markers
        self.counts = counts

    def leave_Module(self, _original: Any, updated: Any) -> Any:
        return updated.with_changes(body=self._settle(updated.body))

    def leave_IndentedBlock(self, _original: Any, updated: Any) -> Any:
        return updated.with_changes(
            body=self._settle(updated.body) or [cst.parse_statement("pass")]
        )

    def _settle(self, body: Any) -> list[Any]:
        """Drop dead statements, moving their blank lines onto the next statement kept."""
        out: list[Any] = []
        carried: list[Any] = []
        for stmt in body:
            if isinstance(stmt, cst.SimpleStatementLine) and self._removable(stmt):
                carried.extend(stmt.leading_lines)
                continue
            if carried:
                stmt = stmt.with_changes(leading_lines=[*carried, *stmt.leading_lines])
                carried = []
            out.append(stmt)
        return out

    def on_leave(self, original: Any, updated: Any) -> Any:
        result = super().on_leave(original, updated)
        texts = self.markers.get(id(original), [])
        if not texts or not isinstance(
            result, (cst.SimpleStatementLine, cst.BaseCompoundStatement)
        ):
            return result
        existing = {line.comment.value for line in result.leading_lines if line.comment}
        lines = list(result.leading_lines)
        for text in dict.fromkeys(texts):
            comment = f"{MARKER} {' '.join(text.split())}"
            if comment not in existing:
                lines.append(cst.EmptyLine(comment=cst.Comment(comment)))
        return result.with_changes(leading_lines=lines)

    def _removable(self, line: Any) -> bool:
        stmt = line.body[0] if len(line.body) == 1 else None
        if _package(stmt) == self.package:
            return all(self.counts.get(n, 0) == 0 for n in _bound_names(stmt))
        if not (isinstance(stmt, cst.Assign) and len(stmt.targets) == 1):
            return False
        target = stmt.targets[0].target
        if not isinstance(target, cst.Name) or self.counts.get(target.value, 0):
            return False
        code = _code(stmt.value)
        return code == "bt.Session()" or code.startswith("Window.")


def finish(module: Any, spec: EngineSpec, markers: dict[int, list[str]], direction: str) -> str:
    """Attach markers, drop dead bindings and settle the imports of a rewritten module.

    Args:
        module: The rewritten libcst module.
        spec: The foreign engine.
        markers: Marker texts by the `id` of the (rewritten) statement they belong to.
        direction: `inbound` (foreign to Batcher) or `outbound` (Batcher to foreign).

    Returns:
        The finished source.
    """
    package = spec.seeds.package if direction == "inbound" else "batcher"
    finished = module.visit(_Finish(package, markers, referenced_names(module)))
    # A dead binding removed above may have been the last user of an import; settle again.
    finished = finished.visit(_Finish(package, {}, referenced_names(finished)))
    return _add_imports(finished, spec, direction)


def _wanted_imports(module: Any, spec: EngineSpec, direction: str) -> list[str]:
    counts = referenced_names(module)
    code = module.code
    if direction == "inbound":
        return (
            ["import batcher as bt"]
            if counts.get("bt") and "import batcher as bt" not in code
            else []
        )
    wanted = []
    for line in spec.imports:
        alias = line.split(" as ")[-1] if " as " in line else line.split()[-1].split(".")[0]
        if counts.get(alias) and line not in code:
            wanted.append(line)
    if spec.session and counts.get("spark") and "spark =" not in code:
        wanted.extend([f"from pyspark.sql import {spec.session[0]}", spec.session[1]])
    return wanted


def _add_imports(module: Any, spec: EngineSpec, direction: str) -> str:
    wanted = _wanted_imports(module, spec, direction)
    if not wanted:
        return module.code
    body = list(module.body)
    index = 0
    for i, stmt in enumerate(body):
        small = stmt.body[0] if isinstance(stmt, cst.SimpleStatementLine) else None
        if isinstance(small, (cst.Import, cst.ImportFrom)):
            index = i + 1
        elif i == 0 and isinstance(small, cst.Expr) and isinstance(small.value, cst.SimpleString):
            index = 1  # after a module docstring
        else:
            break
    new = [cst.parse_statement(line + "\n") for line in wanted]
    if index < len(body) and index > 0 and not body[index].leading_lines:
        body[index] = body[index].with_changes(leading_lines=[cst.EmptyLine()])
    return module.with_changes(body=[*body[:index], *new, *body[index:]]).code
