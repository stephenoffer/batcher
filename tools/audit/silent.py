"""Failures that leave no trace, promises with no body, and near-copies of each other."""

from __future__ import annotations

import ast
from collections import defaultdict
from collections.abc import Iterator

from tools.audit.context import (
    BROAD_TRY_NODES,
    DUP_SKIP_NAMES,
    MIN_STATEMENTS,
    NEAR_DUP_JACCARD,
    Context,
    Finding,
    _decorator_names,
    _rel,
)

# --- Silent failure ---------------------------------------------------------------

#: Calls that leave a trace, so a handler containing one is handling rather than hiding.
#: `note_suppressed` is the project's own best-effort tracer (`_internal.logging`) and has to
#: be here: a detector that does not recognize its own sanctioned fix keeps reporting the
#: sites it already got fixed, which is how a report loses its reader.
_SIGNAL = (
    "log",
    "warn",
    "error",
    "debug",
    "info",
    "emit",
    "record",
    "raise",
    "print",
    "publish",
    "note_suppressed",
)


def _handler_is_silent(handler: ast.ExceptHandler) -> bool:
    """True when the handler neither reports nor re-raises — the failure just vanishes."""
    for node in ast.walk(handler):
        if isinstance(node, ast.Raise):
            return False
        name = ""
        if isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if any(hint in name.lower() for hint in _SIGNAL):
            return False
    return True


def _is_broad(handler: ast.ExceptHandler) -> bool:
    if handler.type is None:
        return True
    return isinstance(handler.type, ast.Name) and handler.type.id in {"Exception", "BaseException"}


def detect_swallowed(ctx: Context) -> Iterator[Finding]:
    """`except` handlers that hide a failure instead of handling it.

    Deliberately reports only the two shapes that are wrong regardless of intent, because a
    broad `except` returning a documented fallback (an optional dependency probe, a source
    that cannot describe itself) is a legitimate idiom here and flagging it buries the real
    findings:

    * the handler body is exactly `pass` or `continue` — the failure leaves no trace at all;
    * a broad `except` wraps a large `try` — it catches the bug you did not anticipate along
      with the error you did, and turns it into a plausible wrong answer.
    """
    for path, tree in ctx.modules.items():
        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue
            span = sum(len(list(ast.walk(stmt))) for stmt in node.body)
            for handler in node.handlers:
                if not _handler_is_silent(handler):
                    continue
                broad = _is_broad(handler)
                vanishes = all(isinstance(s, ast.Pass | ast.Continue) for s in handler.body)
                if vanishes:
                    yield Finding(
                        "swallowed-error",
                        "high" if broad else "medium",
                        _rel(path),
                        handler.lineno,
                        f"{'broad' if broad else 'narrow'} except with a `pass` body — the "
                        "failure leaves no trace; log it or record it on the event bus",
                    )
                elif broad and span >= BROAD_TRY_NODES:
                    yield Finding(
                        "swallowed-error",
                        "medium",
                        _rel(path),
                        handler.lineno,
                        f"broad except over a {span}-node try body — narrow the try to the "
                        "call that can actually fail, or catch the specific error",
                    )


def _effective_body(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.stmt]:
    body = list(node.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]
    return body


def detect_stub(ctx: Context) -> Iterator[Finding]:
    """Documented functions whose body does nothing — a promise with no implementation."""
    for path, tree in ctx.modules.items():
        for parent in ast.walk(tree):
            if not isinstance(parent, ast.Module | ast.ClassDef):
                continue
            abstract = any(
                getattr(b, "id", getattr(b, "attr", "")) in {"Protocol", "ABC"}
                for b in getattr(parent, "bases", [])
            )
            for node in parent.body:
                if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                    continue
                if abstract or "abstractmethod" in _decorator_names(node):
                    continue
                if not ast.get_docstring(node):
                    continue
                body = _effective_body(node)
                if len(body) != 1:
                    continue
                stmt = body[0]
                # `-> X | None` returning None is a documented "unknown" default, not a
                # stub: the base `LogicalPlan.available_schema` answers "not inferable"
                # that way on purpose, and subclasses override it. Only an unannotated or
                # non-optional return makes a bare `return None` a broken promise.
                optional = "None" in ast.unparse(node.returns) if node.returns else False
                trivial = isinstance(stmt, ast.Pass) or (
                    not optional
                    and isinstance(stmt, ast.Return)
                    and (stmt.value is None or _is_none(stmt.value))
                )
                if trivial:
                    yield Finding(
                        "stub",
                        "medium",
                        _rel(path),
                        node.lineno,
                        f"`{node.name}` has a docstring but a do-nothing body",
                    )


def _is_none(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


# --- Near-duplicates --------------------------------------------------------------


class _Normalize(ast.NodeTransformer):
    """Erase every name, attribute, and constant so only the code's shape survives."""

    def visit_Name(self, node: ast.Name) -> ast.Name:
        return ast.copy_location(ast.Name(id="_", ctx=node.ctx), node)

    def visit_Attribute(self, node: ast.Attribute) -> ast.Attribute:
        self.generic_visit(node)
        return ast.copy_location(ast.Attribute(value=node.value, attr="_", ctx=node.ctx), node)

    def visit_arg(self, node: ast.arg) -> ast.arg:
        return ast.copy_location(ast.arg(arg="_", annotation=None), node)

    def visit_Constant(self, node: ast.Constant) -> ast.Constant:
        return ast.copy_location(ast.Constant(value="_"), node)


def _statement_shapes(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str] | None:
    body = _effective_body(node)
    if len(body) < MIN_STATEMENTS:
        return None
    shapes = set()
    for stmt in body:
        try:
            shapes.add(ast.dump(_Normalize().visit(ast.parse(ast.unparse(stmt)))))
        except (SyntaxError, ValueError, RecursionError):
            return None
    return shapes if len(shapes) >= MIN_STATEMENTS - 1 else None


def detect_near_duplicate(ctx: Context) -> Iterator[Finding]:
    """Cross-file function pairs whose bodies overlap almost completely.

    `lint_duplication` hashes the whole normalized body, so it sees exact copies only. The
    copy that drifted by one line is the more common shape here, and it is invisible to an
    exact hash.
    """
    functions: list[tuple[str, int, str, set[str]]] = []
    for path, tree in ctx.modules.items():
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            if node.name in DUP_SKIP_NAMES:
                continue
            shapes = _statement_shapes(node)
            if shapes:
                functions.append((_rel(path), node.lineno, node.name, shapes))

    index: dict[str, list[int]] = defaultdict(list)
    for i, (_, _, _, shapes) in enumerate(functions):
        for shape in shapes:
            index[shape].append(i)

    seen: set[tuple[int, int]] = set()
    for i, (rel, line, name, shapes) in enumerate(functions):
        overlap: dict[int, int] = defaultdict(int)
        for shape in shapes:
            for j in index[shape]:
                if j != i:
                    overlap[j] += 1
        for j, shared in overlap.items():
            other_rel, other_line, other_name, other_shapes = functions[j]
            if other_rel == rel or (min(i, j), max(i, j)) in seen:
                continue
            jaccard = shared / len(shapes | other_shapes)
            if jaccard < NEAR_DUP_JACCARD:
                continue
            seen.add((min(i, j), max(i, j)))
            yield Finding(
                "near-duplicate",
                "medium",
                rel,
                line,
                f"`{name}` is {jaccard:.0%} the same code as `{other_name}` at "
                f"{other_rel}:{other_line} — lift the shared part into a neutral layer",
            )
