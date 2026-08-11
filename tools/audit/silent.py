"""Failures that leave no trace, promises with no body, and near-copies of each other."""

from __future__ import annotations

import ast
import re
from collections import defaultdict
from collections.abc import Iterator

from tools.audit.context import (
    BROAD_TRY_NODES,
    DUP_SKIP_NAMES,
    MIN_STATEMENTS,
    NEAR_DUP_JACCARD,
    SILENT_ALLOW,
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
    """True when the handler neither reports nor re-raises — the failure just vanishes.

    A handler that *reads* its bound exception is also not silent: threading the error into a
    value the caller reports later (`reason = f"unavailable ({type(exc).__name__})"`) carries
    it just as well as logging it here, and the log statement is often deliberately placed
    outside the `try` so it fires on both the raised and the returned-None branch.
    """
    if handler.name and any(
        isinstance(n, ast.Name) and n.id == handler.name for n in ast.walk(handler)
    ):
        return False
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


#: A handler body made only of these is a *fallback*: it substitutes a default and moves on,
#: without inspecting, reporting, or re-raising anything. That is the shape a swallowed error
#: takes in this codebase — `return None`, `return {}`, `factor = None` — far more often than
#: a bare `pass`.
_FALLBACK_STATEMENTS = (ast.Pass, ast.Continue, ast.Return, ast.Assign, ast.AugAssign, ast.Break)

#: Phrases an author writes when *declaring* a path best-effort. This, not the handler's
#: syntax, is the signal that separates the 47 real findings from the ~150 legitimate
#: fallbacks: a broad `except Exception: return None` on an optional-dependency probe is fine,
#: while the same handler on a path the author has promised "must never break a query" is the
#: learned-stats loop going quietly dead. Declaring the path best-effort is what makes silence
#: on it dangerous, because nothing downstream will ever notice.
_BEST_EFFORT = re.compile(
    r"never break|best.?effort|must not (fail|break)|learning is|feedback must|degrade", re.I
)


def _is_fallback_only(handler: ast.ExceptHandler) -> bool:
    return all(isinstance(s, _FALLBACK_STATEMENTS) for s in handler.body)


def _handler_text(source: list[str], handler: ast.ExceptHandler) -> str:
    end = handler.end_lineno or handler.lineno
    return "\n".join(source[handler.lineno - 1 : min(end, len(source))])


def _has_comment(text: str) -> bool:
    """True when the handler carries a `#` comment explaining why it is silent.

    The house convention is that a deliberately-silent handler says so: `except ImportError:
    pass  # no torch -> fall through`. Treating the comment as the signal is what separates a
    documented control-flow fall-through from an error nobody decided to drop.
    """
    return "#" in text


def _owner_of(tree: ast.Module) -> dict[int, str]:
    """Map ``id(node)`` to the dotted name of the innermost function or class enclosing it.

    This is what lets `SILENT_ALLOW` be keyed by symbol instead of by line. Innermost wins,
    so a handler in a nested helper is keyed on the helper rather than on its parent — a
    waiver stays as narrow as the site it was written for.
    """
    owner: dict[int, str] = {}

    def walk(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = f"{prefix}.{child.name}" if prefix else child.name
                for sub in ast.walk(child):
                    owner[id(sub)] = name
                walk(child, name)
            else:
                walk(child, prefix)

    walk(tree, "")
    return owner


def detect_swallowed(ctx: Context) -> Iterator[Finding]:
    """`except` handlers that hide a failure instead of handling it.

    Three shapes, in descending order of how often they are real. The ordering matters: the
    first rule is the one that found 47 sites the other two were blind to, because it keys on
    *silence* rather than on the handler's syntax.

    * a handler the author has *declared* best-effort ("must never break a query", "learning
      is best-effort") that substitutes a fallback and leaves no trace. That declaration is
      what makes the silence dangerous: nothing downstream will ever notice, so the learned-
      stats loop can be dead for months and every gate stays green. `note_suppressed`
      (`_internal.logging`) exists for exactly this, and a handler that calls it is not
      reported. Keying on the declaration rather than on the handler's syntax is deliberate —
      the syntax-only version of this rule reported 190 sites, ~150 of them legitimate
      optional-dependency fallbacks.
    * a bare `pass` body with no explanatory comment. With a comment this is the documented
      fall-through idiom (an optional import, a parse that is meant to be tried and abandoned)
      and reporting it buries the real findings — calibrated against all 18 `pass` sites in the
      tree, every one of which turned out to be legitimate control flow.
    * a broad `except` wrapping a large `try` — it catches the bug you did not anticipate along
      with the error you did, and turns it into a plausible wrong answer. A blast-radius
      warning rather than a silence one, so it is reported last and lowest.
    """
    seen_keys: set[str] = set()
    for path, tree in ctx.modules.items():
        source = ctx.sources[path].split("\n")
        owner = _owner_of(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue
            span = sum(len(list(ast.walk(stmt))) for stmt in node.body)
            for handler in node.handlers:
                if not _handler_is_silent(handler):
                    continue
                key = f"{_rel(path)}::{owner.get(id(handler), '<module>')}"
                seen_keys.add(key)
                if key in SILENT_ALLOW:
                    continue
                broad = _is_broad(handler)
                text = _handler_text(source, handler)
                # `continue` is not silence: inside a loop it says "skip this entry", which is
                # the whole meaning of the handler. Only a bare `pass` leaves nothing behind.
                # Calibrated against all 16 `continue` sites in the tree — every one was a
                # legitimate skip-this-file / skip-this-line loop body.
                vanishes = all(isinstance(s, ast.Pass) for s in handler.body)
                if _is_fallback_only(handler) and _BEST_EFFORT.search(text):
                    yield Finding(
                        "swallowed-error",
                        "high",
                        _rel(path),
                        handler.lineno,
                        "declared best-effort but leaves no trace — a path that is allowed to "
                        "fail is not allowed to fail invisibly; call "
                        "note_suppressed(subsystem, step, exc)",
                    )
                elif vanishes and not _has_comment(text):
                    yield Finding(
                        "swallowed-error",
                        "high" if broad else "medium",
                        _rel(path),
                        handler.lineno,
                        f"{'broad' if broad else 'narrow'} except with a `pass` body and no "
                        "comment saying why — either trace it, or say what is being ignored",
                    )
                elif broad and span >= BROAD_TRY_NODES:
                    yield Finding(
                        "swallowed-error",
                        "low",
                        _rel(path),
                        handler.lineno,
                        f"broad except over a {span}-node try body — narrow the try to the "
                        "call that can actually fail, or catch the specific error",
                    )

    # A waiver that matches nothing is worse than no waiver: the site it was written for
    # comes back as an unexplained finding while the ledger still claims it is handled.
    # Report it rather than skipping it, so drift costs one line of output instead of a
    # re-investigation. This is the failure the line-numbered keys actually had.
    for key in sorted(set(SILENT_ALLOW) - seen_keys):
        rel, sep, symbol = key.partition("::")
        # A key with no `::` is the retired line-numbered form, which is exactly the drift
        # this reports; name the whole key rather than a symbol that parses out empty.
        target = f"`{symbol}`" if sep and symbol else f"`{key}` (not a file::symbol key)"
        yield Finding(
            "stale-waiver",
            "high",
            rel,
            0,
            f"SILENT_ALLOW waives {target} but no silent handler there matches it — the "
            "code moved or was fixed; re-key the entry or delete it",
        )


def _effective_body(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.stmt]:
    body = list(node.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]
    return body


def _overridden_hooks(ctx: Context) -> dict[str, set[str]]:
    """Map each base-class name to the methods a subclass of it implements for real.

    A base method with a trivial body that a subclass overloads is the template-method
    pattern, not an unkept promise: `CountVectorizer._weights` returns None because plain
    counts have no per-feature multiplier, and `TfidfVectorizer` returns the IDF vector.
    Its sibling hooks were already excused for returning `-> str | None`, so flagging this
    one turned on the annotation (`-> Any`) rather than on anything about the code.

    Bases are matched by *name*, which is all that is available without resolving imports.
    That can over-match two unrelated classes sharing a base name; the population it filters
    is documented-single-trivial-statement methods, which is small enough that the trade is
    worth the false negative it risks.
    """
    subclasses: dict[str, set[str]] = defaultdict(set)
    for tree in ctx.modules.values():
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            real = {
                m.name
                for m in node.body
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
                and len(_effective_body(m)) > 1
            }
            if not real:
                continue
            for base in node.bases:
                name = getattr(base, "id", getattr(base, "attr", ""))
                if name:
                    subclasses[name] |= real
    return subclasses


def detect_stub(ctx: Context) -> Iterator[Finding]:
    """Documented functions whose body does nothing — a promise with no implementation."""
    hooks = _overridden_hooks(ctx)
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
                if isinstance(parent, ast.ClassDef) and node.name in hooks.get(parent.name, ()):
                    continue  # a base hook a subclass implements for real
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
