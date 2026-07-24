"""Library code that is not ready to ship, and the lint suppressions hiding what is.

Every rule here is deliberately narrow, because the broad version of it fired on nothing but
false positives in this codebase: every `print` sat in a `show`/`glimpse`/console sink whose
contract *is* printing, and every `assert` was a type-narrowing invariant with a comment
saying so, which is what `assert` is for. Widening these back is easy and wrong — a report
that is mostly noise teaches its reader to skip it.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator

from tools.audit.context import Context, Finding, _rel

_SUPPRESSION = re.compile(r"#\s*(noqa|type:\s*ignore)|#\[allow\(")
#: Functions whose contract *is* writing to stdout. `bt.show_versions`, `ds.info()`,
#: `ds.glimpse()`, and the `console` streaming sink all print on purpose.
_DISPLAY_NAMES = ("show", "print", "glimpse", "preview", "display", "console", "render")

_Func = ast.FunctionDef | ast.AsyncFunctionDef


def _is_display(fn: _Func, owner: str) -> bool:
    """Whether printing *is* this function's contract (`show`, `glimpse`, a console sink).

    `owner` is the enclosing class name, because `ConsoleStreamSink.write_batch` prints by
    contract while the method name alone says nothing.
    """
    if any(word in f"{owner}.{fn.name}".lower() for word in _DISPLAY_NAMES):
        return True
    doc = (ast.get_docstring(fn) or "").lstrip()
    return doc.split(" ")[0].rstrip(".,") in {"Print", "Show", "Display"}


def _is_narrowing(test: ast.expr) -> bool:
    """Whether an `assert` states a type/None invariant for the type checker.

    `assert isinstance(node, MapBatches)` and `assert value is not None  # narrowed above`
    are not input validation — they are the assertion's correct use, and reporting them
    buries the one case that matters.
    """
    if isinstance(test, ast.Call) and getattr(test.func, "id", "") == "isinstance":
        return True
    if isinstance(test, ast.Compare):
        return all(isinstance(op, ast.Is | ast.IsNot) for op in test.ops)
    if isinstance(test, ast.BoolOp):
        return all(_is_narrowing(v) for v in test.values)
    return False


def _param_names(fn: _Func) -> set[str]:
    args = fn.args
    return {a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)}


def _class_owners(tree: ast.Module) -> dict[int, str]:
    """Each method's enclosing class name, keyed by node identity."""
    return {
        id(fn): parent.name
        for parent in ast.walk(tree)
        if isinstance(parent, ast.ClassDef)
        for fn in parent.body
        if isinstance(fn, _Func)
    }


def _function_findings(fn: _Func, rel: str, owner: str) -> Iterator[Finding]:
    """Printing, argument-validating asserts, and mutable defaults in one function."""
    display = _is_display(fn, owner)
    params = _param_names(fn)
    for node in ast.walk(fn):
        if not display and isinstance(node, ast.Call) and getattr(node.func, "id", "") == "print":
            yield Finding(
                "production",
                "medium",
                rel,
                node.lineno,
                f"`print` inside `{fn.name}`, whose contract is not display — route it "
                "through logging or the event bus",
            )
        # An `assert` guarding an *argument* of a public function is input validation
        # wearing the wrong clothes: `python -O` deletes it and the bad input flows on.
        # An assert over locals is an internal invariant, which is its correct use.
        if (
            isinstance(node, ast.Assert)
            and not fn.name.startswith("_")
            and not _is_narrowing(node.test)
            and {n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)} & params
        ):
            yield Finding(
                "production",
                "medium",
                rel,
                node.lineno,
                f"`assert` validates an argument of public `{fn.name}` — `python -O` "
                "removes it; raise a typed error from `_internal.errors` instead",
            )
    for default in fn.args.defaults + [d for d in fn.args.kw_defaults if d]:
        if isinstance(default, ast.List | ast.Dict | ast.Set):
            yield Finding(
                "production",
                "medium",
                rel,
                fn.lineno,
                f"`{fn.name}` has a mutable default argument",
            )


def _suppression_findings(ctx: Context) -> Iterator[Finding]:
    """Every `# noqa` / `# type: ignore` / `#[allow(...)]` in the shipped source.

    Individually unremarkable; the *count over time* is the signal, which is why these are
    reported at `low` rather than argued about one at a time.
    """
    for path in sorted(list(ctx.modules) + list(ctx.rust_text)):
        rel = _rel(path)
        if not (rel.startswith("python/") or rel.startswith("crates/")):
            continue
        for i, line in enumerate(path.read_text().splitlines(), 1):
            if _SUPPRESSION.search(line):
                yield Finding(
                    "suppression", "low", rel, i, f"lint suppression: {line.strip()[:90]}"
                )


def detect_production(ctx: Context) -> Iterator[Finding]:
    """Printing, `assert` as validation, mutable defaults, and suppression debt."""
    for path, tree in ctx.modules.items():
        rel = _rel(path)
        owners = _class_owners(tree)
        for fn in ast.walk(tree):
            if isinstance(fn, _Func):
                yield from _function_findings(fn, rel, owners.get(id(fn), ""))
    yield from _suppression_findings(ctx)
