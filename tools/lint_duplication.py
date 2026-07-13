#!/usr/bin/env python3
"""Fail on copy-pasted logic in the Python control plane.

`.claude/rules/python-quality.md` says "no duplication — share via the neutral layers". That
was prose, and prose does not fail a build. This makes it a gate.

Why it matters more here than in most codebases: `kyber`, `carbonite`, `core`, and `governance`
are forbidden from importing one another (the independence contract). That is deliberate — but
it means **copy-paste is the only way to share between them wrongly**, and it is exactly what
happened: `_median` was pasted into `kyber` twice and `carbonite` once, and a fourth spelling of
the same idea sat in `carbonite/resilience`. Nothing caught it, because nothing was looking.

Method: parse every module, normalize each function body (erase names, attributes, constants,
and annotations), and hash it. Two functions with the same normalized body are the same code
wearing different variable names. Trivial and boilerplate shapes are excluded so the signal
stays high — this reports copied *logic*, not coincidence.

Exceptions go in `DUPLICATION_ALLOW` with a one-line reason, the same way `lint_structure.py`
handles its allowlist: visible, justified, and shrinking.
"""

from __future__ import annotations

import ast
import hashlib
import pathlib
import sys
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent
TARGETS = [ROOT / "python" / "batcher"]

#: A body must have at least this many statements to be worth reporting. Below it, two
#: functions matching is usually coincidence (`return self._x`, a two-line guard), not a copy.
MIN_STATEMENTS = 4

#: ...and its normalized form must be at least this "big", which filters the many short
#: bodies that normalize to the same tiny shape (a bare return, a single call).
MIN_SIGNATURE_CHARS = 350

#: Dunder and property-ish methods are structurally identical across unrelated classes by
#: design (`__init__` assigning its args, `__eq__` comparing fields). Not duplication.
SKIP_NAMES = {"__init__", "__eq__", "__hash__", "__repr__", "__str__", "__enter__", "__exit__"}

#: Known duplicates, each with a reason. Keyed by the sorted "file:line" list of its sites.
#: This is a *ledger*, not an amnesty: an entry here is debt that is visible and expected to
#: shrink, not a duplicate that has been blessed. Prefer fixing to listing.
DUPLICATION_ALLOW: dict[str, str] = {
    # TRACKED DEBT (not justified): the speculative-relaunch closure pair `_launch`/`_relaunch`
    # is copy-pasted between the Flight sort and the Flight window reducers. They differ only in
    # which actor method they call (`sort_reduce` vs `reduce_window`) and the IR they pass, so
    # the fix is a shared launcher in `dist/executors/ray_runtime` taking the remote call as a
    # parameter — next to `gather_with_backups`, which is already shared. It is listed rather
    # than fixed because it is Ray actor code with no runnable test in this environment, and a
    # blind refactor of the straggler/backup path is a worse risk than the duplication.
    "python/batcher/dist/flight_sort.py:332,python/batcher/dist/flight_window.py:161": (
        "tracked debt: speculative-relaunch closure duplicated; fix = shared launcher in "
        "dist/executors/ray_runtime beside gather_with_backups"
    ),
}


class _Normalize(ast.NodeTransformer):
    """Erase identifiers and literals so only the *shape* of the logic survives."""

    def visit_Name(self, node: ast.Name) -> ast.Name:
        return ast.copy_location(ast.Name(id="_", ctx=node.ctx), node)

    def visit_arg(self, node: ast.arg) -> ast.arg:
        node.arg = "_"
        node.annotation = None
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.Attribute:
        self.generic_visit(node)
        return ast.copy_location(
            ast.Attribute(value=node.value, attr="_", ctx=node.ctx), node
        )

    def visit_Constant(self, node: ast.Constant) -> ast.Constant:
        return ast.copy_location(ast.Constant(value="_"), node)


def _body_signature(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> str | None:
    """The normalized shape of `fn`'s body, or None if it is too small to judge."""
    body = [
        n
        for n in fn.body
        if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))  # docstring
    ]
    if len(body) < MIN_STATEMENTS:
        return None
    try:
        module = ast.parse(ast.unparse(ast.Module(body=body, type_ignores=[])))
        signature = ast.dump(_Normalize().visit(module))
    except (SyntaxError, ValueError, RecursionError):
        return None
    return signature if len(signature) >= MIN_SIGNATURE_CHARS else None


def _collect() -> dict[str, list[tuple[str, int, str]]]:
    """Map each normalized body hash to the (file, line, name) sites that share it."""
    buckets: dict[str, list[tuple[str, int, str]]] = defaultdict(list)
    for target in TARGETS:
        for path in sorted(target.rglob("*.py")):
            try:
                tree = ast.parse(path.read_text())
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                    continue
                if node.name in SKIP_NAMES:
                    continue
                signature = _body_signature(node)
                if signature is None:
                    continue
                digest = hashlib.md5(signature.encode()).hexdigest()
                rel = str(path.relative_to(ROOT))
                buckets[digest].append((rel, node.lineno, node.name))
    return buckets


def main() -> int:
    groups = [sites for sites in _collect().values() if len(sites) > 1]
    # Only cross-file copies. Two identical bodies in one module are usually a deliberate
    # per-type dispatch, and are visible to the reader anyway.
    groups = [g for g in groups if len({file for file, _, _ in g}) > 1]

    failures = []
    for sites in sorted(groups, key=lambda g: (-len(g), g[0][0])):
        key = ",".join(sorted(f"{file}:{line}" for file, line, _ in sites))
        if key in DUPLICATION_ALLOW:
            print(f"allow: {sites[0][2]}() x{len(sites)} — {DUPLICATION_ALLOW[key]}")
            continue
        failures.append(sites)

    for sites in failures:
        names = " / ".join(sorted({name for _, _, name in sites}))
        print(f"\nFAIL: duplicated logic in {len(sites)} places [{names}]")
        for file, line, name in sites:
            print(f"    {file}:{line}  {name}()")
        print(
            "    → lift it into a neutral layer (plan / metadata / config / _internal) and "
            "import it.\n      kyber/carbonite/core/governance cannot import each other, so "
            "copy-paste is the\n      only *wrong* way to share between them — see "
            ".claude/rules/architecture.md."
        )

    if failures:
        print(f"\nlint-duplication: {len(failures)} duplicated block(s)")
        return 1
    print("lint-duplication: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
