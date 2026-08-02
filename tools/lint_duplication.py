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

#: Known duplicates, each with a reason. Keyed by the sorted "file:function" list of its
#: sites — deliberately NOT by line, because an edit anywhere above a listed site shifts its
#: line number, the key stops matching, and a ledgered duplicate silently becomes a red gate
#: that has nothing to do with the change that tripped it. That is not hypothetical: this
#: entry was keyed at 332/161 while the functions sat at 343/162, so `just lint-duplication`
#: (a pre-commit hook) failed at HEAD.
#:
#: This is a *ledger*, not an amnesty: an entry here is debt that is visible and expected to
#: shrink, not a duplicate that has been blessed. Prefer fixing to listing.
# Empty: the one tracked entry (the speculative-relaunch closure copy-pasted across the four
# Flight reducers) was fixed rather than blessed — the shared barrier now lives in
# `dist/executors/ray_runtime/reduce.py::run_bucket_reduce`, and the sort/window/join/aggregate
# reducers each supply only the two closures that vary. The `test_join_recovery` and
# `test_carbonite_recovery_e2e` integration tests exercise the merged path under real
# worker loss, which is what the entry claimed there was no test for.
DUPLICATION_ALLOW: dict[str, str] = {}


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
        return ast.copy_location(ast.Attribute(value=node.value, attr="_", ctx=node.ctx), node)

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


#: Vocabularies already restated across modules when this detector was written, keyed by their
#: canonical contents. Each entry is debt with a destination, not an exemption: the gate exists
#: so the list shrinks, and a NEW restatement is a failure rather than a line to add here.
#:
#: They were invisible because `ruff` cannot see them — `F811` does not fire on a module-level
#: reassignment, so even two adjacent identical `_COMPARISONS` dicts in one file passed — and
#: because the function-body detector above normalizes constants away by design.
VOCABULARY_ALLOW: dict[str, str] = {
    # The comparison set appears once more as an ORDERED TUPLE in three rule families
    # (`math_algebra/rounding`, `temporal_algebra/{epoch,offsets}`), each spelling
    # `("lt", "le", "gt", "ge", "eq", "ne")`. That is not a restated vocabulary: registration
    # order is *run* order, and the order a family registers its per-operator rules in is a
    # decision belonging to that family, not to the vocabulary. Two of the sites here use a
    # different order from `plan.ir_tags.COMPARISON_ORDER`, so folding them onto it would
    # silently reorder 12 rules — which `just lint-rule-order` catches, and which is exactly
    # the class of change this repo has been bitten by (283 of 302 rules once moved). The
    # *content* is what must not drift, and a rule-order snapshot pins that too.
    "eq,ge,gt,le,lt,ne": (
        "three families' own registration ORDER; content pinned by lint-rule-order"
    ),
}

#: A constant literal repeated in at least this many modules is a shared vocabulary that has no
#: home, not a coincidence. Two is too low — a pair of modules naming the same small set is
#: ordinary — and this catches the shape that actually went wrong: the comparison-operator set
#: was spelled out in *twelve* Kyber modules as a dict, a frozenset, a tuple and a dict of
#: callables, one of them silently missing `eq`/`ne`, and `ruff` reports none of it (F811 does
#: not fire on a module-level reassignment, so even the two adjacent identical copies in one
#: file passed).
CONSTANT_MIN_MODULES = 3

#: Below this many elements a repeated literal is not worth a shared home.
CONSTANT_MIN_ELEMENTS = 4


def _constant_literals() -> dict[str, list[tuple[str, int, str]]]:
    """Module-level constant collections, keyed by their canonical contents."""
    found: dict[str, list[tuple[str, int, str]]] = defaultdict(list)
    for root in TARGETS:
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            try:
                tree = ast.parse(path.read_text())
            except SyntaxError:
                continue
            for node in tree.body:
                if not isinstance(node, ast.Assign | ast.AnnAssign):
                    continue
                target = node.targets[0] if isinstance(node, ast.Assign) else node.target
                value = node.value
                if not isinstance(target, ast.Name) or value is None:
                    continue
                elements = _literal_elements(value)
                if elements is None or len(elements) < CONSTANT_MIN_ELEMENTS:
                    continue
                key = ",".join(sorted(elements))
                rel = path.relative_to(ROOT).as_posix()
                found[key].append((rel, node.lineno, target.id))
    return found


def _literal_elements(value: ast.expr) -> list[str] | None:
    """The canonical elements of a string set/tuple/list/dict literal, or `None`.

    A dict contributes `key=value` pairs, not bare keys. Sharing *keys* is not duplication:
    `{"lt": "gt", ...}` (flip a comparison) and `{"lt": operator.lt, ...}` (evaluate one) are
    different mappings over one vocabulary, and collapsing them would buy indirection rather
    than a single source of truth. Sharing keys *and* values is — that is the same fact
    written twice.
    """
    if isinstance(value, ast.Call):  # frozenset({...}) / set([...])
        name = getattr(value.func, "id", "")
        if name not in ("frozenset", "set") or not value.args:
            return None
        value = value.args[0]
    if isinstance(value, ast.Dict):
        out = []
        for key, val in zip(value.keys, value.values, strict=True):
            if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                return None
            if not isinstance(val, ast.Constant):
                return None  # a computed value is not a restated literal
            out.append(f"{key.value}={val.value!r}")
        return out
    if not isinstance(value, ast.Set | ast.Tuple | ast.List):
        return None
    out = []
    for item in value.elts:
        if not isinstance(item, ast.Constant) or not isinstance(item.value, str):
            return None
        out.append(item.value)
    return out


def _report_constants() -> int:
    """Report vocabularies restated in several modules. Returns the failure count."""
    failures = 0
    for key, sites in sorted(_constant_literals().items()):
        modules = {file for file, _, _ in sites}
        if len(modules) < CONSTANT_MIN_MODULES:
            continue
        if key in VOCABULARY_ALLOW:
            print(
                f"allow: {len(modules)} modules restate [{key[:40]}...] — {VOCABULARY_ALLOW[key]}"
            )
            continue
        failures += 1
        names = " / ".join(sorted({name for _, _, name in sites}))
        print(
            f"\nFAIL: the same {len(key.split(','))}-element vocabulary is restated in "
            f"{len(modules)} modules [{names}]"
        )
        for file, line, name in sorted(sites):
            print(f"    {file}:{line}  {name}")
        print(
            "    → give it ONE home (plan/ir_tags.py for IR vocabulary, else a neutral layer) "
            "and import it.\n      Restating it is how the copies drift: one of them ends up "
            "meaning something narrower\n      while still answering to the same name."
        )
    return failures


def main() -> int:
    groups = [sites for sites in _collect().values() if len(sites) > 1]
    # Only cross-file copies. Two identical bodies in one module are usually a deliberate
    # per-type dispatch, and are visible to the reader anyway.
    groups = [g for g in groups if len({file for file, _, _ in g}) > 1]

    failures = []
    for sites in sorted(groups, key=lambda g: (-len(g), g[0][0])):
        key = ",".join(sorted(f"{file}:{name}" for file, _, name in sites))
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

    constant_failures = _report_constants()

    if failures or constant_failures:
        print(
            f"\nlint-duplication: {len(failures)} duplicated block(s), "
            f"{constant_failures} restated vocabular(ies)"
        )
        return 1
    print("lint-duplication: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
