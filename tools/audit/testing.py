"""Tests that cannot fail: no assertion at all, a vacuous one, or order checked unordered.

`CLAUDE.md`'s loudest warning is that "a green gate is not a green light": every gate passed
while `sort(descending=True)` returned unsorted data under spill. Nothing mechanical caught
it, because the test ran and passed — it just was not *checking* the thing.

These rules are **AST-based and precise enough to gate on**, which is a deliberate change of
kind. The earlier implementation matched regexes against unparsed source, and every finding
it produced on this tree was false: it read `.sort(` inside a string literal as a sort, an
`ORDER BY` inside a window clause as a result ordering, and a `warnings.simplefilter("error")`
block — how every "...stays silent" test here is written — as a test that asserts nothing.
A detector with a 100% false-positive rate does not get triaged, it gets ignored, and the
one real finding it might one day produce goes with it.

So each rule below is calibrated to **zero findings on the current tree**, which is what
lets `tools/lint_tests.py` gate on them instead of merely reporting. The precision work is
in the exclusions, and each exclusion records the false positive that motivated it.

Two consumers, one implementation: `tools/audit_health.py --only test-quality` reports these
as `Finding`s alongside the other health detectors, and `tools/lint_tests.py` fails the build
on them.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

from tools.audit.context import ROOT, Context, Finding, _rel

_Func = ast.FunctionDef | ast.AsyncFunctionDef

#: Methods after which row order is part of the result contract.
ORDER_DEFINING = {"sort", "top_k", "bottom_k", "top_n"}

#: Methods that redefine or destroy order, so an order-independent compare is correct again
#: below them. A sorted relation feeding a `GROUP BY` or a `UNION` has no result order to
#: check — the sort chose *which* rows survive, and the consumer discarded their order.
ORDER_DESTROYING = {
    "agg",
    "aggregate",
    "concat",
    "count",
    "distinct",
    "except_",
    "explode",
    "group_by",
    "intersect",
    "join",
    "join_asof",
    "melt",
    "pivot",
    "sample",
    "union",
    "unique",
    "unnest",
    "unpivot",
    "value_counts",
}

#: The order-independent comparison helpers. `assert_same_ordered` is deliberately absent:
#: it is the fix, and a substring test would flag the very helper that resolves the finding.
ORDER_BLIND = {"assert_same", "assert_tables_equal"}

#: Calls that constitute an assertion on their own.
ASSERTING_CALLS = {
    "raises",
    "warns",
    "deprecated_call",
    "fail",
    "importorskip",
    "approx",
    "assert_frame_equal",
    "assert_series_equal",
}


def called_names(node: ast.AST) -> set[str]:
    """Every callable name invoked anywhere under `node`, attribute or bare."""
    names: set[str] = set()
    for call in ast.walk(node):
        if isinstance(call, ast.Call):
            func = call.func
            if isinstance(func, ast.Name):
                names.add(func.id)
            elif isinstance(func, ast.Attribute):
                names.add(func.attr)
    return names


def asserting_helpers(tree: ast.Module) -> set[str]:
    """Functions in this module that assert, directly or through another such helper.

    A test delegating to a local `_same(out, expected)` is asserting. Resolving that — and
    resolving it *transitively*, so a two-hop delegation still counts — is the difference
    between a rule the suite can keep green and one it would suppress.
    """
    funcs = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    asserting: set[str] = set()
    changed = True
    while changed:
        changed = False
        for name, fn in funcs.items():
            if name not in asserting and asserts_anything(fn, asserting):
                asserting.add(name)
                changed = True
    return asserting


def asserts_anything(fn: ast.AST, helpers: set[str]) -> bool:
    """Whether this function can fail."""
    if any(isinstance(n, ast.Assert) for n in ast.walk(fn)):
        return True
    names = called_names(fn)
    if names & ASSERTING_CALLS or names & helpers:
        return True
    if any(n.startswith("assert") or n.startswith("_assert") for n in names):
        return True
    # `with warnings.catch_warnings(): warnings.simplefilter("error")` turns any warning into
    # an exception, so the test fails if the code warns. That is a real (negative) assertion,
    # and it is how every "...stays silent" test in this suite is written — reading it as
    # assertion-free produced 13 false positives on its own.
    if "simplefilter" in names or "filterwarnings" in names:
        return True
    # A bare call as a statement — `reporter.handle(event)  # no raise` — asserts that the
    # call does not raise, a real and widely-used contract here ("survives a closed stream",
    # "a verified principal is accepted"). What remains flagged is a test that only *binds*
    # results and never exercises anything, which is the signature of an assertion lost in
    # a refactor.
    return any(isinstance(s, ast.Expr) and isinstance(s.value, ast.Call) for s in ast.walk(fn))


def _chain_methods(call: ast.Call) -> list[str]:
    """Method names in a fluent chain, outermost first."""
    methods: list[str] = []
    cur: ast.AST = call
    while True:
        if isinstance(cur, ast.Call):
            cur = cur.func
            continue
        if isinstance(cur, ast.Attribute):
            methods.append(cur.attr)
            cur = cur.value
            continue
        break
    return methods


def order_defining_method(call: ast.Call) -> str | None:
    """The order-defining method a chain ends on, or None when order is not its contract."""
    for method in _chain_methods(call):
        if method in ORDER_DEFINING:
            return method
        if method in ORDER_DESTROYING:
            return None
    return None


def _last_assignment(name: str, fn: ast.AST) -> ast.Call | None:
    """The call a local variable was last assigned from, so binding cannot launder a chain."""
    found = None
    for stmt in ast.walk(fn):
        if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
            for target in stmt.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    found = stmt.value
    return found


def _int_literal(node: ast.expr) -> int | None:
    """An integer literal's value, including a negated one.

    ``-1`` parses as ``UnaryOp(USub, Constant(1))``, not ``Constant(-1)``, so matching on
    `ast.Constant` alone silently misses ``assert len(x) > -1`` — the same always-true
    assertion as ``>= 0``.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _int_literal(node.operand)
        return None if inner is None else -inner
    return None


def vacuous_reason(test: ast.expr) -> str | None:
    """A reason if this assertion is true by construction."""
    if isinstance(test, ast.Constant) and test.value:
        return f"`assert {test.value!r}` is always true"
    if isinstance(test, ast.Compare) and len(test.ops) == 1:
        op, left, right = test.ops[0], test.left, test.comparators[0]
        if isinstance(left, ast.Call) and isinstance(left.func, ast.Name) and left.func.id == "len":
            bound = _int_literal(right)
            if bound is not None and (
                (isinstance(op, ast.GtE) and bound <= 0) or (isinstance(op, ast.Gt) and bound < 0)
            ):
                sign = ">=" if isinstance(op, ast.GtE) else ">"
                return f"`assert len(...) {sign} {bound}` is always true"
        # `assert f(x) == f(x)` is a determinism check: the function runs twice and a
        # non-reproducible result fails it. Only a call-free repetition — `assert k == k` —
        # is genuinely true by construction. Flagging the former was this rule's first false
        # positive, on 16 real determinism tests.
        if (
            isinstance(op, ast.Eq)
            and ast.dump(left) == ast.dump(right)
            and not any(isinstance(n, ast.Call) for n in (*ast.walk(left), *ast.walk(right)))
        ):
            return "both sides of the comparison are the same call-free expression"
    return None


def check_module(path: Path, tree: ast.Module) -> Iterator[Finding]:
    """Every finding in one parsed test module."""
    rel = _rel(path)
    helpers = asserting_helpers(tree)

    for fn in ast.walk(tree):
        if not isinstance(fn, _Func) or not fn.name.startswith("test_"):
            continue

        for node in ast.walk(fn):
            if isinstance(node, ast.Assert):
                reason = vacuous_reason(node.test)
                if reason:
                    yield Finding(
                        "vacuous-assertion", "high", rel, node.lineno, f"`{fn.name}`: {reason}"
                    )

        if not asserts_anything(fn, helpers):
            yield Finding(
                "vacuous-test",
                "high",
                rel,
                fn.lineno,
                f"`{fn.name}` asserts nothing, raises nothing, and expects no warning — it "
                f"binds results and discards them",
            )

        for node in ast.walk(fn):
            if not isinstance(node, ast.Call):
                continue
            helper = node.func.id if isinstance(node.func, ast.Name) else None
            if helper not in ORDER_BLIND:
                continue
            if helper == "assert_tables_equal" and any(kw.arg == "ordered" for kw in node.keywords):
                continue
            for arg in node.args:
                chain = arg if isinstance(arg, ast.Call) else None
                if isinstance(arg, ast.Name):
                    chain = _last_assignment(arg.id, fn)
                if chain is None:
                    continue
                method = order_defining_method(chain)
                if method:
                    yield Finding(
                        "order-blind-test",
                        "high",
                        rel,
                        node.lineno,
                        f"`{fn.name}` compares a `{method}()` result with `{helper}`, which "
                        f"ignores row order — use `assert_same_ordered`, or "
                        f"`assert_tables_equal(..., ordered=True)`",
                    )
                    break


def test_modules() -> Iterator[tuple[Path, ast.Module]]:
    """Every parsed `test_*.py` under `tests/`."""
    for path in sorted((ROOT / "tests").rglob("test_*.py")):
        try:
            yield path, ast.parse(path.read_text())
        except (SyntaxError, UnicodeDecodeError):
            continue


def detect_test_quality(ctx: Context) -> Iterator[Finding]:  # noqa: ARG001 — uniform signature
    """Tests that cannot fail, and ordered results checked with an unordered comparison."""
    for path, tree in test_modules():
        yield from check_module(path, tree)
