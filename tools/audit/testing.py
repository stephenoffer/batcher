"""Tests that cannot fail: no assertion at all, or an ordered result compared unordered."""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator
from pathlib import Path

from tools.audit.context import ROOT, Context, Finding, _rel

_ASSERTING = ("assert", "raises", "warns", "approx", "check", "fail", "compare", "match")
#: The order-independent comparison. Matched with a call-shaped regex on purpose:
#: `assert_same` is a prefix of `assert_same_ordered`, so a substring test would flag the
#: very helper that fixes the finding.
_ORDER_BLIND = re.compile(r"\bassert_same\s*\(")
#: What makes a test *produce* an ordered result. Matching the test's body rather than its
#: name matters: `test_sort_merge_join_strategy` names a strategy, not an ordering, and a
#: join's output order is genuinely unspecified — flagging it would be wrong.
_ORDERING_CALL = (".sort(", ".top_n(", "ORDER BY", "order by")
#: An `order_by=` *keyword* is inner ordering, never result ordering: it picks which row an
#: aggregate keeps (`col("v").first(order_by=col("t"))`), which duplicate `distinct` survives
#: (`.distinct(["k"], keep="first", order_by="ts")`), or how a window frame ranks rows
#: (`col("x").rolling_sum(3, order_by=["id"])`). None of the three says anything about the
#: order of the result set, so treating the keyword as an ordering call — which is what the
#: bare `"order_by"` entry above used to do — made every window, first/last, and
#: keep-first-distinct test read as order-blind. All 32 findings it produced were false.
_INNER_ORDER_KWARG = re.compile(r"\border_by\s*=")
#: An `ORDER BY` on the *oracle* side means the expected relation is ordered too, which is
#: what makes an ordered assertion correct rather than flaky.
_ORDER_BY_SQL = re.compile(r"ORDER\s+BY", re.I)
#: Start of a clause whose `ORDER BY` ranks rows *inside* something rather than ordering the
#: result: a window (`OVER (...)`, `.window(...)`, `WINDOW w AS (...)`) or an ordered
#: aggregate (`string_agg(x, ',' ORDER BY y)`).
_INNER_ORDER_CLAUSE = re.compile(
    r"\bOVER\s*\(|\.over\s*\(|\.window\s*\(|\bWINDOW\s+\w+\s+AS\s*\(|\b\w+_agg\s*\(", re.I
)
#: A real relational sort on the Batcher side — the thing that makes row order a contract.
_RELATIONAL_SORT = (".sort(", ".top_n(")
#: ...unless the sorted relation feeds one of these, which discards row order. A sorted
#: input to a `UNION` says nothing about the union's output order, so `assert_same` is right.
_ORDER_DESTROYING = (".union(", ".intersect(", ".except_(", ".group_by(", "UNION", "GROUP BY")

_Func = ast.FunctionDef | ast.AsyncFunctionDef


def _strip_over(source: str) -> str:
    """Remove every window / ordered-aggregate clause, parentheses balanced.

    The ordering inside one of these ranks rows *within a partition or an aggregate*; it says
    nothing about the order of the result set. Without this, every window test reads as "both
    sides are ordered", and converting them to an ordered assertion turned a green suite red
    — 82 failures — the first time this detector was pointed at it.
    """
    while (match := _INNER_ORDER_CLAUSE.search(source)) is not None:
        depth, i = 0, match.end() - 1
        while i < len(source):
            if source[i] == "(":
                depth += 1
            elif source[i] == ")":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        source = source[: match.start()] + source[i + 1 :]
    return source


def _asserting_helpers(trees: dict[Path, ast.Module]) -> set[str]:
    """Names of test helpers that assert on their caller's behalf.

    Without this, every test that delegates to a local `_same(out, expected)` reads as
    assertion-free, which is the difference between a usable report and 100 false positives.
    """
    names: set[str] = set()
    for tree in trees.values():
        for node in ast.walk(tree):
            if not isinstance(node, _Func):
                continue
            source = ast.unparse(node)
            if any(isinstance(n, ast.Assert) for n in ast.walk(node)) or any(
                hint in source for hint in _ASSERTING
            ):
                names.add(node.name)
    return names


def _body_source(node: _Func) -> str:
    """The test's source with its decorators removed.

    A `@pytest.mark.parametrize` list of SQL strings belongs to the *cases*, not to the
    assertion; reading it as the query under test made every parametrized SQL test look
    ordered.
    """
    stripped = type(node)(
        name=node.name,
        args=node.args,
        body=node.body,
        decorator_list=[],
        returns=node.returns,
        type_comment=None,
        type_params=[],
    )
    return ast.unparse(ast.copy_location(stripped, node))


def _asserts_anything(node: _Func, source: str, helpers: set[str]) -> bool:
    """Whether the test can fail: a bare `assert`, an asserting call, or a helper that does."""
    called = {
        n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", "")
        for n in ast.walk(node)
        if isinstance(n, ast.Call)
    }
    return (
        any(isinstance(n, ast.Assert) for n in ast.walk(node))
        or any(hint in source for hint in _ASSERTING)
        or bool(called & helpers)
    )


def _order_finding(node: _Func, source: str, rel: str) -> Finding | None:
    """An order-blindness finding for `node`, or None when its comparison is sound."""
    outer = _INNER_ORDER_KWARG.sub("", _strip_over(source))
    ordered = any(call in outer for call in _ORDERING_CALL)
    explicit = "== [" in source or "assert_same_ordered" in source or ".column(" in source
    if not (ordered and _ORDER_BLIND.search(source) and not explicit):
        return None
    # A sorted relation feeding a `GROUP BY`, `UNION`, `DISTINCT` or `INTERSECT` has no
    # result order to check: the sort chose *which* rows survive or which value an aggregate
    # keeps, and the consuming operator discarded the row order. `assert_same` is the correct
    # comparison there, so saying otherwise is noise that buries the real findings.
    if any(call in outer for call in _ORDER_DESTROYING):
        return None
    # `high` is the unambiguous case only: a real relational sort on the Batcher side *and*
    # an outer `ORDER BY` on the oracle side. Both engines were asked for a specific row
    # order and the result was then compared as a multiset, so the one thing the test set
    # out to check is the one thing it cannot see. Anything weaker — a bare `order_by=`
    # kwarg, an ordering the oracle does not share — is `medium`: read it yourself, because
    # an ordered assertion may be the wrong fix.
    both = (
        any(call in outer for call in _RELATIONAL_SORT)
        and _ORDER_BY_SQL.search(outer) is not None
    )
    return Finding(
        "order-blind-test",
        "high" if both else "medium",
        rel,
        node.lineno,
        f"`{node.name}` orders {'both sides' if both else 'its result'} then compares "
        f"with `assert_same`, which is order-independent"
        + (" — use `assert_same_ordered`" if both else " — check whether order matters"),
    )


def _test_trees() -> dict[Path, ast.Module]:
    trees: dict[Path, ast.Module] = {}
    for path in sorted((ROOT / "tests").rglob("*.py")):
        try:
            trees[path] = ast.parse(path.read_text())
        except (SyntaxError, UnicodeDecodeError):
            continue
    return trees


def detect_test_quality(ctx: Context) -> Iterator[Finding]:  # noqa: ARG001 — uniform signature
    """Tests that cannot fail, and ordered results checked with an unordered comparison."""
    trees = _test_trees()
    helpers = _asserting_helpers(trees)

    for path, tree in trees.items():
        if not path.name.startswith("test_"):
            continue
        rel = _rel(path)
        for node in ast.walk(tree):
            if not isinstance(node, _Func) or not node.name.startswith("test_"):
                continue
            source = _body_source(node)
            if not _asserts_anything(node, source, helpers):
                yield Finding(
                    "vacuous-test",
                    "high",
                    rel,
                    node.lineno,
                    f"`{node.name}` asserts nothing — it passes as long as nothing raises",
                )
                continue
            finding = _order_finding(node, source, rel)
            if finding is not None:
                yield finding
