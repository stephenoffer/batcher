"""The test-quality rules must reject each defect they claim to, and accept the rest.

`tools/audit/testing.py` holds the rules; `tools/lint_tests.py` gates on them and
`tools/audit_health.py --only test-quality` reports them. They fail the build on a test
that cannot fail — and a rule that cannot fail is the same bug one level up: it would
report "clean" over a suite full of dead assertions, and nothing would ever say otherwise.

So every rule is fed a violation and must flag it, and fed the shapes it must *not* flag.
That second half is what decides whether the gate is usable: the previous regex
implementation produced 23 findings on this tree and every one was false, and a rule with
that record gets ignored rather than triaged. Each `accepts` test below pins one of those
false positives so it cannot come back.
"""

from __future__ import annotations

import ast
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from tools.audit import testing

pytestmark = pytest.mark.unit


#: A nominal in-repo path for the synthetic modules below. Nothing is written to it — the
#: rules take a parsed tree, and the path is only used to render a finding's location.
_SUBJECT = testing.ROOT / "tests" / "unit" / "test_synthetic_subject.py"


def _check(source: str) -> list[str]:
    """Run the rules over one synthetic test module; return the categories they fired."""
    return [f.category for f in testing.check_module(_SUBJECT, ast.parse(source))]


# --- order-blind --------------------------------------------------------------------


def test_it_flags_a_sort_compared_with_an_order_independent_helper():
    """The rule that would have caught the spilled-descending-sort bug."""
    assert "order-blind-test" in _check(
        "def test_x():\n    assert_same(ds.sort('k').collect(), duck.sql('...'))\n",
    )


def test_it_flags_a_sort_bound_to_a_variable_first():
    """Assigning the chain to a name must not launder it past the rule."""
    assert "order-blind-test" in _check(
        "def test_x():\n    out = ds.sort('k').collect()\n    assert_same(out, duck.sql('...'))\n",
    )


def test_it_flags_assert_tables_equal_without_the_ordered_flag():
    assert "order-blind-test" in _check(
        "def test_x():\n    assert_tables_equal(ds.sort('k').collect(), oracle)\n",
    )


def test_it_accepts_assert_tables_equal_with_ordered_true():
    assert "order-blind-test" not in _check(
        "def test_x():\n    assert_tables_equal(ds.sort('k').collect(), oracle, ordered=True)\n",
    )


def test_it_accepts_an_ordered_comparison():
    assert "order-blind-test" not in _check(
        "def test_x():\n    assert_same_ordered(ds.sort('k').collect(), duck.sql('...'))\n",
    )


def test_it_accepts_a_sort_whose_order_a_later_operator_destroys():
    """`sort().group_by().agg()` has no row-order contract, so an unordered compare is
    correct — flagging it would make the rule noise, and noise gets suppressed."""
    assert "order-blind-test" not in _check(
        "def test_x():\n"
        "    assert_same(ds.sort('k').group_by('g').agg(s=1).collect(), duck.sql('...'))\n",
    )


def test_it_flags_top_k_and_bottom_k_too():
    for method in ("top_k", "bottom_k"):
        assert "order-blind-test" in _check(
            f"def test_x():\n    assert_same(ds.{method}(3, 'k').collect(), d)\n"
        )


# --- vacuous ------------------------------------------------------------------------


def test_it_flags_assert_true():
    assert "vacuous-assertion" in _check("def test_x():\n    assert True\n")


def test_it_flags_a_length_that_cannot_be_negative():
    assert "vacuous-assertion" in _check("def test_x():\n    assert len(rows) >= 0\n")
    assert "vacuous-assertion" in _check("def test_x():\n    assert len(rows) > -1\n")


def test_it_flags_a_call_free_self_comparison():
    assert "vacuous-assertion" in _check("def test_x():\n    assert key == key\n")


def test_it_accepts_a_determinism_check_that_calls_twice():
    """`assert f(x) == f(x)` runs `f` twice; a non-reproducible result fails it. Flagging
    this was the rule's first false positive, on 16 real determinism tests."""
    assert "vacuous-assertion" not in _check("def test_x():\n    assert f(1) == f(1)\n")


def test_it_accepts_an_ordinary_assertion():
    assert _check("def test_x():\n    assert rows == [1, 2]\n") == []


# --- no-assertion -------------------------------------------------------------------


def test_it_flags_a_test_that_only_binds_results():
    """Computed and thrown away — the signature of an assertion lost in a refactor."""
    assert "vacuous-test" in _check("def test_x():\n    out = compute()\n    other = out + 1\n")


def test_it_accepts_a_must_not_raise_test():
    """A bare call as a statement asserts that the call does not raise."""
    assert "vacuous-test" not in _check("def test_x():\n    reporter.handle(event)  # no raise\n")


def test_it_accepts_pytest_raises_and_warns():
    for ctx in ("raises(ValueError)", "warns(UserWarning)"):
        assert "vacuous-test" not in _check(
            f"def test_x():\n    with pytest.{ctx}:\n        boom = 1\n"
        )


def test_it_accepts_a_warning_silenced_into_an_error():
    """`simplefilter("error")` makes any warning fail the test — a negative assertion,
    and how every "...stays silent" test in this suite is written."""
    assert "vacuous-test" not in _check(
        "def test_x():\n"
        "    with warnings.catch_warnings():\n"
        "        warnings.simplefilter('error')\n"
        "        value = run()\n",
    )


def test_it_follows_an_assertion_into_a_local_helper():
    """A test that delegates to `_check(...)` is asserting. Resolving that is the
    difference between a rule the suite can keep green and one it would suppress."""
    assert "vacuous-test" not in _check(
        "def _same(a, b):\n    assert a == b\n\n\ndef test_x():\n    _same(1, 1)\n",
    )


def test_it_follows_a_helper_that_only_calls_another_helper():
    """Transitively, or a two-hop delegation would read as an unasserted test."""
    assert "vacuous-test" not in _check(
        "def _inner(a, b):\n    assert a == b\n\n\n"
        "def _outer(a, b):\n    _inner(a, b)\n\n\n"
        "def test_x():\n    _outer(1, 1)\n",
    )


# --- scope and plumbing -------------------------------------------------------------


def test_it_ignores_functions_that_are_not_tests():
    assert _check("def helper():\n    assert True\n") == []


def test_a_string_that_merely_looks_like_a_sort_is_not_one():
    """The rules read the AST, not the text.

    Every synthetic case in this file is a Python string containing `.sort(` and
    `assert_same(`. The regex implementation these rules replaced flagged this very file
    twice for that reason. Reading the parse tree is what makes the difference.
    """
    assert _check('def test_x():\n    sql = "assert_same(ds.sort(k))"\n    assert sql\n') == []


def test_the_gate_and_the_health_report_share_one_implementation():
    """Two consumers of one rule set, so they can never disagree about a bad test."""
    from tools import lint_tests
    from tools.audit import DETECTORS

    assert DETECTORS["test-quality"] is testing.detect_test_quality
    assert lint_tests.check_module is testing.check_module


def test_the_real_suite_is_clean():
    """The gate's own claim, asserted rather than trusted.

    If this ever fails, a test that cannot fail was just added — read the finding and fix
    the assertion; do not add an allowlist entry to make it green.
    """
    findings = [
        f for path, tree in testing.test_modules() for f in testing.check_module(path, tree)
    ]
    assert not findings, "\n".join(f"{f.path}:{f.line}: {f.message}" for f in findings)
