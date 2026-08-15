"""No module may hand-roll the missing-column check again.

`ml.stats._shared.require_columns` has now been extracted twice. The first extraction
collapsed four copies from `splitting`, `metrics` and `preprocessors`, and its docstring said
so — and thirty more accumulated anyway, across `ml`'s estimators, composers, corpus tools and
metrics. That is not carelessness: from inside a module that has not imported the helper, the
four-line inline check is what every *neighbour* looks like, so writing it again is the local
best guess. Only a whole-tree check can see the difference.

So this is the whole-tree check. It reads the source rather than the behavior, because the
behavior of a copy is fine — that is exactly why copies survive review. What is not fine is
what a copy costs later:

* the hint. `require_columns` takes one, and the copies mostly hard-coded ``"Pass an existing
  column."``, so improving the message for a whole family meant finding every copy.
* the cost. The helper builds a `set` before looping, because `Dataset.columns` is a list and
  the check runs per requested name — on a wide feature table the naive form is a scan of
  thousands of columns per name. Half the copies looped against the list.
* the order. A missing-column check has to run *before* a numeric-type check, or the user gets
  told their typo is the wrong dtype. `require_fit_columns` fixes that order once; ten estimators
  each getting it right was luck.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

pytestmark = pytest.mark.unit

_ROOT = pathlib.Path(__file__).resolve().parents[2] / "python" / "batcher"

#: The two modules allowed to construct this error directly: the helper itself, and the
#: plan-layer validator that runs against an `available_schema` rather than a `Dataset` and
#: therefore predates and underlies the `ml` helper.
_ALLOWED = {
    "ml/stats/_shared.py",
    "plan/logical/base.py",
    "_internal/errors/suggest.py",
    "_internal/errors/hierarchy.py",
}


def _column_error_sites() -> list[str]:
    """Every ``unknown_message("column", ...)`` construction outside the allowed modules."""
    found: list[str] = []
    for path in sorted(_ROOT.rglob("*.py")):
        rel = path.relative_to(_ROOT).as_posix()
        if rel in _ALLOWED:
            continue
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:  # another session mid-edit; not this test's business
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if name != "unknown_message" or not node.args:
                continue
            first = node.args[0]
            if isinstance(first, ast.Constant) and first.value == "column":
                found.append(f"{rel}:{node.lineno}")
    return found


def test_no_module_builds_its_own_missing_column_error() -> None:
    """A new inline copy fails here rather than in a review nobody runs."""
    sites = _column_error_sites()
    assert not sites, (
        "these sites build the missing-column error themselves instead of calling "
        "batcher.ml.stats._shared.require_columns (or require_names, when the caller holds a "
        f"column list rather than a Dataset): {sites}"
    )


def test_the_helper_checks_membership_against_a_set() -> None:
    """Not a style point: `Dataset.columns` is a list, and the check runs per requested name.

    A wide feature table — a few thousand one-hot columns is ordinary — turns a handful of name
    checks into a few thousand list scans each. The copies this replaced mostly looped against
    the list, so the cost was real and invisible.
    """
    source = (_ROOT / "ml" / "stats" / "_shared.py").read_text()
    body = source[source.index("def require_names") :]
    assert "set(available)" in body


def test_the_helper_takes_a_sequence_so_the_message_is_reproducible() -> None:
    """A `set` would do for the membership test and not for the message.

    `did_you_mean` and the alternatives list both iterate the candidates, and a `set` of
    strings iterates in an order that depends on the interpreter's hash seed. The same typo
    would then produce a different message from one process to the next, which makes an error
    nobody can quote in a bug report. The helper therefore keeps the sequence and builds the
    set only for the `in` test.
    """
    from batcher.ml.stats._shared import require_names

    pool = ["beta", "beto", "gamma"]
    messages = set()
    for _ in range(3):
        with pytest.raises(Exception) as caught:
            require_names(pool, "bet")
        messages.add(str(caught.value))
    assert len(messages) == 1, "the same typo must produce one message"
    assert "beta" in messages.pop()


def test_a_missing_column_is_reported_before_a_wrong_dtype() -> None:
    """The order `require_fit_columns` exists to fix.

    Numeracy-first tells a user their typo has the wrong type — or, because `require_numeric`
    deliberately skips a column it cannot find, says nothing at all and lets the failure surface
    from inside the data plane as an Arrow error naming no column.
    """
    import batcher as bt
    from batcher._internal.errors import ColumnNotFoundError
    from batcher.ml._estimator import require_fit_columns

    ds = bt.from_pydict({"num": [1.0], "text": ["a"], "y": [1.0]})
    with pytest.raises(ColumnNotFoundError, match="Did you mean 'num'"):
        require_fit_columns("Ridge", ds, ["nun", "text"], "y", numeric_target=True)


def test_a_present_but_unusable_column_is_still_reported() -> None:
    """The check the missing-column pass must not swallow."""
    import batcher as bt
    from batcher._internal.errors import PlanError
    from batcher.ml._estimator import require_fit_columns

    ds = bt.from_pydict({"num": [1.0], "text": ["a"], "y": [1.0]})
    with pytest.raises(PlanError, match="text"):
        require_fit_columns("Ridge", ds, ["num", "text"], "y", numeric_target=True)


def test_a_classifier_target_may_be_a_string_and_a_regressor_s_may_not() -> None:
    """`numeric_target` is the one axis the ten copies differed on, so it is the one to pin."""
    import batcher as bt
    from batcher._internal.errors import PlanError
    from batcher.ml._estimator import require_fit_columns

    ds = bt.from_pydict({"num": [1.0], "label": ["cat"]})
    require_fit_columns("LogisticRegression", ds, ["num"], "label")
    with pytest.raises(PlanError, match="target 'label'"):
        require_fit_columns("Ridge", ds, ["num"], "label", numeric_target=True)


def test_an_unsupervised_fit_names_no_target() -> None:
    import batcher as bt
    from batcher.ml._estimator import require_fit_columns

    require_fit_columns("KMeans", bt.from_pydict({"a": [1.0], "b": [2.0]}), ["a", "b"])
