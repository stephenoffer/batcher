"""The plan, in every shape the dashboard needs to show it.

One walk over one IR document, rendered three ways: a laid-out graph (`build_dag`), a
copy-pasteable EXPLAIN tree (`explain_text` / `explain_rows`), and the difference between
the plan as written and the plan that ran (`plan_diff`). They share `describe` — the module
that answers "what does this operator do" — so a step's subtitle is identical wherever it
appears, which is what lets a reader move between the graph and the text without having to
re-establish which box is which line.
"""

from __future__ import annotations

from batcher.observe.dag.build import BREAKERS, build_dag, plan_shape
from batcher.observe.dag.describe import (
    ORDERED_CHILD_KEYS,
    children,
    describe,
    expr_text,
    is_plan,
    kind_of,
)
from batcher.observe.dag.diff import plan_diff
from batcher.observe.dag.explain import explain_rows, explain_text

__all__ = [
    "BREAKERS",
    "ORDERED_CHILD_KEYS",
    "build_dag",
    "children",
    "describe",
    "explain_rows",
    "explain_text",
    "expr_text",
    "is_plan",
    "kind_of",
    "plan_diff",
    "plan_shape",
]
