"""Prompt-construction functions — assembling an LLM prompt from row columns in the data plane.

These are row-wise string builders (used in ``select``/``with_columns``), so they are pinned to the
exact per-row output: named-placeholder interpolation, tag wrapping, and token-budget trimming, plus
the validation that a template and its fields agree.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.plan.functions import prompt as pr


def _row(expr, rows: dict) -> object:
    return bt.from_pydict(rows).select(r=expr).to_pydict()["r"][0]


pytestmark = pytest.mark.unit


def test_render_template_interpolates_named_fields() -> None:
    expr = pr.render_template(
        "Write {n} facts about {topic}.", n=bt.col("n"), topic=bt.col("topic")
    )
    assert _row(expr, {"topic": ["cats"], "n": ["3"]}) == "Write 3 facts about cats."


def test_render_template_with_no_placeholders_is_the_literal() -> None:
    assert _row(pr.render_template("just text"), {"x": [1]}) == "just text"


def test_render_template_missing_field_raises() -> None:
    with pytest.raises(PlanError):
        pr.render_template("{a} and {b}", a=bt.lit("x"))


def test_render_template_unused_field_raises() -> None:
    with pytest.raises(PlanError):
        pr.render_template("{a}", a=bt.lit("x"), b=bt.lit("y"))


def test_render_template_treats_null_as_empty() -> None:
    expr = pr.render_template("[{v}]", v=bt.col("v"))
    out = bt.from_pydict({"v": ["x", None]}).select(r=expr).to_pydict()["r"]
    assert out == ["[x]", "[]"]


def test_wrap_tag_surrounds_the_value() -> None:
    got = _row(pr.wrap_tag(bt.col("q"), "question"), {"q": ["2+2?"]})
    assert got == "<question>2+2?</question>"


def test_truncate_to_token_budget_cuts_to_char_estimate() -> None:
    # budget 1 token * 4 chars/token = 4 chars.
    assert _row(pr.truncate_to_token_budget("t", budget=1), {"t": ["hello world"]}) == "hell"


def test_truncate_to_token_budget_leaves_short_text() -> None:
    assert _row(pr.truncate_to_token_budget("t", budget=10), {"t": ["hi"]}) == "hi"


def test_render_template_composes_multiple_columns() -> None:
    expr = pr.render_template("{a}-{b}-{a}", a=bt.col("a"), b=bt.col("b"))
    out = bt.from_pydict({"a": ["1", "2"], "b": ["x", "y"]}).select(r=expr).to_pydict()["r"]
    assert out == ["1-x-1", "2-y-2"]
