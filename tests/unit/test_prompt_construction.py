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


# --- assembly: tags, chat formats, retrieved context -------------------------------


def test_tagged_fields_delimits_each_field_in_order() -> None:
    expr = pr.tagged_fields(question=bt.col("q"), context=bt.col("c"))
    got = _row(expr, {"q": ["Why?"], "c": ["Because."]})
    assert got == "<question>Why?</question>\n<context>Because.</context>"


def test_tagged_fields_requires_at_least_one_field() -> None:
    with pytest.raises(PlanError):
        pr.tagged_fields()


def test_chatml_prompt_ends_with_an_open_assistant_turn() -> None:
    got = _row(pr.chatml_prompt(bt.col("q")), {"q": ["Hi"]})
    assert got.endswith("<|im_start|>assistant\n")
    assert "<|im_start|>system" not in got


def test_chatml_prompt_includes_a_system_turn_when_given() -> None:
    got = _row(pr.chatml_prompt(bt.col("q"), bt.lit("Be terse.")), {"q": ["Hi"]})
    assert got.startswith("<|im_start|>system\nBe terse.<|im_end|>\n")


def test_instruction_prompt_omits_the_input_section_when_there_is_none() -> None:
    got = _row(pr.instruction_prompt(bt.col("i")), {"i": ["Summarize."]})
    assert "### Input:" not in got
    assert got.endswith("### Response:\n")


def test_instruction_prompt_includes_the_input_section_when_given() -> None:
    got = _row(pr.instruction_prompt(bt.col("i"), bt.col("c")), {"i": ["Sum."], "c": ["Text."]})
    assert "### Input:\nText." in got


def test_join_context_drops_the_gaps_a_short_retrieval_leaves() -> None:
    """A retriever returning fewer than k hits must not leave blank separators behind."""
    expr = pr.join_context(bt.col("hits"), separator=" | ")
    assert _row(expr, {"hits": [["a", "", None, "b"]]}) == "a | b"


def test_join_context_of_an_empty_hit_list_is_empty() -> None:
    """A query whose retrieval found nothing yields an empty context, not a null."""
    out = (
        bt.from_pydict({"hits": [["a"], []]})
        .select(c=pr.join_context(bt.col("hits")))
        .to_pydict()["c"]
    )
    assert out == ["a", ""]


# --- budgeting: fitting the assembled prompt into a window -------------------------


def test_truncate_middle_keeps_both_ends_within_the_budget() -> None:
    got = _row(pr.truncate_middle("t", budget=2, marker="~"), {"t": ["abcdefghijklmnop"]})
    assert got.startswith("abcd")
    assert got.endswith("nop")
    assert len(got) <= 8  # 2 tokens * 4 chars


def test_truncate_middle_leaves_a_short_value_untouched() -> None:
    assert _row(pr.truncate_middle("t", budget=4, marker="~"), {"t": ["abc"]}) == "abc"


def test_truncate_middle_never_exceeds_the_character_budget() -> None:
    """The property the head/tail split exists to satisfy, across a range of budgets."""
    for budget in (2, 3, 5, 10, 40):
        got = _row(pr.truncate_middle("t", budget=budget, marker="\n...\n"), {"t": ["x" * 500]})
        assert len(got) <= budget * 4


def test_truncate_middle_rejects_a_marker_larger_than_the_budget() -> None:
    with pytest.raises(PlanError):
        pr.truncate_middle("t", budget=1, marker="a very long marker indeed")


def test_prompt_token_estimate_sums_the_parts() -> None:
    ds = bt.from_pydict({"a": ["12345678"], "b": ["1234"]})
    assert ds.select(n=pr.prompt_token_estimate(bt.col("a"), bt.col("b"))).to_pydict()["n"] == [3]


def test_prompt_token_estimate_requires_a_part() -> None:
    with pytest.raises(PlanError):
        pr.prompt_token_estimate()


def test_prompt_token_estimate_rejects_a_non_positive_rate() -> None:
    with pytest.raises(PlanError):
        pr.prompt_token_estimate(bt.col("a"), chars_per_token=0.0)


def test_fits_context_reserves_room_for_the_answer() -> None:
    """The point of the reserve: a prompt that exactly fills the window cannot be answered."""
    out = (
        bt.from_pydict({"p": ["x" * 360]})  # ~90 estimated tokens
        .select(
            bare=pr.fits_context("p", window=100),
            reserved=pr.fits_context("p", window=100, reserve_output=50),
        )
        .to_pydict()
    )
    assert out["bare"] == [True]
    assert out["reserved"] == [False]


def test_fits_context_rejects_a_reserve_that_leaves_no_prompt() -> None:
    with pytest.raises(PlanError):
        pr.fits_context("p", window=100, reserve_output=100)


def test_the_budget_functions_reject_a_non_positive_budget() -> None:
    for build in (pr.truncate_to_token_budget, pr.truncate_middle):
        with pytest.raises(PlanError):
            build("t", budget=0)
