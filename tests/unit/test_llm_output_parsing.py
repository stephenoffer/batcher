"""LLM-output-parsing free functions — turning generated text into typed columns.

These pull structured fragments out of a model's prose-wrapped output (a JSON blob, a fenced
code block, an ``<answer>`` tag, a multiple-choice letter, a refusal). Each is a thin wrapper
over the `.str` regex primitives, so the contract to pin is the *regex behavior*: what it
extracts, and that it degrades to an empty string (never an error) when the fragment is absent.
"""

from __future__ import annotations

import pytest

import batcher as bt

pytestmark = pytest.mark.unit


def _col(fn_result, rows: list[str]) -> list:
    """Run a parser expression over a single ``o`` column and return the result list."""
    return bt.from_pydict({"o": rows}).select(r=fn_result).to_pydict()["r"]


def test_extract_json_pulls_the_first_object() -> None:
    rows = ['prefix {"a": 1, "b": [2, 3]} suffix', "no json here"]
    assert _col(bt.extract_json("o"), rows) == ['{"a": 1, "b": [2, 3]}', ""]


def test_extract_json_array_pulls_the_first_list() -> None:
    rows = ["items: [1, 2, 3].", "{}"]
    assert _col(bt.extract_json_array("o"), rows) == ["[1, 2, 3]", ""]


def test_extract_code_block_drops_fences_and_language() -> None:
    fenced = "here:\n```python\nprint(1)\n```\ndone"
    rows = [fenced, "inline ```x``` block", "no fence"]
    got = _col(bt.extract_code_block("o"), rows)
    assert got[0] == "print(1)\n"
    assert got[1] == "x"
    assert got[2] == ""


def test_extract_first_number_parses_to_float() -> None:
    rows = ["the total is 42 apples", "cost: -3.5 dollars", "no number"]
    assert _col(bt.extract_first_number("o"), rows) == [42.0, -3.5, None]


def test_extract_tag_reads_named_xml_tag() -> None:
    rows = ["<answer>Paris</answer> is the capital", "no tag", "<answer></answer>"]
    assert _col(bt.extract_tag("o", "answer"), rows) == ["Paris", "", ""]


def test_extract_tag_escapes_the_tag_name() -> None:
    # A tag name with regex metacharacters must be matched literally, not as a pattern.
    rows = ["<a.b>x</a.b>"]
    assert _col(bt.extract_tag("o", "a.b"), rows) == ["x"]


def test_extract_reasoning_reads_think_block() -> None:
    rows = ["<think>2+2 is 4</think>The answer is 4", "<thinking>hmm</thinking>ok", "plain"]
    assert _col(bt.extract_reasoning("o"), rows) == ["2+2 is 4", "hmm", ""]


def test_strip_reasoning_removes_the_think_block() -> None:
    rows = ["<think>2+2 is 4</think>The answer is 4", "plain text"]
    assert _col(bt.strip_reasoning("o"), rows) == ["The answer is 4", "plain text"]


def test_extract_after_takes_text_after_a_marker() -> None:
    rows = ["Question: 1+1\nAnswer: 2", "no marker here"]
    assert _col(bt.extract_after("o", "Answer:"), rows) == ["2", ""]


def test_extract_between_takes_text_between_markers() -> None:
    rows = ["start[keep this]end", "no markers"]
    assert _col(bt.extract_between("o", "[", "]"), rows) == ["keep this", ""]


def test_is_refusal_flags_common_refusal_phrasings() -> None:
    rows = [
        "I'm sorry, I can't help with that.",
        "As an AI, I am unable to comply.",
        "Sure, here is the answer: 42.",
        "The capital of France is Paris.",
    ]
    assert _col(bt.is_refusal("o"), rows) == [True, True, False, False]


def test_extract_choice_reads_a_standalone_letter() -> None:
    rows = ["The answer is B.", "I think (C) is right", "definitely D", "no letter"]
    assert _col(bt.extract_choice("o"), rows) == ["B", "C", "D", ""]


def test_parsers_accept_an_expression_not_just_a_name() -> None:
    # `str | Expr`: passing an expression (an upcased column) must work like a name.
    ds = bt.from_pydict({"o": ['answer: {"a": 1}']})
    got = ds.select(j=bt.extract_json(bt.col("o"))).to_pydict()["j"]
    assert got == ['{"a": 1}']


# --- the extractors behind the format-compliance metrics ---------------------------


def test_extract_boxed_reads_the_latex_answer() -> None:
    rows = ["so the answer is \\boxed{42}.", "no box here", "\\boxed{x + 1} follows"]
    assert _col(bt.extract_boxed("o"), rows) == ["42", "", "x + 1"]


def test_extract_boxed_takes_the_first_box() -> None:
    rows = ["\\boxed{1} then \\boxed{2}"]
    assert _col(bt.extract_boxed("o"), rows) == ["1"]


def test_extract_boxed_pairs_with_its_own_rate_metric() -> None:
    """The asymmetry this closes: the rate existed, the answer could not be read out."""
    rows = ["\\boxed{7}", "no box"]
    ds = bt.from_pydict({"o": rows})
    assert ds.agg(r=bt.boxed_answer_rate("o")).to_pydict()["r"] == [0.5]
    assert ds.select(a=bt.extract_boxed("o")).to_pydict()["a"] == ["7", ""]


def test_extract_last_number_reads_the_conclusion_not_the_first_step() -> None:
    """A reasoning chain emits its intermediate quantities first; the answer is last."""
    rows = ["12 apples minus 4 leaves 8", "no digits"]
    assert _col(bt.extract_last_number("o"), rows) == [8.0, None]
    assert _col(bt.extract_first_number("o"), rows) == [12.0, None]


def test_extract_last_number_keeps_a_sign_and_a_decimal() -> None:
    assert _col(bt.extract_last_number("o"), ["answer: -3.5"]) == [-3.5]


def test_extract_last_number_of_a_single_number_is_that_number() -> None:
    assert _col(bt.extract_last_number("o"), ["42"]) == [42.0]


def test_extract_citations_lists_every_marker_in_order() -> None:
    rows = ["Paris [1] is the capital [2][1].", "uncited"]
    assert _col(bt.extract_citations("o"), rows) == [["1", "2", "1"], []]


def test_extract_citations_keeps_repeats_so_they_can_be_counted_or_deduped() -> None:
    rows = ["[3][3][3]"]
    assert _col(bt.extract_citations("o"), rows) == [["3", "3", "3"]]
    deduped = bt.from_pydict({"o": rows}).select(
        c=bt.extract_citations("o").list.unique().list.len()
    )
    assert deduped.to_pydict()["c"] == [1]


def test_extract_citations_ignores_a_bracket_that_is_not_a_citation() -> None:
    assert _col(bt.extract_citations("o"), ["see [note] and [12]"]) == [["12"]]


def test_a_citation_can_be_checked_against_what_was_retrieved() -> None:
    """The use the extractor exists for: a marker pointing at a passage never retrieved."""
    answered = bt.from_pydict({"answer": ["backed by [1] and [9]"], "retrieved": [["1", "2"]]})
    got = answered.select(
        fabricated=bt.extract_citations("answer").list.set_difference(bt.col("retrieved"))
    )
    assert got.to_pydict()["fabricated"] == [["9"]]


@pytest.mark.parametrize("parser", [bt.extract_boxed, bt.extract_last_number, bt.extract_citations])
def test_a_null_generation_never_raises(parser) -> None:
    assert len(_col(parser("o"), [None, "text"])) == 2
