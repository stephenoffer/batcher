"""`skip_null_prompts`: a row with no prompt need not reach the engine, or cost anything.

A null prompt column cell renders as the empty string and is dispatched. That is
`requests._cell`'s documented behaviour and `test_llm_template` pins it end to end, so it
stays the default and this flag is the opt-out. Two costs motivate the opt-out, and the
first is the one that reaches an invoice:

* the row **spent a generation**. On `vllm_engine` that is a decode slot; on the hosted
  engines it is a billed request per null row, for a prompt that was never there. A corpus
  that is 10% null paid 10% of its inference bill for nothing.
* the answer was indistinguishable from a real one. An engine returns *something* for an
  empty prompt, so `response` held a plausible generation and nothing downstream could
  separate "the model said this" from "there was nothing to ask".

With the flag on the rows are skipped and their output is null, which is what
`_output_column` already documents for a row the engine could not generate for. With it off
nothing about the dispatch changes, which is what the default-behaviour test below pins.

The tests that matter most here are the alignment ones. Requests are dispatched
**length-sorted** and optionally deduped, so narrowing the batch and scattering the results
back happens either side of two permutations; getting it wrong would put one row's answer on
another row, which is the failure this operator's request-count guard exists to prevent.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher.ml.llm.channels import finish_reason_sink, usage_sink

pytestmark = pytest.mark.unit


class _Spy:
    """An engine that records every dispatch and upper-cases each prompt."""

    def __init__(self) -> None:
        self.seen: list[list] = []

    def __call__(self, prompts):
        self.seen.append(list(prompts))
        return [p.upper() for p in prompts]


@pytest.fixture
def spy():
    return _Spy()


@pytest.fixture
def factory(spy):
    return lambda: spy


def test_a_null_prompt_never_reaches_the_engine(spy, factory):
    """The cost half: a decode slot on a GPU, or a billed request on a hosted engine."""
    ds = bt.from_pydict({"prompt": ["a", None, "c", None]})
    ds.ml.generate(factory, prompt_column="prompt", skip_null_prompts=True).collect()
    assert spy.seen == [["a", "c"]]


def test_a_null_prompt_comes_back_null(factory):
    """The correctness half: not an answer to a question nobody asked."""
    ds = bt.from_pydict({"prompt": ["a", None, "c"]})
    assert ds.ml.generate(factory, prompt_column="prompt", skip_null_prompts=True).to_pydict()[
        "response"
    ] == ["A", None, "C"]


def test_an_all_null_batch_calls_the_engine_not_at_all(spy, factory):
    ds = bt.from_pydict({"prompt": [None, None]})
    out = ds.ml.generate(factory, prompt_column="prompt", skip_null_prompts=True).to_pydict()
    assert out["response"] == [None, None]
    assert spy.seen == []


# --- alignment: the failure that would put one row's answer on another row ---------------
#: Deliberately varied lengths, so `_length_sorted_order` genuinely permutes the dispatch,
#: with nulls interleaved and duplicates present so dedup permutes it a second way.
_PROMPTS = ["bbbb", None, "a", "ccccccc", None, "a", "dd"]


@pytest.mark.parametrize("dedup", [False, True], ids=["plain", "dedup"])
def test_every_row_keeps_its_own_answer_through_the_permutations(dedup, factory):
    ds = bt.from_pydict({"prompt": _PROMPTS})
    out = ds.ml.generate(
        factory, prompt_column="prompt", dedup=dedup, skip_null_prompts=True
    ).to_pydict()
    assert out["response"] == [None if p is None else p.upper() for p in _PROMPTS]


def test_the_side_channels_are_null_on_exactly_the_null_rows():
    """Token counts and finish reasons scatter with the outputs, or they misattribute cost."""

    class Reporting:
        def __call__(self, prompts):
            usage_sink().report([(len(p), 1) for p in prompts])
            finish_reason_sink().report(["stop"] * len(prompts))
            return [p.upper() for p in prompts]

    ds = bt.from_pydict({"prompt": _PROMPTS})
    out = ds.ml.generate(
        lambda: Reporting(),
        prompt_column="prompt",
        usage=True,
        finish_reason=True,
        skip_null_prompts=True,
    ).to_pydict()
    # `prompt_tokens` is the prompt's own length here, so a misalignment is visible per row
    # rather than only as a count.
    assert out["prompt_tokens"] == [None if p is None else len(p) for p in _PROMPTS]
    assert [r is None for r in out["finish_reason"]] == [p is None for p in _PROMPTS]


# --- the paths that must NOT change ------------------------------------------------------
def test_a_template_still_renders_a_null_field_as_empty_text(spy, factory):
    """The documented `_cell` behaviour: a null *field* is not a null *prompt*.

    With a template the prompt is built from other columns, so the row still has one — and
    rendering the field as `str(None)` would put the word "None" in the model's context,
    which is the defect `_cell` exists to prevent. Skipping here would be a different bug.
    """
    ds = bt.from_pydict({"a": ["x", None], "b": ["1", "2"]})
    out = ds.ml.generate(
        factory, prompt_column="a", template="{a}|{b}", skip_null_prompts=True
    ).to_pydict()
    assert spy.seen == [["x|1", "|2"]]
    assert out["response"] == ["X|1", "|2"]


def test_a_batch_with_no_nulls_dispatches_exactly_as_before(spy, factory):
    """The common case must not pay for, or be reshaped by, the skip."""
    ds = bt.from_pydict({"prompt": ["x", "y"]})
    out = ds.ml.generate(factory, prompt_column="prompt", skip_null_prompts=True).to_pydict()
    assert out["response"] == ["X", "Y"]
    assert spy.seen == [["x", "y"]]


def test_an_empty_batch_is_still_empty(factory):
    ds = bt.from_pydict({"prompt": []})
    assert ds.ml.generate(factory, prompt_column="prompt", skip_null_prompts=True).count() == 0


def test_the_default_still_dispatches_a_null_prompt_as_empty_text(spy, factory):
    """The contract this flag opts out of, asserted here so the opt-in cannot drift into it.

    `test_llm_template.test_generate_renders_a_null_column_as_empty_end_to_end` pins the same
    behaviour from the template module's side. It is restated from this one because a reader
    arriving at the flag needs to see what leaving it alone does, and because a default that
    changed silently is exactly what this file would otherwise fail to notice.
    """
    ds = bt.from_pydict({"prompt": ["a", None]})
    assert ds.ml.generate(factory, prompt_column="prompt").to_pydict()["response"] == ["A", ""]
    assert spy.seen == [["a", ""]]
