"""Model-graded evaluation: score, pairwise, and verify.

A judge is a model, so the tests are about what happens when it does not cooperate. An answer
outside the scale, prose instead of a number, a preference that flips when the two responses
change places — each is pinned, because each is the difference between a judged number that
means something and one that reads as signal while measuring the judge.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.ml import llm_pairwise_udf, llm_score_udf, llm_verify_udf

pytestmark = pytest.mark.unit


def _engine(*replies: str):
    """A judge that answers with `replies`, cycling if the batch is longer."""

    def factory():
        def engine(prompts):
            return [replies[i % len(replies)] for i in range(len(prompts))]

        return engine

    return factory


def _recording_engine(replies: list[str], seen: list[list[str]]):
    """A judge that records every prompt batch it was handed, then answers from `replies`."""

    def factory():
        def engine(prompts):
            seen.append(list(prompts))
            start = sum(len(b) for b in seen[:-1])
            return [replies[(start + i) % len(replies)] for i in range(len(prompts))]

        return engine

    return factory


# --- scoring -----------------------------------------------------------------------


def test_a_score_is_parsed_into_a_float_column():
    udf = llm_score_udf(_engine("4"), template="Rate: {a}")
    got = bt.from_pydict({"a": ["x"]}).ml.map_batches(udf).to_pydict()
    assert got["score"] == [4.0]


def test_a_decimal_score_is_kept():
    udf = llm_score_udf(_engine("3.5"), template="Rate: {a}")
    assert bt.from_pydict({"a": ["x"]}).ml.map_batches(udf).to_pydict()["score"] == [3.5]


def test_an_out_of_range_score_is_null_rather_than_clamped():
    """Clamping an 8 to 5 would record a misunderstanding as a strong positive."""
    udf = llm_score_udf(_engine("8"), template="Rate: {a}", low=1, high=5)
    assert bt.from_pydict({"a": ["x"]}).ml.map_batches(udf).to_pydict()["score"] == [None]


def test_prose_instead_of_a_number_is_null():
    udf = llm_score_udf(_engine("I would not rate this above a 2"), template="Rate: {a}")
    assert bt.from_pydict({"a": ["x"]}).ml.map_batches(udf).to_pydict()["score"] == [None]


def test_a_number_leading_the_answer_is_read():
    udf = llm_score_udf(_engine("5 — excellent"), template="Rate: {a}")
    assert bt.from_pydict({"a": ["x"]}).ml.map_batches(udf).to_pydict()["score"] == [5.0]


def test_the_scale_bounds_reach_the_instruction():
    seen: list[list[str]] = []
    udf = llm_score_udf(_recording_engine(["7"], seen), template="Rate: {a}", low=0, high=10)
    bt.from_pydict({"a": ["x"]}).ml.map_batches(udf).to_pydict()
    assert "between 0 and 10" in seen[0][0]


def test_the_instruction_can_be_turned_off():
    seen: list[list[str]] = []
    udf = llm_score_udf(_recording_engine(["4"], seen), template="Rate: {a}", instruct=False)
    bt.from_pydict({"a": ["x"]}).ml.map_batches(udf).to_pydict()
    assert seen[0][0] == "Rate: x"


def test_an_inverted_scale_is_rejected():
    with pytest.raises(PlanError):
        llm_score_udf(_engine("1"), template="Rate: {a}", low=5, high=1)


def test_a_template_naming_an_absent_column_fails_clearly():
    udf = llm_score_udf(_engine("4"), template="Rate: {missing}")
    with pytest.raises(PlanError, match="missing"):
        bt.from_pydict({"a": ["x"]}).ml.map_batches(udf).to_pydict()


def test_a_null_cell_renders_as_empty_not_as_none():
    seen: list[list[str]] = []
    udf = llm_score_udf(_recording_engine(["4"], seen), template="Rate: [{a}]", instruct=False)
    bt.from_pydict({"a": [None]}).ml.map_batches(udf).to_pydict()
    assert seen[0][0] == "Rate: []"


def test_scoring_keeps_every_input_column():
    udf = llm_score_udf(_engine("4"), template="Rate: {a}")
    got = bt.from_pydict({"a": ["x"], "id": [7]}).ml.map_batches(udf).to_pydict()
    assert got["id"] == [7]
    assert set(got) == {"a", "id", "score"}


# --- pairwise ----------------------------------------------------------------------


def _pairwise(factory, **kwargs):
    return llm_pairwise_udf(
        factory,
        template="Compare {left} and {right}",
        a_column="left",
        b_column="right",
        **kwargs,
    )


def test_a_consistent_preference_survives_the_swap():
    """A judge preferring the *response* answers A forward and B reversed."""
    seen: list[list[str]] = []
    udf = _pairwise(_recording_engine(["A", "B"], seen))
    got = bt.from_pydict({"left": ["one"], "right": ["two"]}).ml.map_batches(udf).to_pydict()
    assert got["winner"] == ["A"]
    assert len(seen) == 2  # judged forward, then swapped


def test_a_position_preference_is_recorded_as_a_tie():
    """Answering A both ways means position decided it, not the responses."""
    udf = _pairwise(_engine("A"))
    got = bt.from_pydict({"left": ["one"], "right": ["two"]}).ml.map_batches(udf).to_pydict()
    assert got["winner"] == ["TIE"]


def test_the_swap_can_be_turned_off():
    udf = _pairwise(_engine("A"), swap=False)
    got = bt.from_pydict({"left": ["one"], "right": ["two"]}).ml.map_batches(udf).to_pydict()
    assert got["winner"] == ["A"]


def test_the_swapped_pass_actually_exchanges_the_two_responses():
    seen: list[list[str]] = []
    udf = _pairwise(_recording_engine(["A", "B"], seen), instruct=False)
    bt.from_pydict({"left": ["ONE"], "right": ["TWO"]}).ml.map_batches(udf).to_pydict()
    assert seen[0][0] == "Compare ONE and TWO"
    assert seen[1][0] == "Compare TWO and ONE"


def test_an_explicit_tie_is_kept():
    udf = _pairwise(_engine("TIE"), swap=False)
    got = bt.from_pydict({"left": ["one"], "right": ["two"]}).ml.map_batches(udf).to_pydict()
    assert got["winner"] == ["TIE"]


def test_an_unparseable_verdict_is_null():
    udf = _pairwise(_engine("both are good"), swap=False)
    got = bt.from_pydict({"left": ["one"], "right": ["two"]}).ml.map_batches(udf).to_pydict()
    assert got["winner"] == [None]


def test_a_verdict_is_case_insensitive():
    udf = _pairwise(_engine("b"), swap=False)
    got = bt.from_pydict({"left": ["one"], "right": ["two"]}).ml.map_batches(udf).to_pydict()
    assert got["winner"] == ["B"]


def test_comparing_a_column_with_itself_is_rejected():
    with pytest.raises(PlanError):
        llm_pairwise_udf(_engine("A"), template="{x} vs {x}", a_column="x", b_column="x")


# --- verify ------------------------------------------------------------------------


def test_a_yes_becomes_true():
    udf = llm_verify_udf(_engine("YES"), template="Is {a} fine?")
    assert bt.from_pydict({"a": ["x"]}).ml.map_batches(udf).to_pydict()["passed"] == [True]


def test_a_no_becomes_false():
    udf = llm_verify_udf(_engine("No, it is not."), template="Is {a} fine?")
    assert bt.from_pydict({"a": ["x"]}).ml.map_batches(udf).to_pydict()["passed"] == [False]


def test_an_unusable_verdict_is_null_not_false():
    """A confused judge must not look like a failing dataset."""
    udf = llm_verify_udf(_engine("It depends."), template="Is {a} fine?")
    assert bt.from_pydict({"a": ["x"]}).ml.map_batches(udf).to_pydict()["passed"] == [None]


def test_verification_scores_a_whole_batch_row_by_row():
    udf = llm_verify_udf(_engine("YES", "NO"), template="Is {a} fine?")
    got = bt.from_pydict({"a": ["p", "q", "r", "s"]}).ml.map_batches(udf).to_pydict()
    assert got["passed"] == [True, False, True, False]


def test_a_judged_column_aggregates_like_any_other():
    """The point of a column rather than a loop: the eval is one scan.

    The judged column has to be declared through `output_columns` for the plan above the
    stage to see it, which is the same contract every `map_batches` schema change has.
    """
    udf = llm_score_udf(_engine("4", "2"), template="Rate: {a}")
    scored = bt.from_pydict({"a": ["p", "q", "r", "s"]}).ml.map_batches(
        udf, output_columns=["a", "score"]
    )
    assert scored.agg(m=bt.col("score").mean()).to_pydict()["m"] == [3.0]


def test_a_judge_template_only_materializes_the_columns_it_names():
    """A judged eval runs over rows a generation stage just produced.

    So the batch still carries whatever that stage read — contexts, retrieved passages,
    embeddings, images — and converting all of it to Python per row, to fill a two-slot
    rubric, cost more than the judge call it was preparing.
    """
    import pyarrow as pa

    from batcher.ml.llm.judge import _render

    touched: list[str] = []

    class _Batch:
        schema = pa.schema([("a", pa.string()), ("ctx", pa.string())])
        num_rows = 2

        def column(self, name):
            touched.append(name)
            return pa.array(["x", "y"])

    assert _render("Rate: {a}", _Batch()) == ["Rate: x", "Rate: y"]
    assert touched == ["a"]


def test_a_judge_template_naming_a_missing_column_still_names_what_is_available():
    import pyarrow as pa

    from batcher.ml.llm.judge import _render

    batch = pa.RecordBatch.from_pydict({"a": ["x"], "b": ["y"]})
    with pytest.raises(PlanError, match=r"nope.*available.*'a'.*'b'"):
        _render("{nope}", batch)
