"""Mixing, filtering, and decontaminating a training corpus.

Each of these deletes or reweights data, so the tests are about the failure that would be
invisible: a mixture that silently follows the sources' sizes instead of the declared weights,
a filter that removes a corpus it was mis-tuned for, and a contamination check that either
misses a quotation or deletes ordinary English.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher._internal.errors import ColumnNotFoundError, PlanError
from batcher.ml import (
    QualityThresholds,
    contamination_rate,
    decontaminate,
    mix_corpora,
    quality_filter,
    quality_flags,
    quality_report,
)

pytestmark = pytest.mark.unit


def _corpus(tag: str, n: int) -> bt.Dataset:
    return bt.from_pydict({"text": [f"{tag} document"] * n})


# --- mixing ------------------------------------------------------------------------


def test_a_mixture_follows_the_declared_weights_not_the_source_sizes():
    """The whole point: two equal-sized sources at 3:1 must come out 3:1."""
    mixed, report = mix_corpora(
        {"web": _corpus("web", 800), "code": _corpus("code", 800)},
        {"web": 3, "code": 1},
        total_rows=400,
    )
    assert report.realized_weights == {"web": 0.75, "code": 0.25}
    assert mixed.count() == 400


def test_weights_are_normalized_so_any_positive_scale_works():
    counts = []
    for weights in ({"a": 3, "b": 1}, {"a": 0.75, "b": 0.25}, {"a": 30, "b": 10}):
        _, report = mix_corpora(
            {"a": _corpus("a", 400), "b": _corpus("b", 400)}, weights, total_rows=200
        )
        counts.append(report.realized_weights)
    assert counts[0] == counts[1] == counts[2]


def test_the_source_column_records_where_each_row_came_from():
    mixed, _ = mix_corpora(
        {"web": _corpus("web", 100), "code": _corpus("code", 100)},
        {"web": 1, "code": 1},
        total_rows=100,
    )
    assert sorted(set(mixed.select("source").to_pydict()["source"])) == ["code", "web"]


def test_the_source_column_can_be_omitted():
    mixed, _ = mix_corpora(
        {"a": _corpus("a", 50), "b": _corpus("b", 50)},
        {"a": 1, "b": 1},
        total_rows=20,
        source_column=None,
    )
    assert mixed.columns == ["text"]


def test_a_source_too_small_for_its_share_is_reported_not_upsampled():
    """Repeating rows changes a source's effective epoch count; that is a decision."""
    _, report = mix_corpora(
        {"big": _corpus("big", 100), "tiny": _corpus("tiny", 2)},
        {"big": 0.5, "tiny": 0.5},
        total_rows=100,
    )
    assert report.shortfalls == {"tiny": 48}
    assert report.taken["tiny"] == 2


def test_the_default_size_is_the_largest_mixture_needing_no_upsampling():
    mixed, report = mix_corpora(
        {"big": _corpus("big", 100), "tiny": _corpus("tiny", 2)},
        {"big": 0.5, "tiny": 0.5},
    )
    assert report.shortfalls == {}
    assert mixed.count() == 4  # two from each, the most `tiny` can support


def test_a_zero_weight_source_contributes_nothing():
    mixed, report = mix_corpora(
        {"keep": _corpus("keep", 100), "drop": _corpus("drop", 100)},
        {"keep": 1, "drop": 0},
        total_rows=50,
    )
    assert report.taken["drop"] == 0
    assert set(mixed.select("source").to_pydict()["source"]) == {"keep"}


def test_a_seed_makes_the_mixture_reproducible():
    def build():
        mixed, _ = mix_corpora(
            {"a": bt.from_pydict({"text": [str(i) for i in range(200)]})},
            {"a": 1},
            total_rows=20,
            seed=7,
        )
        return sorted(mixed.select("text").to_pydict()["text"])

    assert build() == build()


@pytest.mark.parametrize(
    ("sources", "weights"),
    [
        ({}, {}),
        ({"a": None}, {}),
        ({"a": None}, {"a": 1, "b": 1}),
        ({"a": None}, {"a": -1}),
        ({"a": None}, {"a": 0}),
    ],
)
def test_an_unbuildable_mixture_is_rejected_before_any_scan(sources, weights):
    filled = {name: _corpus(name, 10) for name in sources}
    with pytest.raises(PlanError):
        mix_corpora(filled, weights)


def test_a_non_positive_total_is_rejected():
    with pytest.raises(PlanError):
        mix_corpora({"a": _corpus("a", 10)}, {"a": 1}, total_rows=0)


# --- quality filtering -------------------------------------------------------------

_DOCS = [
    "A real sentence about something, written out properly.",  # prose
    "buy now!!!",  # too short, punctuation-heavy
    "1234 5678 9012 3456",  # digits
]


def test_the_filter_keeps_prose_and_drops_the_rest():
    docs = bt.from_pydict({"text": _DOCS})
    kept = quality_filter(docs, "text", QualityThresholds(min_words=3)).to_pydict()["text"]
    assert kept == [_DOCS[0]]


def test_the_report_shows_what_each_rule_would_remove():
    """Per-rule keep rates are how you notice a rule mis-tuned for your corpus."""
    docs = bt.from_pydict({"text": _DOCS})
    report = quality_report(docs, "text", QualityThresholds(min_words=3))
    assert report["digit_ratio"] == pytest.approx(2 / 3)
    assert report["all"] == pytest.approx(1 / 3)


def test_the_report_and_the_filter_agree():
    docs = bt.from_pydict({"text": _DOCS})
    thresholds = QualityThresholds(min_words=3)
    report = quality_report(docs, "text", thresholds)
    kept = quality_filter(docs, "text", thresholds).count()
    assert report["all"] == pytest.approx(kept / docs.count())


def test_a_null_document_is_dropped_rather_than_kept_by_default():
    """A row the filter cannot read is not a row to train on."""
    docs = bt.from_pydict({"text": [None, _DOCS[0]]})
    assert quality_filter(docs, "text", QualityThresholds(min_words=3)).count() == 1


def test_the_non_ascii_gate_is_off_by_default():
    """A multilingual corpus must survive the defaults."""
    docs = bt.from_pydict({"text": ["naïve café résumé written out at some length here."]})
    assert quality_filter(docs, "text", QualityThresholds(min_words=3)).count() == 1
    strict = QualityThresholds(min_words=3, max_non_ascii_ratio=0.01)
    assert quality_filter(docs, "text", strict).count() == 0


def test_the_terminal_punctuation_rule_can_be_turned_off():
    docs = bt.from_pydict({"text": ["a heading with no full stop"]})
    assert quality_filter(docs, "text", QualityThresholds(min_words=3)).count() == 0
    lenient = QualityThresholds(min_words=3, require_terminal_punctuation=False)
    assert quality_filter(docs, "text", lenient).count() == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_words": -1},
        {"min_words": 10, "max_words": 5},
        {"min_mean_word_length": 8.0, "max_mean_word_length": 3.0},
        {"max_punctuation_ratio": 1.5},
        {"max_digit_ratio": -0.1},
    ],
)
def test_impossible_thresholds_are_rejected(kwargs):
    with pytest.raises(PlanError):
        QualityThresholds(**kwargs)


def test_filtering_an_absent_column_names_the_ones_that_exist():
    with pytest.raises(ColumnNotFoundError):
        quality_filter(bt.from_pydict({"body": ["x"]}), "text")


# --- decontamination ---------------------------------------------------------------

_TRAIN = ["what is the capital of france", "an entirely unrelated document here"]
_EVAL = ["what is the capital of france"]


def test_a_quoted_eval_question_is_removed():
    train = bt.from_pydict({"text": _TRAIN})
    evals = bt.from_pydict({"text": _EVAL})
    assert decontaminate(train, "text", evals, n=4).to_pydict()["text"] == [_TRAIN[1]]


def test_the_contamination_rate_measures_before_removing():
    train = bt.from_pydict({"text": _TRAIN})
    evals = bt.from_pydict({"text": _EVAL})
    assert contamination_rate(train, "text", evals, n=4) == 0.5


def test_a_longer_span_matches_less():
    """`n` is the judgement: a long span catches quotations, a short one catches English."""
    train = bt.from_pydict({"text": ["the capital city is large and busy"]})
    evals = bt.from_pydict({"text": ["what is the capital city of france"]})
    assert contamination_rate(train, "text", evals, n=2) > 0
    assert contamination_rate(train, "text", evals, n=8) == 0


def test_a_reformatted_quotation_still_matches():
    """Normalization is why: casing and punctuation must not hide a copy."""
    train = bt.from_pydict({"text": ["What Is The Capital Of France?"]})
    evals = bt.from_pydict({"text": ["what is the capital of france"]})
    assert contamination_rate(train, "text", evals, n=4) == 1.0


def test_an_uncontaminated_corpus_is_returned_whole():
    train = bt.from_pydict({"text": ["completely different subject matter entirely"]})
    evals = bt.from_pydict({"text": ["what is the capital of france"]})
    assert decontaminate(train, "text", evals, n=4).count() == 1
    assert contamination_rate(train, "text", evals, n=4) == 0.0


def test_decontamination_keeps_the_original_columns():
    train = bt.from_pydict({"text": _TRAIN, "id": [1, 2]})
    evals = bt.from_pydict({"text": _EVAL})
    assert decontaminate(train, "text", evals, n=4).columns == ["text", "id"]


def test_the_eval_column_may_be_named_differently():
    train = bt.from_pydict({"text": _TRAIN})
    evals = bt.from_pydict({"question": _EVAL})
    clean = decontaminate(train, "text", evals, eval_column="question", n=4)
    assert clean.to_pydict()["text"] == [_TRAIN[1]]


def test_a_span_below_one_is_rejected():
    train = bt.from_pydict({"text": _TRAIN})
    evals = bt.from_pydict({"text": _EVAL})
    with pytest.raises(PlanError):
        decontaminate(train, "text", evals, n=0)
    with pytest.raises(PlanError):
        contamination_rate(train, "text", evals, n=0)


def test_an_empty_corpus_has_no_contamination():
    train = bt.from_pydict({"text": ["x"]}).filter(bt.col("text") == bt.lit("nothing"))
    evals = bt.from_pydict({"text": _EVAL})
    assert contamination_rate(train, "text", evals, n=4) == 0.0


def test_quality_flags_explains_which_rule_dropped_a_document():
    """`quality_filter` deletes and `quality_report` counts; neither says why *this* row went."""
    docs = bt.from_pydict({"text": _DOCS})
    flagged = quality_flags(docs, "text", QualityThresholds(min_words=3))
    got = flagged.select("min_words", "digit_ratio", "passes_all").to_pydict()
    # The prose row passes everything; the digit row fails only the digit rule.
    assert got["passes_all"] == [True, False, False]
    assert got["digit_ratio"] == [True, True, False]
    assert got["min_words"] == [True, False, True]


def test_the_flags_agree_with_the_filter_row_for_row():
    """A flag that disagreed with the filter would be worse than no flag at all."""
    docs = bt.from_pydict({"text": _DOCS, "id": [0, 1, 2]})
    thresholds = QualityThresholds(min_words=3)
    kept = set(quality_filter(docs, "text", thresholds).to_pydict()["id"])
    flagged = quality_flags(docs, "text", thresholds).to_pydict()
    passing = {i for i, ok in zip(flagged["id"], flagged["passes_all"], strict=True) if ok}
    assert kept == passing


def test_flagging_keeps_the_original_columns():
    docs = bt.from_pydict({"text": _DOCS, "id": [0, 1, 2]})
    flagged = quality_flags(docs, "text", QualityThresholds(min_words=3))
    assert flagged.columns[:2] == ["text", "id"]
    assert "passes_all" in flagged.columns


def test_a_null_document_fails_every_flag():
    """A row the rules cannot read is not a row to train on, and the flags have to say so."""
    docs = bt.from_pydict({"text": [None, _DOCS[0]]})
    got = quality_flags(docs, "text", QualityThresholds(min_words=3)).to_pydict()
    assert got["passes_all"] == [False, True]
    assert got["min_words"] == [False, True]
