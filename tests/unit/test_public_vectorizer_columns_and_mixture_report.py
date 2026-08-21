"""Vectorizer output-column names and the corpus-mixture report, neither of which was tested.

Three small pieces of public surface, each of which is a *name* or a *number* a caller has
to build the next stage around:

* ``CountVectorizer.indices_column`` / ``values_column`` (and the same pair on
  ``HashingVectorizer``) tell a caller what the sparse output is called before the
  transform runs, so a downstream model can be wired up without guessing. If they
  disagreed with what ``transform`` actually emits, every pipeline built on them breaks at
  the point the guess is used rather than at the point it was made.
* ``MixtureReport`` records what ``mix_corpora`` really drew from each source. Its whole
  reason to exist is that a shortfall should be visible rather than inferred from a loss
  curve, so a report that under-reported would defeat the feature.

Both accessors are checked against the transform they describe, not against a literal, so
a change to the naming scheme has to change both or fail here.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher.ml import CountVectorizer, HashingVectorizer, MixtureReport, mix_corpora

pytestmark = pytest.mark.unit

DOCS = ["a cat sat on the mat", "the cat ran fast", "dogs run and dogs bark", "cat cat cat"]


@pytest.fixture
def corpus():
    return bt.from_pydict({"txt": DOCS})


VECTORIZERS = [CountVectorizer, HashingVectorizer]


@pytest.mark.parametrize("cls", VECTORIZERS)
def test_the_declared_column_names_are_the_ones_the_transform_emits(cls, corpus):
    """The promise and the delivery, checked against each other rather than against a literal."""
    vectorizer = cls("txt")
    fitted = vectorizer.fit(corpus)
    produced = set(fitted.transform(corpus).schema.names)
    assert vectorizer.indices_column in produced, (
        f"{cls.__name__} promised {vectorizer.indices_column}, which the transform did not emit"
    )
    assert vectorizer.values_column in produced
    assert vectorizer.indices_column != vectorizer.values_column


@pytest.mark.parametrize("cls", VECTORIZERS)
def test_the_column_names_follow_the_output_column(cls, corpus):
    """Renaming the output must rename both halves of the sparse pair, not just one."""
    vectorizer = cls("txt", output_column="bow")
    assert vectorizer.indices_column == "bow_indices"
    assert vectorizer.values_column == "bow_values"
    produced = set(vectorizer.fit(corpus).transform(corpus).schema.names)
    assert {"bow_indices", "bow_values"} <= produced


@pytest.mark.parametrize("cls", VECTORIZERS)
def test_the_default_output_column_is_features(cls):
    """The default a caller gets when they say nothing, pinned because pipelines rely on it."""
    vectorizer = cls("txt")
    assert vectorizer.output_column == "features"
    assert vectorizer.indices_column == "features_indices"
    assert vectorizer.values_column == "features_values"


def test_the_sparse_pair_lines_up_row_for_row(corpus):
    """Indices and values must be the same length per row, or the vector is meaningless."""
    out = CountVectorizer("txt").fit(corpus).transform(corpus).to_pydict()
    indices = out["features_indices"]
    values = out["features_values"]
    assert len(indices) == len(DOCS)
    for row_indices, row_values in zip(indices, values, strict=True):
        assert len(row_indices) == len(row_values), "an index with no value, or the reverse"
        assert len(set(row_indices)) == len(row_indices), "a repeated index in one row"
        assert all(v > 0 for v in row_values), "a zero has no place in a sparse vector"


def test_a_dense_vectorizer_emits_the_output_column_itself(corpus):
    """Under ``dense=True`` there is no index column, which the docstring states."""
    vectorizer = CountVectorizer("txt", dense=True)
    produced = set(vectorizer.fit(corpus).transform(corpus).schema.names)
    assert vectorizer.output_column in produced
    assert vectorizer.indices_column not in produced, (
        "a dense vectorizer emits no index column, so promising one would mislead"
    )


def test_the_hashing_vectorizer_needs_no_vocabulary(corpus):
    """It must transform text it never saw, which is the reason to choose it."""
    vectorizer = HashingVectorizer("txt")
    unseen = bt.from_pydict({"txt": ["words that were never fitted on"]})
    out = vectorizer.fit(corpus).transform(unseen).to_pydict()
    assert len(out[vectorizer.indices_column]) == 1
    assert out[vectorizer.indices_column][0], "an unseen document produced no features"


def test_mixture_report_records_what_was_asked_for_and_what_was_taken():
    """The three maps, over a mixture every source can satisfy."""
    web = bt.from_pydict({"text": ["w"] * 100})
    code = bt.from_pydict({"text": ["c"] * 100})
    mixed, report = mix_corpora({"web": web, "code": code}, {"web": 0.75, "code": 0.25})

    assert isinstance(report, MixtureReport)
    assert set(report.requested) == {"web", "code"}
    assert set(report.available) == {"web", "code"}
    assert set(report.taken) == {"web", "code"}
    assert report.available == {"web": 100, "code": 100}
    assert report.shortfalls == {}, "both sources can supply what was asked of them"
    assert sum(report.taken.values()) == mixed.count()
    assert report.taken["web"] > report.taken["code"], "the weights must reach the draw"


def test_the_default_mixture_shrinks_to_what_the_smallest_source_can_fill():
    """No ``total_rows`` means the largest mixture that repeats nothing, so no shortfall.

    A thousand rows against five, at even weights, gives five and five -- not a thousand
    and five up-sampled to a thousand. That is the documented default, and it is why the
    report is empty here: the shortfall was avoided rather than tolerated.
    """
    big = bt.from_pydict({"text": ["w"] * 1000})
    tiny = bt.from_pydict({"text": ["c"] * 5})
    mixed, report = mix_corpora({"big": big, "tiny": tiny}, {"big": 0.5, "tiny": 0.5})

    assert report.available == {"big": 1000, "tiny": 5}
    assert report.taken == {"big": 5, "tiny": 5}
    assert report.shortfalls == {}, "shrinking the mixture is not a shortfall"
    assert report.realized_weights == {"big": 0.5, "tiny": 0.5}
    assert mixed.count() == 10

    rows = mixed.to_pydict()["text"]
    assert rows.count("c") == 5, "the small source must not be repeated to fill the weight"


def test_an_explicit_total_reports_the_source_that_could_not_fill_its_share():
    """Asking for more than a source holds is where a shortfall becomes real and visible.

    This is what the report exists for: the shortfall is named and the realized weights say
    what the mixture actually is, rather than leaving it to be inferred from a loss curve.
    """
    big = bt.from_pydict({"text": ["w"] * 1000})
    tiny = bt.from_pydict({"text": ["c"] * 5})
    mixed, report = mix_corpora(
        {"big": big, "tiny": tiny}, {"big": 0.5, "tiny": 0.5}, total_rows=500
    )

    assert report.requested["tiny"] == 250
    assert report.taken["tiny"] == 5, "no source may yield more rows than it has"
    assert report.shortfalls == {"tiny": 245}, "the missing rows are counted, not just flagged"
    assert sum(report.taken.values()) == mixed.count()

    assert report.realized_weights["tiny"] < 0.5, (
        "the realized weight must show the mixture is not the one that was asked for"
    )
    assert sum(report.realized_weights.values()) == pytest.approx(1.0)
    assert mixed.to_pydict()["text"].count("c") == 5, "the shortfall is not filled by repeats"


def test_the_report_accounts_for_every_row_the_mixture_holds():
    """``taken`` is a claim about the output, so it has to match the output."""
    sources = {name: bt.from_pydict({"text": [name] * 60}) for name in ("a", "b", "c")}
    mixed, report = mix_corpora(sources, {"a": 0.5, "b": 0.3, "c": 0.2})
    rows = mixed.to_pydict()["text"]
    counted = {name: rows.count(name) for name in sources}
    assert counted == report.taken, f"the report says {report.taken}, the data says {counted}"
    assert sum(counted.values()) == len(rows)
