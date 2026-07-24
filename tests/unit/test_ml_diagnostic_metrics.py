"""Diagnostic-test classification metrics — the epidemiology vocabulary of the confusion matrix.

Each metric is pinned two ways: against scikit-learn where it has the same definition, and
against the algebraic identity that relates it to the metrics already trusted. The identities
are the stronger check — informedness *is* Youden's J, markedness and informedness compose to
the Matthews correlation coefficient, a false-discovery rate *is* one minus precision — so a
sign error or an inverted ratio cannot hide.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import batcher as bt

pytestmark = pytest.mark.unit

skm = pytest.importorskip("sklearn.metrics", reason="scikit-learn is the metric oracle")


@pytest.fixture(scope="module")
def labels() -> tuple[np.ndarray, np.ndarray, bt.Dataset]:
    """A binary problem with all four confusion cells non-empty."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 600)
    p = (rng.random(600) < 0.4 + 0.3 * y).astype(int)
    return y, p, bt.from_pydict({"y": y.tolist(), "p": p.tolist()})


def _agg(ds: bt.Dataset, expr) -> float:
    return ds.agg(m=expr).collect().column("m")[0].as_py()


def test_jaccard_matches_sklearn(labels) -> None:
    y, p, ds = labels
    assert _agg(ds, bt.jaccard_score("y", "p")) == pytest.approx(skm.jaccard_score(y, p))


def test_false_discovery_rate_is_one_minus_precision(labels) -> None:
    y, p, ds = labels
    got = _agg(ds, bt.false_discovery_rate("y", "p"))
    assert got == pytest.approx(1.0 - skm.precision_score(y, p))


def test_false_omission_rate_is_one_minus_npv(labels) -> None:
    y, p, ds = labels
    fdr = _agg(ds, bt.false_omission_rate("y", "p"))
    # NPV = tn / (tn + fn); false-omission rate is its complement.
    tn = int(((y == 0) & (p == 0)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    assert fdr == pytest.approx(1.0 - tn / (tn + fn))


def test_informedness_is_youdens_j(labels) -> None:
    y, p, ds = labels
    got = _agg(ds, bt.informedness("y", "p"))
    recall = skm.recall_score(y, p)
    specificity = skm.recall_score(y, p, pos_label=0)
    assert got == pytest.approx(recall + specificity - 1.0)


def test_markedness_and_informedness_compose_to_mcc(labels) -> None:
    # The defining identity: MCC = sign * sqrt(informedness * markedness).
    y, p, ds = labels
    row = ds.agg(inf=bt.informedness("y", "p"), mk=bt.markedness("y", "p")).to_pydict()
    product = row["inf"][0] * row["mk"][0]
    reconstructed = math.copysign(math.sqrt(abs(product)), row["inf"][0])
    assert reconstructed == pytest.approx(skm.matthews_corrcoef(y, p))


def test_positive_likelihood_ratio_is_sensitivity_over_one_minus_specificity(labels) -> None:
    y, p, ds = labels
    got = _agg(ds, bt.positive_likelihood_ratio("y", "p"))
    sensitivity = skm.recall_score(y, p)
    specificity = skm.recall_score(y, p, pos_label=0)
    assert got == pytest.approx(sensitivity / (1.0 - specificity))


def test_diagnostic_odds_ratio_is_the_ratio_of_likelihood_ratios(labels) -> None:
    _, _, ds = labels
    row = ds.agg(
        dor=bt.diagnostic_odds_ratio("y", "p"),
        plr=bt.positive_likelihood_ratio("y", "p"),
        nlr=bt.negative_likelihood_ratio("y", "p"),
    ).to_pydict()
    assert row["dor"][0] == pytest.approx(row["plr"][0] / row["nlr"][0])


def test_fowlkes_mallows_is_the_geometric_mean_of_precision_and_recall(labels) -> None:
    y, p, ds = labels
    got = _agg(ds, bt.fowlkes_mallows_index("y", "p"))
    assert got == pytest.approx(math.sqrt(skm.precision_score(y, p) * skm.recall_score(y, p)))


def test_a_perfect_classifier_maxes_the_diagnostic_metrics() -> None:
    ds = bt.from_pydict({"y": [1, 1, 0, 0], "p": [1, 1, 0, 0]})
    assert _agg(ds, bt.jaccard_score("y", "p")) == pytest.approx(1.0)
    assert _agg(ds, bt.informedness("y", "p")) == pytest.approx(1.0)
    assert _agg(ds, bt.false_discovery_rate("y", "p")) == pytest.approx(0.0)


def test_a_chance_classifier_has_zero_informedness() -> None:
    # Predicting all-positive: sensitivity 1, specificity 0, so informedness 0.
    ds = bt.from_pydict({"y": [1, 0, 1, 0], "p": [1, 1, 1, 1]})
    assert _agg(ds, bt.informedness("y", "p")) == pytest.approx(0.0)


def test_prevalence_threshold_is_in_the_unit_interval(labels) -> None:
    _, _, ds = labels
    value = _agg(ds, bt.prevalence_threshold("y", "p"))
    assert 0.0 <= value <= 1.0


def test_the_diagnostic_metrics_are_reachable_from_the_top_level() -> None:
    for name in (
        "jaccard_score",
        "false_discovery_rate",
        "false_omission_rate",
        "positive_likelihood_ratio",
        "negative_likelihood_ratio",
        "diagnostic_odds_ratio",
        "informedness",
        "markedness",
        "fowlkes_mallows_index",
        "prevalence_threshold",
    ):
        assert hasattr(bt, name), name
