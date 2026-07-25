"""Univariate feature scores — checked against scikit-learn where a match exists.

`f_classif_scores` and `f_regression_scores` reproduce scikit-learn's `f_classif` and
`f_regression` exactly, because those are the functions users will compare against. The chi2
and mutual-information scores are pinned to the ordering that makes them useful (a linked
column outscores an independent one), and `select_k_best` to its structural contract.
"""

from __future__ import annotations

import numpy as np
import pytest

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.ml.feature_scores import (
    chi2_scores,
    f_classif_scores,
    f_regression_scores,
    mutual_info_scores,
    select_k_best,
)

pytestmark = pytest.mark.unit

sk_fs = pytest.importorskip("sklearn.feature_selection")


def test_f_classif_matches_sklearn() -> None:
    rng = np.random.default_rng(0)
    y = np.array(["a"] * 100 + ["b"] * 100 + ["c"] * 100)
    f1 = np.concatenate([rng.normal(0, 1, 100), rng.normal(1, 1, 100), rng.normal(2, 1, 100)])
    f2 = rng.normal(0, 1, 300)
    ds = bt.from_pydict({"y": y.tolist(), "f1": f1.tolist(), "f2": f2.tolist()})
    got = f_classif_scores(ds, "y")
    expected, _ = sk_fs.f_classif(np.c_[f1, f2], y)
    assert got["f1"] == pytest.approx(expected[0], abs=1e-6)
    assert got["f2"] == pytest.approx(expected[1], abs=1e-6)


def test_f_regression_matches_sklearn() -> None:
    rng = np.random.default_rng(1)
    y = rng.normal(0, 1, 300)
    f1 = 2 * y + rng.normal(0, 0.5, 300)
    f2 = rng.normal(0, 1, 300)
    ds = bt.from_pydict({"y": y.tolist(), "f1": f1.tolist(), "f2": f2.tolist()})
    got = f_regression_scores(ds, "y")
    expected, _ = sk_fs.f_regression(np.c_[f1, f2], y)
    assert got["f1"] == pytest.approx(expected[0], abs=1e-4)
    assert got["f2"] == pytest.approx(expected[1], abs=1e-4)


def test_f_classif_defaults_to_every_non_target_column() -> None:
    ds = bt.from_pydict({"y": ["a", "b"], "a": [1.0, 2.0], "b": [3.0, 4.0]})
    assert set(f_classif_scores(ds, "y")) == {"a", "b"}


def test_chi2_ranks_a_linked_column_first() -> None:
    ds = bt.from_pydict(
        {"y": ["p", "p", "q", "q"], "linked": ["a", "a", "b", "b"], "free": ["a", "b", "a", "b"]}
    )
    scores = chi2_scores(ds, "y")
    assert scores["linked"] > scores["free"]


def test_mutual_info_ranks_a_copy_first() -> None:
    ds = bt.from_pydict(
        {"y": ["a", "a", "b", "b"], "copy": ["a", "a", "b", "b"], "rand": ["a", "b", "a", "b"]}
    )
    scores = mutual_info_scores(ds, "y")
    assert scores["copy"] > scores["rand"]


def test_select_k_best_returns_the_top_k_by_score() -> None:
    assert select_k_best({"a": 10.0, "b": 1.0, "c": 5.0}, 2) == ["a", "c"]


def test_select_k_best_drops_nan_scores_last() -> None:
    # A degenerate fit (a constant feature) scores NaN and must never crowd out a real score.
    assert select_k_best({"a": float("nan"), "b": 3.0}, 1) == ["b"]


def test_scores_pipe_into_selection() -> None:
    rng = np.random.default_rng(2)
    y = np.array(["a"] * 50 + ["b"] * 50)
    strong = np.concatenate([rng.normal(0, 1, 50), rng.normal(5, 1, 50)])
    weak = rng.normal(0, 1, 100)
    ds = bt.from_pydict({"y": y.tolist(), "strong": strong.tolist(), "weak": weak.tolist()})
    assert select_k_best(f_classif_scores(ds, "y"), 1) == ["strong"]


# --------------------------------------------------------------------------- #
# A categorical scorer on a continuous column scores every feature identically
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("scorer", [chi2_scores, mutual_info_scores])
def test_a_categorical_scorer_refuses_an_all_distinct_column(scorer) -> None:
    """The score is pinned at its maximum, so it ranks unrelated features equally.

    A column with one level per row determines the target by construction: `chi2_scores`
    returned exactly the row count and `mutual_info_scores` exactly the target's entropy, the
    same for every such feature however unrelated. `select_k_best` then ranked features that all
    scored alike and returned whichever the dict ordered first. Nothing raised, so the caller saw
    a confident selection resting on no information.
    """
    rng = np.random.default_rng(1)
    values = np.abs(rng.normal(5.0, 2.0, 100))
    ds = bt.from_pydict({"a": values.tolist(), "y": (values > 5.0).astype(int).tolist()})

    with pytest.raises(PlanError, match="one distinct value per row"):
        scorer(ds, "y", ["a"])


@pytest.mark.parametrize("scorer", [chi2_scores, mutual_info_scores])
def test_the_message_points_at_a_scorer_that_does_take_continuous_features(scorer) -> None:
    rng = np.random.default_rng(2)
    values = rng.normal(0.0, 1.0, 40)
    ds = bt.from_pydict({"a": values.tolist(), "y": (values > 0).astype(int).tolist()})

    with pytest.raises(PlanError) as excinfo:
        scorer(ds, "y", ["a"])
    message = str(excinfo.value)
    assert "f_classif_scores" in message and "f_regression_scores" in message
    assert "'a'" in message, "the message names the offending column"


def test_the_continuous_scorers_still_accept_a_continuous_column() -> None:
    """The guard belongs to the categorical scorers only."""
    rng = np.random.default_rng(3)
    values = rng.normal(0.0, 1.0, 60)
    ds = bt.from_pydict(
        {
            "a": values.tolist(),
            "cls": (values > 0).astype(int).tolist(),
            "reg": (2.0 * values).tolist(),
        }
    )
    assert f_classif_scores(ds, "cls", ["a"])["a"] > 0
    assert f_regression_scores(ds, "reg", ["a"])["a"] > 0


def test_a_repeated_categorical_column_is_accepted() -> None:
    """Distinct-per-row is the rejection rule, not "numeric" — a coded category is fine."""
    ds = bt.from_pydict(
        {"y": [0, 0, 1, 1, 0, 1], "coded": [1, 1, 2, 2, 1, 2], "free": [1, 2, 1, 2, 1, 2]}
    )
    assert chi2_scores(ds, "y")["coded"] > chi2_scores(ds, "y")["free"]
    assert mutual_info_scores(ds, "y")["coded"] > mutual_info_scores(ds, "y")["free"]


def test_a_tiny_input_is_not_rejected() -> None:
    """Below three rows every column is trivially all-distinct; that is not the bug."""
    ds = bt.from_pydict({"y": ["p", "q"], "a": ["x", "y"]})
    assert set(chi2_scores(ds, "y")) == {"a"}
