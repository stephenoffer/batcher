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
