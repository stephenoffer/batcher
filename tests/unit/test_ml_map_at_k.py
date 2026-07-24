"""Mean average precision at k.

MAP@k is rank-aware, so a hand reference computes it position by position and the engine's
windowed version is checked against it on random queries. The defining properties are pinned
too: a perfect ranking scores 1, order matters (unlike precision@k), and the denominator caps
at the query's relevant count.
"""

from __future__ import annotations

import numpy as np
import pytest

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.ml.metrics import map_at_k, precision_at_k

pytestmark = pytest.mark.unit


def _apk(scores: np.ndarray, rels: np.ndarray, k: int) -> float:
    order = np.argsort(-scores, kind="stable")
    ranked = rels[order]
    total_relevant = int((rels > 0).sum())
    cum = 0
    score = 0.0
    for i, rel in enumerate(ranked[:k], start=1):
        if rel > 0:
            cum += 1
            score += cum / i
    denom = min(k, total_relevant)
    return score / denom if denom > 0 else 0.0


def test_matches_the_position_by_position_reference() -> None:
    rng = np.random.default_rng(0)
    rows: dict[str, list] = {"q": [], "s": [], "rel": []}
    refs = []
    for q in range(40):
        n = int(rng.integers(5, 15))
        scores = rng.random(n)
        rels = (rng.random(n) < 0.3).astype(int)
        rows["q"] += [q] * n
        rows["s"] += scores.tolist()
        rows["rel"] += rels.tolist()
        refs.append(_apk(scores, rels, 5))
    ds = bt.from_pydict(rows)
    assert map_at_k(ds, "q", "s", "rel", k=5) == pytest.approx(float(np.mean(refs)))


def test_a_perfect_ranking_scores_one() -> None:
    ds = bt.from_pydict({"q": ["a"] * 4, "s": [0.9, 0.8, 0.7, 0.6], "rel": [1, 1, 0, 0]})
    assert map_at_k(ds, "q", "s", "rel", k=4) == pytest.approx(1.0)


def test_order_matters_unlike_precision_at_k() -> None:
    # Same top-k set, different order: precision@k is identical, MAP@k is not.
    good = bt.from_pydict({"q": ["a"] * 4, "s": [0.9, 0.8, 0.2, 0.1], "rel": [1, 0, 1, 0]})
    bad = bt.from_pydict({"q": ["a"] * 4, "s": [0.9, 0.8, 0.2, 0.1], "rel": [0, 1, 0, 1]})
    assert precision_at_k(good, "q", "s", "rel", k=4) == precision_at_k(bad, "q", "s", "rel", k=4)
    assert map_at_k(good, "q", "s", "rel", k=4) > map_at_k(bad, "q", "s", "rel", k=4)


def test_rejects_a_bad_k() -> None:
    ds = bt.from_pydict({"q": ["a"], "s": [0.5], "rel": [1]})
    with pytest.raises(PlanError, match="k must be at least 1"):
        map_at_k(ds, "q", "s", "rel", k=0)


def test_averages_over_queries_not_rows() -> None:
    # One easy query, one hard: the score is the mean of the two per-query APs, not pooled.
    ds = bt.from_pydict(
        {
            "q": ["a", "a", "b", "b"],
            "s": [0.9, 0.1, 0.9, 0.1],
            "rel": [1, 0, 0, 1],
        }
    )
    # AP(a) = 1.0 (relevant at rank 1); AP(b) = 0.5 (relevant at rank 2). Mean = 0.75.
    assert map_at_k(ds, "q", "s", "rel", k=2) == pytest.approx(0.75)
