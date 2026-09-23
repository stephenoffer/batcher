"""Ranking, clustering, and model-comparison metrics from ``batcher.ml.metrics``.

These are Dataset functions rather than ``bt.*`` aggregate expressions, because each needs
more than one pass: a ranking metric sorts within every query, a clustering score reads the
contingency table of two labelings, and ``compare_models`` scores several prediction
columns side by side. Every value below is checked against a hand computation on data small
enough to verify on paper.

    python examples/metrics/ranking_and_clustering.py
"""

from __future__ import annotations

import math

import batcher as bt
from batcher.ml.metrics import (
    adjusted_rand_score,
    average_precision,
    compare_models,
    homogeneity_score,
    map_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    normalized_mutual_info_score,
    precision_at_k,
    rand_score,
)


def _dcg(gains: list[float]) -> float:
    """Discounted cumulative gain of a ranked gain list: gain / log2(rank + 1)."""
    return sum(g / math.log2(rank + 1) for rank, g in enumerate(gains, 1))


def ranking() -> None:
    # Two users. Alice's list puts a relevant item first; Bob's buries his only one at 3.
    # Carol has nothing relevant: she scores 0 on every metric and stays in the mean.
    recs = bt.from_pydict(
        {
            "user": ["alice"] * 4 + ["bob"] * 3 + ["carol"] * 2,
            "score": [0.9, 0.8, 0.7, 0.1, 0.9, 0.5, 0.2, 0.6, 0.4],
            "clicked": [1, 0, 1, 0, 0, 0, 1, 0, 0],
            "rating": [3, 0, 1, 0, 0, 0, 2, 0, 0],
        }
    )

    p2 = precision_at_k(recs, "user", "score", "clicked", k=2)
    # alice: 1 of the top 2; bob: 0 of 2; carol: 0 of 2.
    assert abs(p2 - (0.5 + 0.0 + 0.0) / 3) < 1e-12

    mrr = mean_reciprocal_rank(recs, "user", "score", "clicked")
    assert abs(mrr - (1 / 1 + 1 / 3 + 0.0) / 3) < 1e-12

    ndcg = ndcg_at_k(recs, "user", "score", "clicked", k=3)
    alice = _dcg([1, 0, 1]) / _dcg([1, 1])
    bob = _dcg([0, 0, 1]) / _dcg([1])
    assert abs(ndcg - (alice + bob + 0.0) / 3) < 1e-12

    # With graded=True the label is the gain itself, as in scikit-learn's ndcg_score.
    graded = ndcg_at_k(recs, "user", "score", "rating", k=3, graded=True)
    alice = _dcg([3, 0, 1]) / _dcg([3, 1])
    bob = _dcg([0, 0, 2]) / _dcg([2])
    assert abs(graded - (alice + bob + 0.0) / 3) < 1e-12

    mean_ap = map_at_k(recs, "user", "score", "clicked", k=3)
    # alice: precision 1/1 at rank 1 and 2/3 at rank 3, over min(k, 2) relevant items.
    alice_ap = (1 / 1 + 2 / 3) / 2
    bob_ap = (1 / 3) / 1
    assert abs(mean_ap - (alice_ap + bob_ap + 0.0) / 3) < 1e-12
    print(f"precision@2={p2:.4f} mrr={mrr:.4f} ndcg@3={ndcg:.4f} graded={graded:.4f}")
    print(f"map@3={mean_ap:.4f}")

    # Ties never favour the relevant item: four tied candidates with one relevant score
    # 1/4 at k=1, the value averaged over every order the tie allows.
    tied = bt.from_pydict({"q": [0] * 4, "s": [0.5] * 4, "y": [1, 0, 0, 0]})
    assert ndcg_at_k(tied, "q", "s", "y", k=1) == 0.25

    # Average precision is the global (not per-query) area under the precision/recall
    # curve: the mean precision at each positive, walking the scores downward.
    scored = bt.from_pydict({"y": [1, 0, 1, 0], "s": [0.9, 0.8, 0.7, 0.1]})
    ap = average_precision(scored, "y", "s")
    assert abs(ap - (1 / 1 + 2 / 3) / 2) < 1e-12
    print(f"average_precision={ap:.4f}")


def clustering() -> None:
    # Two labelings of six points. The prediction splits true cluster "b" in two.
    points = bt.from_pydict(
        {"truth": ["a", "a", "b", "b", "b", "b"], "cluster": [0, 0, 1, 1, 2, 2]}
    )
    rand = rand_score(points, "truth", "cluster")
    # Of the 15 point pairs, the labelings disagree on the 4 "b" pairs split across 1 and 2.
    assert abs(rand - 11 / 15) < 1e-12
    homogeneity = homogeneity_score(points, "truth", "cluster")
    assert abs(homogeneity - 1.0) < 1e-12  # every predicted cluster holds one true class
    ari = adjusted_rand_score(points, "truth", "cluster")
    nmi = normalized_mutual_info_score(points, "truth", "cluster")
    assert 0.0 < ari < 1.0 and 0.0 < nmi < 1.0
    print(f"rand={rand:.4f} homogeneity={homogeneity:.4f} ari={ari:.4f} nmi={nmi:.4f}")


def comparison() -> None:
    # Two models' probabilities for the same labels, scored in one pass.
    preds = bt.from_pydict(
        {
            "y": [1, 0, 1, 0, 1, 0],
            "good": [0.9, 0.2, 0.7, 0.4, 0.6, 0.1],
            "coin": [0.6, 0.6, 0.4, 0.4, 0.6, 0.4],
        }
    )
    table = compare_models(preds, "y", {"good": "good", "coin": "coin"})
    rows = table.to_pydict()
    print(rows)
    at = {model: i for i, model in enumerate(rows["model"])}
    # At the default 0.5 threshold "good" is right on all six rows; "coin" on four.
    assert rows["accuracy"][at["good"]] == 1.0
    assert abs(rows["accuracy"][at["coin"]] - 4 / 6) < 1e-12
    # The default set carries the threshold and probability metrics, not the rank metrics,
    # which each need their own sort.
    assert "log_loss" in rows and "roc_auc" not in rows


def main() -> None:
    ranking()
    clustering()
    comparison()


if __name__ == "__main__":
    main()
