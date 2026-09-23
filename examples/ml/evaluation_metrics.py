"""Scoring a model: the metrics, and why accuracy alone is a trap.

On an imbalanced problem a model that always predicts the majority class scores well on
accuracy and is useless. Precision and recall are what separate the two, ROC AUC scores the
ranking a threshold throws away, and ``ds.ml.evaluate`` reports the whole set in one call.

Every number below is checked against the same quantity computed by hand from the
confusion counts or from the pairwise definition of AUC, so the script fails loudly if a
metric ever drifts from its definition.

    python examples/ml/evaluation_metrics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import batcher as bt
from _common import tpch
from batcher import col
from batcher.ml.metrics import roc_auc


def _hand_counts(labels: np.ndarray, predicted: np.ndarray) -> tuple[int, int, int, int]:
    """The four confusion cells, counted in NumPy as the reference."""
    tp = int(np.sum(labels & predicted))
    fp = int(np.sum(~labels & predicted))
    fn = int(np.sum(labels & ~predicted))
    tn = int(np.sum(~labels & ~predicted))
    return tp, fp, fn, tn


def _hand_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """ROC AUC from its definition: P(a random positive outscores a random negative)."""
    positive, negative = np.sort(scores[labels]), scores[~labels]
    below = np.searchsorted(positive, negative, side="right")
    ties = below - np.searchsorted(positive, negative, side="left")
    wins = (len(positive) - below).sum() + 0.5 * ties.sum()
    return float(wins) / (len(positive) * len(negative))


def main() -> None:
    # A deliberately imbalanced label: only the very largest orders are "high value".
    orders = tpch("orders").select("o_orderkey", "o_totalprice")
    threshold = orders.agg(t=bt.quantile(col("o_totalprice"), 0.95)).to_pydict()["t"][0]
    # A noisy score: the price, jittered by a deterministic function of the key, so the
    # ranking is good but not perfect.
    jitter = ((col("o_orderkey") % 97).cast("float64") - 48.0) * (threshold / 2000.0)
    labelled = orders.with_columns(
        actual=col("o_totalprice") > threshold,
        score=col("o_totalprice") + jitter,
    ).with_columns(
        lazy=bt.lit(False),  # always predict the majority class
        real=col("score") >= threshold,  # a real, slightly noisy model
    )

    # One aggregate pass computes every metric, for both models.
    metrics = labelled.agg(
        lazy_accuracy=bt.accuracy("actual", "lazy"),
        lazy_recall=bt.recall("actual", "lazy", positive=True),
        real_accuracy=bt.accuracy("actual", "real"),
        real_precision=bt.precision("actual", "real", positive=True),
        real_recall=bt.recall("actual", "real", positive=True),
        real_f1=bt.f1_score("actual", "real", positive=True),
    ).to_pydict()
    metrics = {name: values[0] for name, values in metrics.items()}
    for name, value in metrics.items():
        print(f"{name:<15} {value:.4f}")

    # The reference: the same metrics from confusion counts computed in NumPy.
    table = labelled.select("actual", "score", "real").to_pydict()
    labels = np.asarray(table["actual"], dtype=bool)
    predicted = np.asarray(table["real"], dtype=bool)
    tp, fp, fn, tn = _hand_counts(labels, predicted)
    assert metrics["real_accuracy"] == (tp + tn) / (tp + fp + fn + tn)
    assert metrics["real_precision"] == tp / (tp + fp)
    assert metrics["real_recall"] == tp / (tp + fn)
    assert abs(metrics["real_f1"] - 2 * tp / (2 * tp + fp + fn)) < 1e-12

    # The lazy model scores well on accuracy and finds nothing. Its recall is exactly 0,
    # and its precision is 0.0 by the zero-division convention: it predicted no positives.
    assert metrics["lazy_accuracy"] > 0.9
    assert metrics["lazy_recall"] == 0.0
    # The real one finds the positives, which accuracy alone would not have revealed.
    assert metrics["real_recall"] > 0.8 and metrics["real_precision"] > 0.5

    # ROC AUC scores the ranking itself, before any threshold is chosen. It lives in
    # batcher.ml.metrics because it needs a sort, which an aggregate cannot express.
    auc = roc_auc(labelled, "actual", "score", positive=True)
    print(f"roc_auc         {auc:.6f}")
    assert abs(auc - _hand_auc(labels, np.asarray(table["score"], dtype=float))) < 1e-9

    # ds.ml.evaluate runs a task's whole metric set in one call. With only a score column,
    # the hard predictions are derived at `threshold`, so AUC and F1 come from one input.
    report = labelled.ml.evaluate(
        "actual",
        y_score="score",
        positive=True,
        threshold=threshold,
        metrics=["accuracy", "precision", "recall", "f1", "roc_auc"],
    )
    print(report)
    assert report["accuracy"] == metrics["real_accuracy"]
    assert report["precision"] == metrics["real_precision"]
    assert report["recall"] == metrics["real_recall"]
    assert report["roc_auc"] == auc


if __name__ == "__main__":
    main()
