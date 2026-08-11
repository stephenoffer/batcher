"""Scoring a model: the metrics, and why accuracy alone is a trap.

On an imbalanced problem a model that always predicts the majority class scores well on
accuracy and is useless. Precision and recall are what separate the two, and they need the
confusion counts rather than a single number.

    python examples/ml/evaluation_metrics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    # A deliberately imbalanced label: only the very largest orders are "high value".
    orders = tpch("orders").select("o_orderkey", "o_totalprice")
    threshold = orders.agg(t=bt.quantile(col("o_totalprice"), 0.95)).to_pydict()["t"][0]
    labelled = orders.with_columns(actual=col("o_totalprice") > threshold)

    positives = labelled.agg(n=bt.count_if(col("actual"))).to_pydict()["n"][0]
    print(f"{positives} positives of {labelled.count()} ({positives / labelled.count():.2%})")

    # The lazy model: always predict the majority class.
    lazy = labelled.with_columns(predicted=bt.lit(False))

    # A real model: a slightly wrong threshold.
    real = labelled.with_columns(predicted=col("o_totalprice") > threshold * 0.98)

    def score(name: str, dataset: bt.Dataset) -> dict[str, float]:
        counts = dataset.agg(
            tp=bt.count_if(col("actual") & col("predicted")),
            fp=bt.count_if(~col("actual") & col("predicted")),
            fn=bt.count_if(col("actual") & ~col("predicted")),
            tn=bt.count_if(~col("actual") & ~col("predicted")),
        ).to_pydict()
        tp, fp, fn, tn = (counts[key][0] for key in ("tp", "fp", "fn", "tn"))
        accuracy = (tp + tn) / (tp + fp + fn + tn)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        print(f"{name:<6} accuracy={accuracy:.4f} precision={precision:.4f} recall={recall:.4f}")
        return {"accuracy": accuracy, "precision": precision, "recall": recall}

    lazy_scores = score("lazy", lazy)
    real_scores = score("real", real)

    # The lazy model scores well on accuracy and finds nothing.
    assert lazy_scores["accuracy"] > 0.9
    assert lazy_scores["recall"] == 0.0

    # The real one finds the positives, which accuracy alone would not have revealed.
    assert real_scores["recall"] > 0.9
    assert real_scores["precision"] > 0.5


if __name__ == "__main__":
    main()
