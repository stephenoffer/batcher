"""Classification metrics computed as aggregates over a predictions table.

These are aggregate expressions, so evaluation is a ``select`` (or a ``group_by`` if you
want the metric per segment) rather than a pull into pandas. On a table too big for memory
that difference is the whole ballgame.

    python examples/metrics/classification.py
"""

from __future__ import annotations

import batcher as bt


def main() -> None:
    # 10 rows with a known confusion matrix: TP=3, FP=2, FN=1, TN=4.
    preds = bt.from_pydict(
        {
            "segment": ["a", "a", "a", "a", "a", "b", "b", "b", "b", "b"],
            "y_true": [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            "y_pred": [1, 1, 1, 0, 1, 1, 0, 0, 0, 0],
        }
    )

    scores = preds.select(
        tp=bt.true_positives("y_true", "y_pred"),
        fp=bt.false_positives("y_true", "y_pred"),
        fn=bt.false_negatives("y_true", "y_pred"),
        tn=bt.true_negatives("y_true", "y_pred"),
        accuracy=bt.accuracy("y_true", "y_pred"),
        precision=bt.precision("y_true", "y_pred"),
        recall=bt.recall("y_true", "y_pred"),
        f1=bt.f1_score("y_true", "y_pred"),
        # Weight recall twice as heavily as precision.
        f2=bt.fbeta_score("y_true", "y_pred", beta=2.0),
        specificity=bt.specificity("y_true", "y_pred"),
        balanced=bt.balanced_accuracy("y_true", "y_pred"),
        npv=bt.negative_predictive_value("y_true", "y_pred"),
        fpr=bt.false_positive_rate("y_true", "y_pred"),
        fnr=bt.false_negative_rate("y_true", "y_pred"),
        mcc=bt.matthews_corrcoef("y_true", "y_pred"),
        kappa=bt.cohen_kappa("y_true", "y_pred"),
        prevalence=bt.prevalence("y_true"),
    ).to_pydict()

    print(scores)

    assert scores["tp"] == [3]
    assert scores["fp"] == [2]
    assert scores["fn"] == [1]
    assert scores["tn"] == [4]
    # Seven of the ten predictions are correct.
    assert scores["accuracy"] == [0.7]
    # Precision is three true positives over five positive calls.
    assert scores["precision"] == [0.6]
    # Recall is three true positives over four actual positives.
    assert scores["recall"] == [0.75]
    assert abs(scores["f1"][0] - 2 * 0.6 * 0.75 / (0.6 + 0.75)) < 1e-12
    # F2 leans toward recall, so it sits above F1 here.
    assert scores["f2"][0] > scores["f1"][0]
    # Specificity is four true negatives over six actual negatives.
    assert abs(scores["specificity"][0] - 4 / 6) < 1e-12
    assert abs(scores["fpr"][0] - 2 / 6) < 1e-12
    assert abs(scores["fnr"][0] - 1 / 4) < 1e-12
    assert scores["prevalence"] == [0.4]

    # The reason these are aggregates: one metric per segment, in one pass.
    per_segment = (
        preds.group_by("segment")
        .agg(accuracy=bt.accuracy("y_true", "y_pred"), n=bt.count())
        .sort("segment")
        .to_pydict()
    )
    print(per_segment)
    assert per_segment["segment"] == ["a", "b"]
    assert per_segment["n"] == [5, 5]


if __name__ == "__main__":
    main()
